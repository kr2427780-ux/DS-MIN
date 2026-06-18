import torch
import numpy as np
import os
import json
import time
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree

from model import DS_MIN
from train import train_one_fold
from tqdm import tqdm


# ==========================================
# 评估配置
# ==========================================
class EvalConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    d_k = 64
    num_layers = 3
    chunk_size = 8192
    num_votes = 5            # 测试时增强投票次数
    tau = 3                  # 边界带宽度 (x APS)
    aps = 0.05               # 平均点间距 (mm)
    boundary_tolerance = 3 * aps  # d = 0.15 mm，用于 mBIoU


# ==========================================
# mBIoU 计算 (论文 Section 4.2)
# ==========================================
def compute_miou_biou(pred_labels, gt_labels, points_xyz, d=0.15):
    """
    对单个样本同时计算 mIoU 和 mBIoU。
    实现论文 Section 4.2, Eq.(7)-(9)。
    """
    N = len(pred_labels)
    classes = sorted(set(list(gt_labels)))

    # 构建每类的点集
    gt_sets = {c: set(np.where(gt_labels == c)[0]) for c in classes}
    pred_sets = {c: set(np.where(pred_labels == c)[0]) for c in classes}

    # 标准 mIoU
    ious = {}
    for c in classes:
        gt_c = gt_sets[c]
        pred_c = pred_sets[c]
        inter = len(gt_c & pred_c)
        union = len(gt_c | pred_c)
        ious[c] = inter / union if union > 0 else 0.0
    miou = np.mean(list(ious.values()))

    # 边界 mIoU
    # 为每类构建 KD-Tree 以快速计算边界距离
    kdtrees_gt = {}
    for c in classes:
        idx_c = np.array(list(gt_sets[c]))
        if len(idx_c) > 0:
            kdtrees_gt[c] = cKDTree(points_xyz[idx_c])

    gt_boundary_sets = {}
    pred_boundary_sets = {}
    for c in classes:
        if c not in kdtrees_gt:
            gt_boundary_sets[c] = set()
            pred_boundary_sets[c] = set()
            continue

        # 类别 c 的真实边界
        gt_c_points = points_xyz[np.array(list(gt_sets[c]))]
        gt_c_idx = np.array(list(gt_sets[c]))
        boundary_mask_gt = np.zeros(len(gt_c_idx), dtype=bool)
        for other_c in classes:
            if other_c == c or other_c not in kdtrees_gt:
                continue
            dists, _ = kdtrees_gt[other_c].query(gt_c_points, k=1)
            boundary_mask_gt |= (dists <= d)
        gt_boundary_sets[c] = set(gt_c_idx[boundary_mask_gt])

        # 类别 c 的预测边界
        pred_c_idx = np.array(list(pred_sets[c]))
        if len(pred_c_idx) == 0:
            pred_boundary_sets[c] = set()
            continue
        pred_c_points = points_xyz[pred_c_idx]
        boundary_mask_pred = np.zeros(len(pred_c_idx), dtype=bool)
        for other_c in classes:
            if other_c == c or other_c not in kdtrees_gt:
                continue
            dists, _ = kdtrees_gt[other_c].query(pred_c_points, k=1)
            boundary_mask_pred |= (dists <= d)
        pred_boundary_sets[c] = set(pred_c_idx[boundary_mask_pred])

    # 计算每类 BIoU
    bious = {}
    for c in classes:
        gb = gt_boundary_sets[c]
        pb = pred_boundary_sets[c]
        inter = len(gb & pb)
        union = len(gb | pb)
        bious[c] = inter / union if union > 0 else 0.0
    mbiou = np.mean(list(bious.values()))
    return miou, mbiou, ious, bious


# ==========================================
# 单文件评估
# ==========================================
def evaluate_single(model, npy_path, img_path, roi_path, norm_path, compute_biou=True):
    """
    评估单个样本，可选计算 mBIoU。
    返回: miou, iou_conn, mbiou, conn_biou, per_class_biou
    """
    data = np.load(npy_path)
    # 数据结构: 0:3 XYZ, 3:6 Normal, 6:9 RGB, 9:11 UV, 11 dist, 12 label
    xyz = data[:, 0:3]
    normals = data[:, 3:6]
    uv = data[:, 9:11]
    labels = data[:, 12].astype(np.int64)
    distance_field = data[:, 11]
    num_points = int(xyz.shape[0])

    # 加载图像
    image = Image.open(img_path).convert("RGB")
    H, W = image.size[1], image.size[0]

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = img_transform(image).unsqueeze(0).to(EvalConfig.device)

    # 考核时也不预乘 RoI，保留原始 2D 物理光影属性
    roi_img = Image.open(roi_path).convert("L")

    # 使用全局归一化参数（与 training 一致）
    with open(norm_path, "r") as f:
        norm_params = json.load(f)
    centroid = np.array(norm_params["centroid"])
    max_dist = norm_params["max_dist"]
    xyz_norm = (xyz - centroid) / (max_dist + 1e-6)

    u_norm = (uv[:, 0] / (W - 1)) * 2.0 - 1.0
    v_norm = (uv[:, 1] / (H - 1)) * 2.0 - 1.0
    uv_norm = np.stack((u_norm, v_norm), axis=-1)

    # 测试时增强 (TTA) 投票
    vote_logits = np.zeros((num_points, EvalConfig.num_classes), dtype=np.float32)

    # ---- 计时 & 显存统计（Section 4.1 Hardware/Computational Cost）----
    sample_infer_times = []   # 每次 TTA 投票的推理耗时 (ms)
    peak_mem = 0.0           # 峰值显存 (GB)

    for _ in tqdm(range(EvalConfig.num_votes),
                  desc=f"TTA ({os.path.basename(npy_path)})", leave=False):
        logits_full = np.zeros((num_points, EvalConfig.num_classes), dtype=np.float32)

        for i in range(0, num_points, EvalConfig.chunk_size):
            end = min(i + EvalConfig.chunk_size, num_points)
            curr_size = end - i

            i_int, end_int, curr_int = int(i), int(end), int(curr_size)

            xyz_part = xyz_norm[i_int:end_int]
            normals_part = normals[i_int:end_int]
            chunk = np.hstack([xyz_part, normals_part])  # [curr, 6]
            chunk_uv = uv_norm[i_int:end_int]

            if curr_int < EvalConfig.chunk_size:
                pad_size = EvalConfig.chunk_size - curr_int
                pad_idx = np.random.choice(curr_int, pad_size, replace=True)
                chunk = np.vstack([chunk, chunk[pad_idx]])
                chunk_uv = np.vstack([chunk_uv, chunk_uv[pad_idx]])

            pts_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(EvalConfig.device)
            uv_tensor = torch.tensor(chunk_uv, dtype=torch.float32).unsqueeze(0).to(EvalConfig.device)

            if EvalConfig.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with torch.no_grad():
                logits = model(pts_tensor, img_tensor, uv_tensor)

            if EvalConfig.device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                sample_infer_times.append(start.elapsed_time(end))
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024**3)

            logits_chunk = logits.squeeze(0).cpu().numpy()
            logits_full[i_int:end_int] = logits_chunk[:curr_int]

        vote_logits += logits_full

    # 汇总计时统计
    if sample_infer_times:
        avg_chunk_ms = np.mean(sample_infer_times)
        total_ms = np.sum(sample_infer_times)
    else:
        avg_chunk_ms = total_ms = 0.0

    preds = np.argmax(vote_logits, axis=1)

    # 全局 mIoU
    ious = []
    for c in range(EvalConfig.num_classes):
        pred_mask = (preds == c).astype(np.int32)
        label_mask = (labels == c).astype(np.int32)
        inter = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        ious.append(inter / union if union > 0 else 0.0)
    miou = np.mean(ious)

    # 连接区域 IoU (类别 3)
    conn_class = 3
    pred_mask = (preds == conn_class)
    label_mask = (labels == conn_class)
    inter = np.logical_and(pred_mask, label_mask).sum()
    union = np.logical_or(pred_mask, label_mask).sum()
    iou_conn = inter / union if union > 0 else 0.0

    # 边界 mBIoU
    if compute_biou:
        _, mbiou, _, per_class_biou = compute_miou_biou(
            preds, labels, xyz, d=EvalConfig.boundary_tolerance
        )
        conn_biou = per_class_biou.get(conn_class, 0.0)
        return miou, iou_conn, mbiou, conn_biou, per_class_biou, avg_chunk_ms, peak_mem

    return miou, iou_conn, avg_chunk_ms, peak_mem


# ==========================================
# 留一交叉验证主流程 (LOOCV)
# ==========================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "train_data_v2")
    img_dir = os.path.join(script_dir, "train_images_v2")
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    N = len(all_files)

    # LOOCV 累积（含计时/显存统计）
    all_miou = []
    all_mbiou = []
    all_iou_conn = []
    all_conn_biou = []
    all_infer_times = []  # 每样本推理耗时 (ms)
    all_peak_mems = []    # 每样本峰值显存 (GB)

    print(f"样本总数: {N}")

    for fold in tqdm(range(N), desc="LOOCV 进度"):
        test_file = all_files[fold]
        train_files = [f for i, f in enumerate(all_files) if i != fold]

        # 输出测试文件信息
        test_npy_path = os.path.join(data_dir, test_file)
        data_debug = np.load(test_npy_path)
        labels_debug = data_debug[:, 12]
        conn_count = np.sum(labels_debug == 3)
        print(f"\n--- 第 {fold+1}/{N} 折: {test_file} | 过渡区点数: {conn_count}/{len(labels_debug)} ---")

        save_path = os.path.join(script_dir, "checkpoints", f"fold_{fold}.pth")

        train_one_fold(
            train_files=train_files,
            data_dir=data_dir,
            image_dir=img_dir,
            save_path=save_path,
            num_epochs=200
        )

        model = DS_MIN(
            num_classes=EvalConfig.num_classes,
            d_k=EvalConfig.d_k,
            num_layers=EvalConfig.num_layers
        ).to(EvalConfig.device)

        model.load_state_dict(torch.load(save_path, map_location=EvalConfig.device))
        model.eval()

        test_npy_path = os.path.join(data_dir, test_file)
        img_path = os.path.join(img_dir, test_file.replace(".npy", ".png"))
        roi_path = os.path.join(data_dir, test_file.replace(".npy", "_roi.png"))
        norm_path = os.path.join(data_dir, test_file.replace(".npy", "_norm.json"))

        results = evaluate_single(model, test_npy_path, img_path, roi_path, norm_path, compute_biou=True)
        miou, iou_conn, mbiou, conn_biou, per_class_biou, avg_chunk_ms, peak_mem = results

        print(f"  mIoU={miou:.4f}  mBIoU={mbiou:.4f}  IoU(过渡区)={iou_conn:.4f}  BIoU(过渡区)={conn_biou:.4f}  "
              f"推理={avg_chunk_ms:.1f}ms/chunk  显存峰值={peak_mem:.2f}GB")

        all_miou.append(miou)
        all_mbiou.append(mbiou)
        all_iou_conn.append(iou_conn)
        all_conn_biou.append(conn_biou)
        all_infer_times.append(avg_chunk_ms)
        all_peak_mems.append(peak_mem)

    # 输出汇总结果
    print("\n" + "=" * 60)
    print("留一交叉验证最终结果 (LOOCV RESULTS)")
    print("=" * 60)
    print(f"mIoU:              {np.mean(all_miou)*100:.2f} +/- {np.std(all_miou)*100:.2f} %")
    print(f"mBIoU:             {np.mean(all_mbiou)*100:.2f} +/- {np.std(all_mbiou)*100:.2f} %")
    print(f"IoU (过渡区):       {np.mean(all_iou_conn)*100:.2f} +/- {np.std(all_iou_conn)*100:.2f} %")
    print(f"BIoU (过渡区):      {np.mean(all_conn_biou)*100:.2f} +/- {np.std(all_conn_biou)*100:.2f} %")
    print("-" * 60)
    if all_infer_times:
        print(f"推理耗时 (per chunk): {np.mean(all_infer_times):.1f} +/- {np.std(all_infer_times):.1f} ms")
        # 每样本总推理时间 ≈ num_chunks * num_votes * avg_chunk_ms / 1000
        n_chunks = int(np.ceil(350000 / EvalConfig.chunk_size))  # ~350k pts per sample
        per_sample_s = n_chunks * EvalConfig.num_votes * np.mean(all_infer_times) / 1000.0
        print(f"推理耗时 (per sample): ~{per_sample_s:.2f} s ({n_chunks} chunks x {EvalConfig.num_votes} votes)")
    if all_peak_mems:
        print(f"峰值显存:           {np.mean(all_peak_mems):.2f} +/- {np.std(all_peak_mems):.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
