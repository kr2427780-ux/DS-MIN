import torch
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
from model import DS_MIN


class InferenceConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    d_k = 64
    num_layers = 3
    target_points = 16000  # 推理目标点数
    infer_chunk_size = 16384  # 单次KNN推理的最大点数（防止 [N,N] OOM）
    num_votes = 1            # 测试时增强 (TTA) 投票次数（与 evaluate.py 一致）

    # 工业标准的可视化颜色映射
    color_map = {
        0: [0.76, 0.76, 0.76],  # 背景 - 灰
        1: [0.0, 0.87, 0.0],    # 杆 - 绿
        2: [0.0, 0.33, 1.0],    # 帽 - 蓝
        3: [0.93, 0.93, 0.0],   # 过渡区 - 黄
        4: [0.90, 0.22, 0.27]   # 大臂 - 红
    }


def main():
    import sys

    # ---- 消融模型权重支持 ----
    # 用法:
    #   python predict.py                                    # 默认 debug_model.pth
    #   python predict.py ablation_3d_only.pth               # Table 5 Config 1
    #   python predict.py ablation_3d_shading.pth             # Table 5 Config 2
    #   python predict.py ablation_full_sgdci.pth             # Table 5 Config 3
    if len(sys.argv) > 1:
        ckpt_name = sys.argv[1]
        if not ckpt_name.endswith(".pth"):
            ckpt_name += ".pth"
    else:
        ckpt_name = "debug_model.pth"

    print("DS-MIN 真实工业推理模式启动")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "TestData", "0320-15")
    raw_ply_path = os.path.join(test_dir, "PointCloud.ply")

    image_path = os.path.join(test_dir, "Image.png")
    checkpoint_path = os.path.join(script_dir, "checkpoints", ckpt_name)

    if not os.path.exists(raw_ply_path):
        raise FileNotFoundError(f"未找到输入 PLY 文件: {raw_ply_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")

    # 根据权重文件名自动选择模型类
    ckpt_lower = ckpt_name.lower()
    if "3d_only" in ckpt_lower or "ablation_3d_only" in ckpt_lower:
        from ablation_3d_only import Ablation3DOnly
        model = Ablation3DOnly().to(InferenceConfig.device)
        uses_2d = False
        print(f"[模型] Ablation3DOnly (Table 5 Config 1)")
    elif "3d_shading" in ckpt_lower and "sgdci" not in ckpt_lower:
        from ablation_3d_shading import Ablation3DShading
        model = Ablation3DShading().to(InferenceConfig.device)
        uses_2d = True
        print(f"[模型] Ablation3DShading (Table 5 Config 2)")
    else:
        model = DS_MIN(
            num_classes=InferenceConfig.num_classes,
            d_k=InferenceConfig.d_k,
            num_layers=InferenceConfig.num_layers
        ).to(InferenceConfig.device)
        uses_2d = True
        print(f"[模型] DS_MIN (full)")

    model.load_state_dict(torch.load(checkpoint_path, map_location=InferenceConfig.device))
    model.eval()

    # 图像预处理
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 读取点云
    pcd = o3d.t.io.read_point_cloud(raw_ply_path)
    points = pcd.point['positions'].numpy()
    colors = pcd.point['colors'].numpy()
    N_total = len(points)

    # 自动推断图像尺寸
    image_W, image_H = 2448, 2048
    if image_W * image_H != N_total:
        raise ValueError("点云不是有序结构光格式，无法构建像素映射")

    print(f"检测到有序结构光点云: {image_W} x {image_H}")

    # 加载图像（优先真实 Image.png）
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        print(f"使用真实图像: {image_path}")
    else:
        colors_img = colors.reshape((image_H, image_W, 3))
        if colors_img.max() <= 1.0:
            img_array = (colors_img * 255).astype(np.uint8)
        else:
            img_array = colors_img.astype(np.uint8)
        image = Image.fromarray(img_array)
        print("未找到 Image.png，从点云颜色重建图像")

    img_tensor = img_transform(image).unsqueeze(0).to(InferenceConfig.device)

    # 构建 uv 映射
    u = np.arange(N_total) % image_W
    v = np.arange(N_total) // image_W
    uv_coords = np.vstack((u, v)).T

    # 数据清洗
    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]
    uv_coords = uv_coords[valid_mask]

    # Z 轴截断：与 preprocess_data.py 对齐（自适应阈值逻辑）
    z_sorted = np.sort(points[:, 2])
    p95_z = z_sorted[int(len(z_sorted) * 0.95)]
    auto_z_threshold = min(550.0, p95_z + 5.0)
    print(f"Z 阈值: {auto_z_threshold:.1f}mm (第95百分位={p95_z:.1f}mm)")
    z_mask = points[:, 2] < auto_z_threshold
    points = points[z_mask]
    colors = colors[z_mask]
    uv_coords = uv_coords[z_mask]

    # ---- 关键修复：RoI 掩码由全部前景点生成（覆盖率 ~7%），而非仅降采样后的 10k 点（覆盖率 0.2%）----
    ALL_FG_POINTS = points.copy()
    all_fg_u = uv_coords[:, 0].astype(int)
    all_fg_v = uv_coords[:, 1].astype(int)

    roi_mask = np.zeros((image_H, image_W), dtype=np.float32)
    valid_uv = (all_fg_u >= 0) & (all_fg_u < image_W) & (all_fg_v >= 0) & (all_fg_v < image_H)
    roi_mask[all_fg_v[valid_uv], all_fg_u[valid_uv]] = 1.0
    roi_tensor = torch.tensor(roi_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(InferenceConfig.device)
    img_tensor = img_tensor * roi_tensor
    print(f"RoI 掩码填充率: {valid_uv.sum() / (image_W * image_H) * 100:.1f}%")
    # ---- RoI 掩码修复结束 ----

    # 动态体素降采样
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(points)

    voxel_size = 0.1
    down_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)
    while len(down_pcd.points) > InferenceConfig.target_points * 1.5:
        voxel_size += 0.05
        down_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)
    while len(down_pcd.points) < InferenceConfig.target_points * 0.8 and voxel_size > 0.02:
        voxel_size -= 0.02
        down_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)

    xyz_down = np.asarray(down_pcd.points)
    print(f"体素化后点数: {xyz_down.shape[0]}")

    # KD-Tree 映射 uv
    tree = cKDTree(points)
    _, idx = tree.query(xyz_down, k=1)
    uv_down = uv_coords[idx]

    # 法向量估计
    from preprocess_data import estimate_normals
    normals_down = estimate_normals(xyz_down, k=30)

    # ---- 关键修复：几何归一化使用全部前景点的参数，而非降采样后的 1 万点 ----
    centroid = np.mean(ALL_FG_POINTS, axis=0)
    xyz_norm = xyz_down - centroid
    max_dist = np.max(np.sqrt(np.sum((ALL_FG_POINTS - centroid) ** 2, axis=1)))
    xyz_norm = xyz_norm / (max_dist + 1e-6)
    # ---- 归一化修复结束 ----

    # UV 归一化到 [-1, 1]
    u_norm = (uv_down[:, 0] / (image_W - 1)) * 2.0 - 1.0
    v_norm = (uv_down[:, 1] / (image_H - 1)) * 2.0 - 1.0
    uv_norm = np.stack((u_norm, v_norm), axis=-1)

    # 分块推理 + TTA 投票
    N = int(xyz_norm.shape[0])
    chunk_size = int(InferenceConfig.infer_chunk_size)
    num_votes = int(InferenceConfig.num_votes)
    vote_logits = np.zeros((N, InferenceConfig.num_classes), dtype=np.float32)

    # ---- 计时 & 显存统计（Section 4.1）----
    all_chunk_times = []   # ms per chunk
    peak_mem_gb = 0.0

    for vote_idx in range(num_votes):
        logits_full = np.zeros((N, InferenceConfig.num_classes), dtype=np.float32)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            curr_size = end - i

            i_int, end_int, curr_int = int(i), int(end), int(curr_size)

            chunk = np.hstack([xyz_norm[i_int:end_int], normals_down[i_int:end_int]])
            chunk_uv = uv_norm[i_int:end_int]

            if curr_int < chunk_size:
                pad_size = chunk_size - curr_int
                pad_idx = np.random.choice(curr_int, pad_size, replace=True)
                chunk = np.vstack([chunk, chunk[pad_idx]])
                chunk_uv = np.vstack([chunk_uv, chunk_uv[pad_idx]])

            pts_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(InferenceConfig.device)
            uv_tensor = torch.tensor(chunk_uv, dtype=torch.float32).unsqueeze(0).to(InferenceConfig.device)

            if InferenceConfig.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with torch.no_grad():
                logits_chunk = model(pts_tensor, img_tensor, uv_tensor).squeeze(0).cpu().numpy()

            if InferenceConfig.device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                all_chunk_times.append(start.elapsed_time(end))
                peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated() / 1024**3)

            logits_full[i_int:end_int] = logits_chunk[:curr_int]

        vote_logits += logits_full
        print(f"  TTA 投票: {vote_idx + 1}/{num_votes}")

    # 输出计时/显存汇总
    if all_chunk_times:
        avg_ms = np.mean(all_chunk_times)
        total_s = np.sum(all_chunk_times) / 1000.0
        print(f"[计时] 每块推理耗时: {avg_ms:.1f} ms (±{np.std(all_chunk_times):.1f}), "
              f"单样本总推理时间: {total_s:.2f} s ({len(all_chunk_times)} chunks)")
    print(f"[显存] 单样本推理峰值显存: {peak_mem_gb:.2f} GB")

    preds = np.argmax(vote_logits, axis=1)

    # 上色可视化
    pred_colors = np.array([InferenceConfig.color_map[int(p)] for p in preds])
    down_pcd.colors = o3d.utility.Vector3dVector(pred_colors)

    print("推理完成，正在打开可视化窗口...")
    o3d.visualization.draw_geometries([down_pcd],
                                      window_name="DS-MIN 真实工业推理")


if __name__ == "__main__":
    main()
