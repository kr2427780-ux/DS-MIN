import open3d as o3d
import numpy as np
import os
import json
from scipy.spatial import cKDTree
from tqdm import tqdm
from PIL import Image


def estimate_normals(points, k=30):
    """
    使用 PCA 在局部邻域上估计逐点法向量。
    返回的法向量已统一朝向视点原点方向。
    """
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        _, idx = tree.query(points[i], k=k)
        neighbours = points[idx]
        cov = np.cov(neighbours.T)
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        # 朝向原点
        if np.dot(normal, points[i]) > 0:
            normal = -normal
        normals[i] = normal
    return normals


def compute_distance_field(points, labels):
    """计算点云中每个点到最近理论交界面的距离 D(p_i)"""
    N = points.shape[0]
    distance_field = np.zeros(N)
    unique_classes = np.unique(labels)

    kdtrees = {}
    for cls in unique_classes:
        class_points = points[labels == cls]
        if class_points.shape[0] > 0:
            kdtrees[cls] = cKDTree(class_points)

    for cls in tqdm(unique_classes, desc="  边界距离计算中", leave=False):
        current_class_mask = (labels == cls)
        current_points = points[current_class_mask]
        if current_points.shape[0] == 0:
            continue

        min_dists = np.full(current_points.shape[0], np.inf)
        for other_cls in unique_classes:
            if cls == other_cls or other_cls not in kdtrees:
                continue
            dists, _ = kdtrees[other_cls].query(current_points, k=1, workers=-1)
            min_dists = np.minimum(min_dists, dists)

        distance_field[current_class_mask] = min_dists
    return distance_field


def process_and_save(input_ply_path, output_npy_path, voxel_size=0.1, z_threshold=550.0, target_points=350000):
    print(f"\n正在处理: {os.path.basename(input_ply_path)}")

    try:
        # 1. 读取原始点云
        pcd = o3d.t.io.read_point_cloud(input_ply_path)
        points = pcd.point['positions'].numpy()

        # 颜色仅作为参考保留，真正的光影特征将由网络读取真实 2D 照片提取
        colors = pcd.point['colors'].numpy() if 'colors' in pcd.point else np.zeros_like(points)

        N_total = len(points)
        print(f"  -> 原始读取数量: {N_total} 点")

        # 智能寻找标签字段
        label_keys = ['label', 'class', 'scalar_class', 'scalar_label']
        labels = None
        for key in label_keys:
            if key in pcd.point:
                labels = pcd.point[key].numpy().flatten()
                break
        if labels is None:
            print("  未找到标签字段，跳过该文件")
            return

        # 利用结构光相机的有序特性，显式生成指向真实照片的 (u, v) 索引
        if N_total == 5013504:
            W, H = 2448, 2048
            print(f"  检测到标准有序点云 ({W}x{H})，正在生成物理级 2D 像素索引...")
            u = np.arange(N_total) % W   # 真实照片中的 X 像素坐标
            v = np.arange(N_total) // W  # 真实照片中的 Y 像素坐标
            uv_coords = np.vstack((u, v)).T  # 形状 (5013504, 2)
        else:
            raise ValueError(f"点云数量为 {N_total}，不是期望的有序尺寸，请检查相机导出设置")

        # 2. 清洗：剔除 NaN 和 (0,0,0) 无效点
        valid_indices = np.all(np.isfinite(points), axis=1) & np.any(points != [0.0, 0.0, 0.0], axis=1)

        # 同步过滤，剔除无效点，同时保留真实照片上的 (u, v) 坐标映射
        points = points[valid_indices]
        colors = colors[valid_indices]
        labels = labels[valid_indices]
        uv_coords = uv_coords[valid_indices]

        # 3. 物理切割：Z 轴深度截断 (剥离背景板)
        # 自动检测合理的 Z 阈值：取 Z 值第 95 百分位作为参考
        z_sorted = np.sort(points[:, 2])
        p95_z = z_sorted[int(len(z_sorted) * 0.95)]
        # 如果显式传入的 z_threshold 不合理（例如几乎所有点都小于它），则自动调整
        auto_threshold = min(z_threshold, p95_z + 5.0)
        roi_indices = points[:, 2] < auto_threshold
        print(f"  -> Z 阈值: {auto_threshold:.1f}mm (第95百分位={p95_z:.1f}mm)")
        points = points[roi_indices]
        colors = colors[roi_indices]
        labels = labels[roi_indices]
        uv_coords = uv_coords[roi_indices]
        print(f"  -> 切除背景板后，ROI 区域点数: {len(points)} 点")

        # 4. 空间均匀化：自适应体素降采样
        # 根据 ROI 点数动态调整体素大小，目标约 350k 点
        clean_pcd = o3d.geometry.PointCloud()
        clean_pcd.points = o3d.utility.Vector3dVector(points)
        down_pcd = clean_pcd.voxel_down_sample(voxel_size=voxel_size)
        # 如果点数太多，增大体素
        while len(down_pcd.points) > target_points * 1.5 and voxel_size < 2.0:
            voxel_size += 0.05
            down_pcd = clean_pcd.voxel_down_sample(voxel_size=voxel_size)
        # 如果点数太少，减小体素
        while len(down_pcd.points) < target_points * 0.5 and voxel_size > 0.02:
            voxel_size -= 0.02
            down_pcd = clean_pcd.voxel_down_sample(voxel_size=voxel_size)
        down_points = np.asarray(down_pcd.points)
        print(f"  -> 体素降采样后: {len(down_points)} 点 (voxel_size={voxel_size:.2f}, 空间绝对均匀)")

        # 5. KD-Tree 映射：降采样后的点继承最近原始点的 2D 像素 (u, v) 索引
        print("  -> 正在进行二维坐标索引追踪 (2D-Index Tracking)...")
        tree = cKDTree(points)
        _, idx = tree.query(down_points, k=1, workers=-1)

        down_colors = colors[idx]
        down_labels = labels[idx]
        down_uvs = uv_coords[idx]  # 继承了真实 2D 照片的整型像素坐标

        # 6. 计算法向量 & 微观距离场 D(p_i)
        print("  -> 正在估计法向量...")
        down_normals = estimate_normals(down_points, k=30)
        print("  -> 正在计算微观边界距离场...")
        distance_field = compute_distance_field(down_points, down_labels)
        distance_field = np.clip(distance_field, 0.001, None)

        # 7. 保存为 .npy 供 PyTorch Dataset 读取
        # npy 列结构: xyz(3), normal(3), rgb(3), uv(2), dist(1), label(1) = 13 列
        processed_data = np.hstack((
            down_points,                      # 0:3   XYZ 坐标
            down_normals,                     # 3:6   法向量
            down_colors,                      # 6:9   RGB 颜色
            down_uvs,                         # 9:11  像素 UV 坐标
            distance_field.reshape(-1, 1),    # 11    距离场
            down_labels.reshape(-1, 1)        # 12    标签
        ))

        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, processed_data)
        print(f"  成功保存至 {output_npy_path}")

        # 保存全局几何归一化参数（训练和推理时统一使用）
        norm_params = {
            "centroid": down_points.mean(axis=0).tolist(),
            "max_dist": float(np.max(np.sqrt(np.sum((down_points - down_points.mean(axis=0)) ** 2, axis=1)))),
            "image_W": W,
            "image_H": H,
        }
        norm_path = output_npy_path.replace(".npy", "_norm.json")
        with open(norm_path, "w") as f:
            json.dump(norm_params, f)

        # 生成并保存 RoI 二值掩码图像（完整 35 万点投影，非采样时的 8192 点）
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        u_int = down_uvs[:, 0].astype(int)
        v_int = down_uvs[:, 1].astype(int)
        valid_uv = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        roi_mask[v_int[valid_uv], u_int[valid_uv]] = 255
        roi_img = Image.fromarray(roi_mask, mode="L")
        roi_path = output_npy_path.replace(".npy", "_roi.png")
        roi_img.save(roi_path)
        print(f"  归一化参数 -> {norm_path}")
        print(f"  RoI 掩码 -> {roi_path}")

    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == "__main__":
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, "DataSet")
    output_dir = os.path.join(script_dir, "train_data_v2")

    ply_files = glob.glob(os.path.join(raw_data_dir, "*.ply"))
    print(f"共找到 {len(ply_files)} 个 .ply 文件准备预处理")

    for ply_file in ply_files:
        base_name = os.path.basename(ply_file)
        npy_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + ".npy")
        if os.path.exists(npy_path):
            print(f"  {npy_path} 已存在，跳过...")
            continue
        process_and_save(ply_file, npy_path, voxel_size=0.1, z_threshold=550.0, target_points=350000)
