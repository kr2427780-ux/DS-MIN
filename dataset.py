import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image
import torchvision.transforms as transforms


class ElectrodeDataset(Dataset):
    def __init__(self, data_dir, image_dir, is_train=True,
                 expected_width=2448, expected_height=2048,
                 debug_check=False):
        """
        :param data_dir: 存放 .npy 点云文件的目录
        :param image_dir: 存放对应 2D .png 图像的目录
        :param expected_width: 结构光相机图像宽度
        :param expected_height: 结构光相机图像高度
        :param debug_check: 是否开启像素级对齐调试
        """
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.is_train = is_train
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.debug_check = debug_check

        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

        # 文件存在性检查
        for npy_file in self.file_list:
            img_file = npy_file.replace('.npy', '.png')
            img_path = os.path.join(self.image_dir, img_file)
            if not os.path.exists(img_path):
                raise RuntimeError(f"未找到对应图像文件: {img_file}")
            # 检查 RoI 掩码和归一化参数是否存在
            roi_file = npy_file.replace('.npy', '_roi.png')
            roi_path = os.path.join(self.data_dir, roi_file)
            if not os.path.exists(roi_path):
                raise RuntimeError(f"未找到 RoI 掩码文件: {roi_file}，请重新运行 preprocess_data.py")
            norm_file = npy_file.replace('.npy', '_norm.json')
            norm_path = os.path.join(self.data_dir, norm_file)
            if not os.path.exists(norm_path):
                raise RuntimeError(f"未找到归一化参数文件: {norm_file}，请重新运行 preprocess_data.py")

        print(f"数据集初始化完成，共 {len(self.file_list)} 对多模态样本")

        # 图像预处理（ResNet 标准归一化）
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # RoI 掩码预处理（灰度图转二值 tensor，预加载到内存）
        self.roi_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npy_filename = self.file_list[idx]
        npy_path = os.path.join(self.data_dir, npy_filename)
        img_filename = npy_filename.replace('.npy', '.png')
        img_path = os.path.join(self.image_dir, img_filename)

        roi_filename = npy_filename.replace('.npy', '_roi.png')
        roi_path = os.path.join(self.data_dir, roi_filename)
        norm_filename = npy_filename.replace('.npy', '_norm.json')
        norm_path = os.path.join(self.data_dir, norm_filename)

        data = np.load(npy_path)

        # 数据结构:
        # 0:3    XYZ 坐标
        # 3:6    法向量 (nx, ny, nz)
        # 6:9    RGB 颜色
        # 9:11   UV 像素坐标
        # 11     距离场 distance_field
        # 12     标签 label
        xyz = data[:, 0:3]
        normals = data[:, 3:6]
        rgb = data[:, 6:9]
        uv = data[:, 9:11]
        distance_field = data[:, 11]
        labels = data[:, 12].astype(np.int64)

        # 读取图像
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # 分辨率一致性检查
        if W != self.expected_width or H != self.expected_height:
            raise ValueError(
                f"图像分辨率不匹配: {img_filename}, "
                f"得到 {W}x{H}, 期望 {self.expected_width}x{self.expected_height}"
            )

        img_tensor = self.img_transform(image)

        # RoI 掩码不预乘到图像上，保留原始图像用于 Shading 分支
        # CNNBackbone2D 内部的 ShadingFeatureEnhancer 需要在全分辨率图像上计算 Sobel 梯度
        roi_img = Image.open(roi_path).convert("L")
        roi_tensor = self.roi_transform(roi_img)  # [1, H, W]，值 0.0/1.0

        # 不再对图像进行 RoI 预掩蔽,避免在全分辨率 Sobel 梯度图中引入
        # 前景-背景边界的大幅值伪影(M_hf 噪声主来源)

        # 像素级对齐调试（可选）
        if self.debug_check:
            rand_id = np.random.randint(0, len(uv))
            u_pixel = int(uv[rand_id, 0])
            v_pixel = int(uv[rand_id, 1])

            img_np = np.array(image)
            pixel_color = img_np[v_pixel, u_pixel] / 255.0
            point_color = rgb[rand_id]

            diff = np.linalg.norm(pixel_color - point_color)
            if diff > 0.1:
                print(f"像素对齐偏差较大: diff={diff:.4f}")

        # 分层采样：过渡区（类别3）强制至少 800 点，其余均匀分配
        max_points = 8192
        num_points = xyz.shape[0]
        transition_mask = (labels == 3)
        n_transition = transition_mask.sum()

        if n_transition > 0:
            n_target_transition = min(n_transition, 800)
            n_other = max_points - n_target_transition

            trans_idx = np.where(transition_mask)[0]
            other_idx = np.where(~transition_mask)[0]

            trans_choice = np.random.choice(trans_idx, n_target_transition, replace=(n_target_transition > n_transition))
            other_choice = np.random.choice(other_idx, n_other, replace=(n_other > len(other_idx)))

            choice = np.concatenate([trans_choice, other_choice])
            np.random.shuffle(choice)
        else:
            choice = np.random.choice(num_points, max_points, replace=(num_points < max_points))

        xyz = xyz[choice]
        normals = normals[choice]
        rgb = rgb[choice]
        uv = uv[choice]
        distance_field = distance_field[choice]
        labels = labels[choice]

        # UV 归一化到 [-1, 1]
        u_norm = (uv[:, 0] / (W - 1)) * 2.0 - 1.0
        v_norm = (uv[:, 1] / (H - 1)) * 2.0 - 1.0
        uv_indices = np.stack((u_norm, v_norm), axis=-1)

        # 几何归一化（使用预处理阶段保存的全局参数，训练和推理一致）
        with open(norm_path, "r") as f:
            norm_params = json.load(f)
        centroid = np.array(norm_params["centroid"])
        max_dist = norm_params["max_dist"]
        xyz = (xyz - centroid) / (max_dist + 1e-6)

        # 轻量级数据增强（不破坏物理映射关系）
        if self.is_train:
            # 仅做微小扰动，不旋转
            xyz += np.random.normal(0, 0.005, size=xyz.shape)

            color_scale = np.random.uniform(0.9, 1.1)
            rgb = np.clip(rgb * color_scale +
                          np.random.normal(0, 0.02, size=rgb.shape),
                          0.0, 1.0)

        return {
            'xyz': torch.tensor(np.hstack([xyz, normals]), dtype=torch.float32),
            'rgb': torch.tensor(rgb, dtype=torch.float32),
            'images': img_tensor,
            'uv_indices': torch.tensor(uv_indices, dtype=torch.float32),
            'distance_field': torch.tensor(distance_field, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
