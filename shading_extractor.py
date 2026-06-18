import torch
import torch.nn as nn
import torch.nn.functional as F


class ShadingFeatureEnhancer(nn.Module):
    """
    阴影特征增强器 - 提取高频率阴影梯度并将其作为辅助约束注入到2D特征中
    (对应论文 Section 3.3: High-Frequency Shading Feature Extraction)

    实现过程:
    1. 使用方向性偏导卷积核 (Sobel算子) 提取局部梯度响应图 G_u, G_v
    2. 计算梯度幅值图 G_mag = sqrt(G_u ⊙ G_u + G_v ⊙ G_v)
    3. 通过轻量级映射操作符 H(·) 生成高频响应掩码 M_hf = Sigmoid(H(G_mag))
    4. 通过残差调制注入2D特征: F_2D(raw) = F_img + F_img ⊗ M_hf
    """

    def __init__(self, in_channels=3):
        super(ShadingFeatureEnhancer, self).__init__()

        # 方向性偏导卷积核 W_u, W_v — 初始化为 Sobel 算子, 随后在训练中自适应优化
        self.W_u = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                             padding=1, groups=in_channels, bias=False)
        self.W_v = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                             padding=1, groups=in_channels, bias=False)

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)

        with torch.no_grad():
            self.W_u.weight.copy_(sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
            self.W_v.weight.copy_(sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

        # 轻量级映射操作符 H(·): G_mag → M_hf
        self.mapping_op = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1),
            nn.InstanceNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),
        )

    def forward(self, images, base_features=None):
        """
        Args:
            images: [B, 3, H, W]  原始图像 I' (乘以 M_roi 后的图像)
            base_features: [B, C, H', W']  2D CNN 输出的 F_img
        Returns:
            enhanced_features: [B, C, H', W']  F_2D(raw) 或梯度图/掩码
        """
        B, C, H, W = images.shape

        # G_u = I' * W_u,  G_v = I' * W_v
        G_u = self.W_u(images)                                    # [B, C, H, W]
        G_v = self.W_v(images)                                    # [B, C, H, W]

        # G_mag = sqrt(G_u ⊙ G_u + G_v ⊙ G_v)
        G_mag = torch.sqrt(G_u.pow(2) + G_v.pow(2) + 1e-8)       # [B, C, H, W]
        G_mag = torch.mean(G_mag, dim=1, keepdim=True)            # [B, 1, H, W]

        # M_hf = Sigmoid(H(G_mag))
        M_hf = torch.sigmoid(self.mapping_op(G_mag))              # [B, 1, H, W]

        if base_features is not None:
            target_H, target_W = base_features.shape[2], base_features.shape[3]
            if (target_H, target_W) != (H, W):
                M_hf = F.interpolate(M_hf, size=(target_H, target_W),
                                     mode='bilinear', align_corners=True)
            # F_2D(raw) = F_img + F_img ⊗ M_hf
            return base_features + base_features * M_hf
        else:
            return G_mag, M_hf
