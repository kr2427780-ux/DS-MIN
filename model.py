import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

from shading_extractor import ShadingFeatureEnhancer


# --- 1. KNN-based Graph Convolution for 3D Local Geometry ---

def knn_gpu(x, k):
    """Efficient KNN on GPU via matrix multiply, as used in DGCNN-family architectures."""
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_local_features(x, idx):
    """Extract (neighbour - centre, centre) pairs for each point."""
    batch_size, num_points, c = x.size()
    k = idx.size(2)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    neighbors = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, c)
    x_repeated = x.unsqueeze(2).repeat(1, 1, k, 1)
    return torch.cat((neighbors - x_repeated, x_repeated), dim=3)


class GeometryAwareBlock(nn.Module):
    """
    3D geometric branch: KNN-based graph convolution + two-stage local
    feature aggregation, producing the raw 3D representation F_3D(raw).
    """
    def __init__(self, in_channels, out_channels, k=24):
        super(GeometryAwareBlock, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(192, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        B, N, C = x.shape

        idx1 = knn_gpu(x, k=self.k)
        feat1 = get_local_features(x, idx1)
        feat1 = self.conv1(feat1.view(B * N * self.k, -1)).view(B, N, self.k, -1).max(dim=2)[0]

        idx2 = knn_gpu(feat1, k=self.k)
        feat2 = get_local_features(feat1, idx2)
        feat2 = self.conv2(feat2.view(B * N * self.k, -1)).view(B, N, self.k, -1).max(dim=2)[0]

        out = torch.cat([feat1, feat2], dim=-1)
        out = self.fusion(out.view(B * N, -1)).view(B, N, -1)
        return out


# --- 2. 2D CNN Backbone with Shading Gradient Enhancement ---

class CNNBackbone2D(nn.Module):
    """
    2D shading branch: ResNet18 (truncated at layer2) extracts shallow
    image features, then the ShadingFeatureEnhancer injects high-frequency
    gradient responses via residual modulation.

    Pipeline:
      I'                       -> 2D CNN -> F_img  (shallow feature map)
      I'                       -> Sobel  -> G_u, G_v -> G_mag -> H(.) -> M_hf
      F_2D(raw) = F_img + F_img * M_hf                   (residual modulation)
    """
    def __init__(self, out_channels=64):
        super(CNNBackbone2D, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            resnet.conv1,      # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # /4
            resnet.layer1,     # /4 (output 64 channels at 1/4 spatial size)
        )

        self.shading_enhancer = ShadingFeatureEnhancer(in_channels=3)

        # Project to target channel dimension
        self.proj = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W]  RoI-masked RGB images (I' = I * M_roi)
        Returns:
            F_2D_map: [B, out_channels, H', W']  shading-enhanced feature map
        """
        cnn_features = self.cnn(images)                               # [B, 128, H/8, W/8]
        F_2D_raw = self.shading_enhancer(images, cnn_features)        # shade-enhanced
        F_2D_map = self.proj(F_2D_raw)                                # [B, 64, H/8, W/8]
        return F_2D_map


# --- 3. Shading-Geometry Dual-Stream Cross-Modal Interaction (SG-DCI) ---

class SG_DCI_Block(nn.Module):
    """
    A single SG-DCI layer implementing bidirectional cross-attention:

    Stage 1 (3D as Query):   F_3D^(l+1) = Softmax(Q_3D.K_2D^T / sqrt(d_k)) . V_2D + F_3D^(l)
    Stage 2 (2D as Query):   F_2D^(l+1) = Softmax(Q_2D.K_3D^T / sqrt(d_k)) . V_3D + F_2D^(l)

    With residual connections and LayerNorm after each stage.
    """
    def __init__(self, d_k=64, num_heads=4):
        super(SG_DCI_Block, self).__init__()
        self.attn_3d_to_2d = nn.MultiheadAttention(embed_dim=d_k, num_heads=num_heads, batch_first=True)
        self.attn_2d_to_3d = nn.MultiheadAttention(embed_dim=d_k, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_k)
        self.norm2 = nn.LayerNorm(d_k)

    def forward(self, F_3D, F_2D):
        # Stage 1: 3D queries 2D shading gradients to enhance boundary features
        attn_out_3d, _ = self.attn_3d_to_2d(query=F_3D, key=F_2D, value=F_2D)
        F_3D_next = self.norm1(F_3D + attn_out_3d)

        # Stage 2: 2D queries 3D topology to suppress background hallucination
        attn_out_2d, _ = self.attn_2d_to_3d(query=F_2D, key=F_3D_next, value=F_3D_next)
        F_2D_next = self.norm2(F_2D + attn_out_2d)

        return F_3D_next, F_2D_next


# --- 4. DS-MIN Main Architecture ---

class DS_MIN(nn.Module):
    """
    DS-MIN: Boundary-Aware Dual-Stream Multimodal Interaction Network

    Architecture:
      Point Cloud -> GeometryAwareBlock -> LN -> F_3D^(0)
      Image -> CNNBackbone2D + ShadingEnhancer -> LN -> F_2D^(0)
      F_3D^(0), F_2D^(0) -> SG-DCI Blocks x L -> MLP -> Softmax -> y_hat

    The loss L_Total is applied externally (see loss.py).
    """
    def __init__(self, num_classes=5, d_k=64, num_layers=3):
        super(DS_MIN, self).__init__()
        self.d_k = d_k
        self.num_layers = num_layers

        # Dual-stream backbones
        self.backbone_3d = GeometryAwareBlock(in_channels=6, out_channels=64, k=24)
        self.backbone_2d = CNNBackbone2D(out_channels=64)

        # Linear projections into shared D-dimensional space
        self.proj_3d = nn.Linear(64, d_k)
        # 2D projection as 1x1 conv to operate on feature map before sampling
        self.proj_2d = nn.Conv2d(64, d_k, kernel_size=1)

        # Cascaded SG-DCI interaction modules
        self.sg_dci_blocks = nn.ModuleList([
            SG_DCI_Block(d_k=d_k, num_heads=4) for _ in range(num_layers)
        ])

        # MLP classifier head (LayerNorm 替代 BatchNorm，batch_size=1 稳定)
        self.classifier = nn.Sequential(
            nn.Linear(d_k, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points, images, uv_indices):
        """
        Args:
            points:     [B, N, 6]      XYZ + normals (normalised)
            images:     [B, 3, H, W]   full-resolution 2D RGB images
            uv_indices: [B, N, 2]      per-point pixel coords, normalised to [-1, 1]
        Returns:
            logits:     [B, N, C]      raw class scores (pre-Softmax)
        """
        B, N, _ = points.shape

        # RoI 前景掩码已在数据管道中应用（dataset.py / evaluate.py / predict.py）
        # 此处直接使用已掩码的图像，不再重复生成稀疏掩码

        # Stream 1: 3D geometric features
        F_3D_raw = self.backbone_3d(points)                # [B, N, 64]

        # Stream 2: 2D shading-enhanced features
        F_2D_map = self.backbone_2d(images)                 # [B, 64, H', W']

        # Linear projection to shared D-dimensional space (Section 3.3)
        F_3D = self.proj_3d(F_3D_raw)                      # [B, N, d_k]

        # Project 2D feature map, then bilinear-sample per-point
        F_2D_map_proj = self.proj_2d(F_2D_map)              # [B, d_k, H/4, W/4]
        grid = uv_indices.unsqueeze(1)                      # [B, 1, N, 2]
        F_2D_sampled = F.grid_sample(F_2D_map_proj, grid,
                                     mode='bilinear', padding_mode='zeros',
                                     align_corners=True)    # [B, d_k, 1, N]
        F_2D = F_2D_sampled.squeeze(2).transpose(1, 2)     # [B, N, d_k]

        # Cascaded SG-DCI bidirectional interaction
        for block in self.sg_dci_blocks:
            F_3D, F_2D = block(F_3D, F_2D)

        # MLP classifier
        logits = self.classifier(F_3D.reshape(B * N, -1)).view(B, N, -1)
        return logits


def _generate_roi_mask(uv_indices, image_dims):
    """Generate binary foreground mask M_roi from projected uv coordinates."""
    B = uv_indices.shape[0]
    H, W = image_dims
    u_pixel = ((uv_indices[..., 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
    v_pixel = ((uv_indices[..., 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
    mask = torch.zeros(B, 1, H, W, device=uv_indices.device)
    for b in range(B):
        mask[b, 0, v_pixel[b], u_pixel[b]] = 1.0
    return mask
