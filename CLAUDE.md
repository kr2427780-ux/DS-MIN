# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目概览

DS-MIN: A Boundary-Aware Dual-Stream Multimodal Interaction Network for Part-Level
Segmentation of Homogeneous Industrial Point Clouds. 投稿 The Visual Computer (TVC).

**当前阶段**: Revise & Resubmit (截止 2026-06-19)，第二轮修改已完成 (0618.pdf)。

## 环境要求

- Python 3.8+ (conda 环境 `lunwen`)
- PyTorch, torchvision, open3d, scipy, PIL/Pillow, numpy, tqdm, sklearn
- CUDA GPU (RTX 4090 推荐，推理峰值 ~4GB)
- 输入: 结构光相机有序点云 (2448×2048, ~5M pts) + 对应 2D 图像

## 常用命令

```bash
python preprocess_data.py      # PLY → .npy (仅一次, 生成 _roi.png + _norm.json)
python validate_prior.py       # 物理先验验证 (Section 3.2, Table 1)
python train.py                # 单折全数据训练 (快速验证)
python evaluate.py             # LOOCV 主实验 (Section 4.3, Table 2)
python predict.py [checkpoint] # 推理可视化 (可选权重文件名)
python ablation_3d_only.py     # Table 5 Config 1: 纯 3D Backbone 消融
python ablation_3d_shading.py  # Table 5 Config 2: 3D+Shading 消融
python ablation_full.py        # Table 5 Config 3: 3D+Shading+SG-DCI 消融
python vis_downsample.py [npy] # 降采样可视化 (16000 点)
```

## 核心架构

### 数据流

```
PLY 有序点云 (5M pts)
  → preprocess_data.py → .npy (13列, ~350k pts) + _roi.png + _norm.json
    → dataset.py → DataLoader (分层采样 8192 pts/样本)
      → model.py → 逐点 logits [B,N,5]
        → loss.py → L_Total = λ·L_DWCE + (1-λ)·L_BLS
```

### 模型结构 (model.py)

| 组件 | 规格 | 输出形状 |
|------|------|----------|
| GeometryAwareBlock | KNN k=24, 两阶段 GCN | [B,N,64] |
| CNNBackbone2D | ResNet18 conv1→layer1 (1/4 分辨率), 64ch | [B,64,H/4,W/4] |
| ShadingFeatureEnhancer | 固定 Sobel→G_mag→InstanceNorm→H(·)→M_hf→残差调制 | [B,64,H/4,W/4] |
| SG_DCI_Block × 3 | 双向交叉注意力, d_k=64, h=4 | [B,N,64] |
| Classifier | Concat[F_3D,F_2D]→512→256→5 | [B,N,5] |

### 损失函数参数 (loss.py)

| 参数 | 值 | 含义 |
|------|-----|------|
| lambda_weight | 0.5 | L_DWCE 与 L_BLS 平衡系数 |
| sigma | 2.0 | 指数衰减速率 (mm), 有效边界带 3σ≈6mm |
| gamma | 2.0 | 边界惩罚增益, ω_max=1+γ=3.0 |
| tau | 5.0 | L_BLS 硬边界掩码阈值 (mm) |
| class_weights | [1.0, 1.0, 1.2, 8.0, 1.2] | 背景/杆/帽/过渡区/臂 |

### .npy 数据格式 (13 列)

| 列 | 内容 | 说明 |
|----|------|------|
| 0:3 | XYZ 坐标 | mm, 已归一化 |
| 3:6 | 法向量 | PCA 估计 k=30 |
| 6:9 | RGB | 0-255 |
| 9:11 | UV 像素坐标 | u∈[0,2447], v∈[0,2047] |
| 11 | distance_field | mm, 到最近异类点的欧氏距离 |
| 12 | label | 0:背景 1:杆 2:帽 3:过渡区 4:臂 |

### 预处理参数 (preprocess_data.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| voxel_size | 0.1 | 体素降采样初始步长 (mm) |
| z_threshold | 550.0 | Z 轴截断上限 (自适应: min(550, p95_z+5)) |
| target_points | 350000 | 预处理目标点数 |
| image_W × image_H | 2448 × 2048 | 结构光相机分辨率 |

### 训练/评估/推理参数

| 参数 | 训练 | 评估 | 推理 |
|------|------|------|------|
| num_epochs | 100 | 200 (per fold) | — |
| batch_size | 1 | — | — |
| learning_rate | 1e-3 | 1e-3 | — |
| weight_decay | 1e-4 | 1e-4 | — |
| scheduler | CosineAnnealing(T_max=num_epochs) | 同 | — |
| 每样本采样点数 | 8192 | ~350k (全量) | ~16k (voxel) |
| 过渡区采样保证 | ≥800/pts | — | — |
| TTA num_votes | — | 5 | 1 |
| chunk_size | — | 8192 | 16384 |
| boundary_tolerance (mBIoU) | — | 0.15mm (3×APS) | — |

## 目录结构

```
DS-MIN/
  DataSet/           # 原始 .ply 点云 (10 样本)
  train_data_v2/     # 预处理 .npy + _roi.png + _norm.json
  train_images_v2/   # 2D .png 图像 (2448×2048)
  TestData/          # 推理测试 PLY + Image
  checkpoints/       # 模型权重 .pth
```

## 关键设计决策

- UV 按有序点云行列索引生成 (`u=idx%2448, v=idx//2448`)，无需相机矩阵
- 图像不预乘 RoI 掩码: 预乘会在 Sobel 梯度图上产生硬边界阶跃伪影
- Sobel 梯度核固定 (requires_grad=False): 可学习版本在训练中漂移，恶化边缘响应
- ShadingEnhancer 用 InstanceNorm2d 非 BatchNorm2d: batch_size=1 时 BN 归一化为零
- 分类器输入拼接 [F_3D, F_2D] (2*d_k=128): 比只用 F_3D 高 1-2% mIoU
- KNN k=24: 较大的邻域改善低对比度区域的局部特征稳定性
