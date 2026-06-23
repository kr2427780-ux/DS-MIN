# Boundary-Aware Multimodal Segmentation of Homogeneous Industrial Point Clouds Using Geometry–Shading Interaction

Official PyTorch implementation for *"Boundary-Aware Multimodal Segmentation of Homogeneous Industrial Point Clouds Using Geometry–Shading Interaction"* (under review at **The Visual Computer**).


## Overview

DS-MIN is a physics-guided multimodal framework for part-level segmentation of homogeneous industrial point clouds. It treats 3D geometry and 2D shading as complementary low- and high-frequency signals and aligns them through bidirectional cross-attention (SG-DCI). A boundary-aware joint loss further constrains structural continuity in low-contrast connection regions under few-shot conditions (10 training scans, LOOCV protocol).

| Metric | Value |
|--------|-------|
| Inference memory | ~9.0 GB |
| Per-sample inference | ~0.69 s |


## Quick Start

### Requirements

- Python 3.8+
- CUDA-capable GPU (9 GB+ VRAM recommended)
- [PyTorch](https://pytorch.org) ≥ 2.0

Install dependencies:

```bash
pip install -r requirements.txt
```

### Preprocessing (once)

Place your raw .ply files in `DataSet/` and run:

```bash
python preprocess_data.py
```

This produces per-scan `.npy` files (13 columns), `_roi.png` RoI masks, and `_norm.json` normalisation parameters in `train_data_v2/`.

### Training & Evaluation

**Single-fold quick check** ( 10 samples, 100 epochs, ~10 min on RTX 4090):

```bash
python train.py
```

**Full Leave-One-Out Cross-Validation** (10 × 200 epochs):

```bash
python evaluate.py
```

This prints per-fold mIoU / mBIoU, averages across folds, and wall-clock training + inference time.


**Inference on a new scan:**

```bash
python predict.py [optional_checkpoint.pth]
```

### Verify the physical prior (Table 1)


## Repository Structure

```
DS-MIN/
├── preprocess_data.py      # PLY → .npy pipeline
├── dataset.py              # PyTorch Dataset (8192-pt stratified sampling)
├── model.py                # DS-MIN architecture
├── shading_extractor.py    # ShadingFeatureEnhancer (Sobel + residual modulation)
├── loss.py                 # BoundaryAwareJointLoss (DWCE + Lovász-Softmax)
├── train.py                # Single-fold training
├── evaluate.py             # LOOCV + mBIoU evaluation
├── predict.py              # Single-scan inference & visualization
├── requirements.txt
├── README.md
├── DataSet/                # (user-provided) Raw .ply files
├── train_data_v2/          # Preprocessed .npy + _roi.png + _norm.json
├── train_images_v2/        # Corresponding 2D .png images
├── TestData/               # Example scan for inference
└── checkpoints/            # Saved model weights
```

## Dataset & Reproducibility

The MEPS (Multi-modal Electrode Part Segmentation) dataset consists of 30 structured-light scans (2448 × 2048 px) of welding electrode assemblies captured with an industrial 3D scanner (Z-axis precision 0.015 mm). Five functional classes are annotated: background, electrode rod, electrode cap, transition region, and arm.

Due to commercial confidentiality, absolute spatial scales and CAD-aligned coordinates are obfuscated. We provide:

- **Anonymised benchmark package**: zero-mean normalised point clouds with local topology and shading information preserved. The normalisation parameters (`_norm.json`) produced by `preprocess_data.py` define the exact mean-centering and max-distance scaling applied. The `.npy` files, RoI masks, and 2D images in `train_data_v2/` and `train_images_v2/` constitute the self-contained benchmark.
- **Complete evaluation protocol**: `evaluate.py` implements the full LOOCV with the same chunking and mBIoU boundary-band computation used in the paper.
- **Standardised split file**: The LOOCV partition is deterministic — `evaluate.py` iterates over the sorted file list in `train_data_v2/`, holding out exactly one sample per fold. No random train/test splitting is involved.

To reproduce the full experiment on your own data:

1. Collect structured-light scans of homogeneous metallic assemblies with corresponding 2D images.
2. Annotate point clouds using the 5-class functional label scheme described in Section 4.1.
3. Run `preprocess_data.py` to generate normalised `.npy` files.
4. Run `evaluate.py` for LOOCV results.






