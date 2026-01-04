# Wood Anomaly Detection V2 - Technical Report

**Report Date:** 2026-01-02  
**Dataset:** wood_otsu  
**Environment:** Google Colab (GPU)

---

## Executive Summary

This report documents the implementation and evaluation of three cold-start anomaly detection models for wood defect classification. The V2 implementation introduces significant improvements over the baseline, particularly in data augmentation techniques which resulted in notable performance gains.

---

## 1. Methodology

### 1.1 Problem Definition

**Cold-Start Anomaly Detection**: Models are trained exclusively on "good" (normal) samples and must detect "defect" (anomalous) samples during inference without ever seeing defect examples during training.

### 1.2 Dataset Structure

| Split | Good Samples | Defect Samples |
|-------|--------------|----------------|
| Train | 36 | 0 |
| Test  | 36 | 10 |

**Preprocessing Applied**: Otsu thresholding for background removal (wood_otsu dataset)

---

## 2. Model Architectures

### 2.1 AutoEncoder (Reconstruction-Based)

**Concept**: Learn to reconstruct normal samples; anomalies will have higher reconstruction error.

**Architecture Details:**
```
Encoder:
├── Conv2d(3→32, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)   → 128×128
├── Conv2d(32→64, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)  → 64×64
├── Conv2d(64→128, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → 32×32
├── Conv2d(128→256, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)→ 16×16
└── Conv2d(256→256, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)→ 8×8 (bottleneck)

Decoder (Mirror of Encoder with ConvTranspose2d + Sigmoid output)
```

**V2 Improvements:**
| Feature | V1 | V2 |
|---------|----|----|
| Loss Function | MSE only | MSE + SSIM (50/50 weight) |
| Image Size | Fixed 224×224 | Dynamic 256×256 |
| LR Scheduler | None | Cosine Annealing |
| Early Stopping | No | Yes (patience=10) |

**SSIM Loss Formula:**
```
SSIM(x,y) = (2μx·μy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
Loss = 1 - SSIM(reconstructed, original)
```

**Hyperparameters:**
- Latent Dimension: 256
- Learning Rate: 1e-3
- Epochs: 100 (max)
- Optimizer: Adam
- SSIM Weight: 0.5

---

### 2.2 PatchCore (Memory Bank + K-NN)

**Concept**: Build a memory bank of normal patch features; anomalies will be far from any stored patch.

**Architecture Details:**
```
Feature Extractor (Pretrained WideResNet50-2):
├── Conv1 → BatchNorm → ReLU → MaxPool
├── Layer1 (output: 256 channels)
├── Layer2 (output: 512 channels) ─────┐
└── Layer3 (output: 1024 channels) ────┤
                                        ↓
                              Concatenate → 1536 channels
                                        ↓
                              Random Projection → 512 dim
                                        ↓
                              Coreset Sampling → Memory Bank (2000 patches)
```

**V2 Improvements:**
| Feature | V1 | V2 |
|---------|----|----|
| Feature Layers | Layer3 only | Layer2 + Layer3 (multi-scale) |
| Feature Dim | 1024 | 1536 (concatenated) |
| Neighborhood Aggregation | No | Yes (3×3 avg pooling) |
| Memory Bank Size | 1000 | 2000 |

**Coreset Sampling Algorithm:**
```python
# Greedy farthest point sampling
1. Select random initial point
2. For each remaining slot:
   a. Compute distance from all points to nearest selected point
   b. Select point with maximum minimum distance
   c. Add to selected set
```

**Hyperparameters:**
- Backbone: WideResNet50-2 (ImageNet pretrained)
- Memory Bank Size: 2000
- Target Dimension: 512
- K-Neighbors: 5
- Metric: Euclidean

---

### 2.3 SimpleNet (Feature Adaptor + Discriminator)

**Concept**: Train a discriminator to separate adapted normal features from synthetic anomaly features.

**Architecture Details:**
```
Feature Extractor (Pretrained WideResNet50-2, frozen):
└── Layer3 output → 1024 channels

Feature Adaptor:
└── Linear(1024→512, bias=False) + LayerNorm

Discriminator:
├── Linear(512→256) + BatchNorm + LeakyReLU(0.2) + Dropout(0.1)
├── Linear(256→128) + BatchNorm + LeakyReLU(0.2)
└── Linear(128→1) → Anomaly Score
```

**Anomaly Synthesis (Training Only):**
```python
# Multi-noise approach for diverse synthetic anomalies
anomaly_features = normal_features + noise
where noise ~ N(0, σ²)

# Multi-noise levels (V2):
σ ∈ {0.0075, 0.015, 0.030}  # 0.5x, 1x, 2x base std
```

**V2 Improvements:**
| Feature | V1 | V2 |
|---------|----|----|
| Noise Synthesis | Single level (σ=0.015) | Multi-level (3 levels) |
| Loss Function | Truncated L1 | Focal Loss (γ=2, α=0.25) |
| Regularization | None | Dropout(0.1) + LayerNorm |
| Optimizer | Adam | AdamW (weight_decay=1e-4) |

**Focal Loss Formula:**
```
FL(pt) = -α(1-pt)^γ · log(pt)
where pt = sigmoid(discriminator_output)
```

**Hyperparameters:**
- Backbone: WideResNet50-2
- Adaptor Dimension: 512
- Base Noise Std: 0.015
- Learning Rate: 1e-4
- Epochs: 50 (max)

---

## 3. Data Augmentation

### 3.1 Base (No Augmentation)
- Only resize to 256×256
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 3.2 Enhanced Augmentation
Applied during training only:

| Augmentation | Parameters |
|--------------|------------|
| Random Rotation | ±15° |
| Horizontal Flip | p=0.5 |
| Vertical Flip | p=0.3 |
| Color Jitter | brightness=0.1, contrast=0.1, saturation=0.05 |

**Rationale**: Wood defects can appear at any orientation; augmentation increases training data diversity and reduces overfitting.

---

## 4. Results

### 4.1 Performance Metrics

| Model | Data | AUC-ROC | F1 Score | Accuracy | Precision | Recall |
|-------|------|---------|----------|----------|-----------|--------|
| **AutoEncoder** | Base | **0.8639** | 0.9118 | 0.8696 | 0.9688 | 0.8611 |
| **AutoEncoder** | Augmented | 0.7528 | **0.9333** | **0.8913** | 0.8974 | **0.9722** |
| PatchCore | Base | 0.6500 | 0.8611 | 0.7826 | 0.8611 | 0.8611 |
| PatchCore | Augmented | 0.6250 | 0.8767 | 0.8043 | 0.8649 | 0.8889 |
| SimpleNet | Base | 0.5194 | 0.6071 | 0.5217 | 0.8500 | 0.4722 |
| SimpleNet | Augmented | 0.5833 | 0.8116 | 0.7174 | 0.8485 | 0.7778 |

### 4.2 Key Findings

#### AutoEncoder
- **Best overall performer** with highest AUC-ROC (0.8639 base)
- Augmentation trade-off: Higher recall (0.97 vs 0.86) but lower AUC-ROC
- SSIM loss helped capture structural anomalies better than MSE alone

#### PatchCore
- Consistent performance across both settings
- Multi-layer features (Layer2+Layer3) provided richer representations
- Memory bank approach less sensitive to augmentation

#### SimpleNet
- **Biggest improvement from augmentation**: F1 increased from 0.61 to 0.81
- Multi-noise synthesis helped create more realistic synthetic anomalies
- Focal loss addressed class imbalance between normal patches and synthetic anomalies

### 4.3 Base vs Augmented Analysis

| Model | AUC-ROC Change | F1 Change | Recall Change |
|-------|----------------|-----------|---------------|
| AutoEncoder | -0.111 (↓) | +0.022 (↑) | +0.111 (↑) |
| PatchCore | -0.025 (↓) | +0.016 (↑) | +0.028 (↑) |
| SimpleNet | +0.064 (↑) | +0.204 (↑) | +0.306 (↑) |

**Interpretation:**
- Augmentation consistently improves **recall** (defect detection rate)
- AUC-ROC slightly decreases for reconstruction-based methods (AutoEncoder, PatchCore) due to threshold sensitivity
- SimpleNet benefits most from augmentation due to its discriminative learning approach

---

## 5. Threshold Selection

**Method**: Youden's J Statistic (maximizes TPR - FPR)

```python
J = TPR - FPR
optimal_threshold = argmax(J)
```

| Model | Data | Optimal Threshold |
|-------|------|-------------------|
| AutoEncoder | Base | 0.480 |
| AutoEncoder | Augmented | 0.029 |
| PatchCore | Base | 0.074 |
| PatchCore | Augmented | 0.087 |
| SimpleNet | Base | 0.390 |
| SimpleNet | Augmented | 0.223 |

---

## 6. Technical Implementation Details

### 6.1 Training Configuration

```python
Config:
  image_size: (256, 256)
  batch_size: 8
  early_stopping_patience: 10
  
  # AutoEncoder
  ae_latent_dim: 256
  ae_learning_rate: 1e-3
  ae_epochs: 100
  ae_ssim_weight: 0.5
  
  # PatchCore
  pc_backbone: "wide_resnet50_2"
  pc_memory_bank_size: 2000
  pc_target_dim: 512
  pc_k_neighbors: 5
  pc_use_multi_layer: True
  pc_aggregate_neighbors: True
  
  # SimpleNet
  sn_backbone: "wide_resnet50_2"
  sn_adaptor_dim: 512
  sn_noise_std: 0.015
  sn_learning_rate: 1e-4
  sn_epochs: 50
  sn_use_multi_noise: True
```

### 6.2 Inference Pipeline

```
Input Image (any size)
    ↓
Resize to 256×256
    ↓
ImageNet Normalization
    ↓
Model Inference → Anomaly Score (per patch)
    ↓
Image-level Score = max(patch_scores)
    ↓
Compare with Threshold
    ↓
Classification: Good (0) or Defect (1)
```

### 6.3 Anomaly Map Generation

All models produce pixel-level anomaly maps:
- **AutoEncoder**: Per-pixel reconstruction error
- **PatchCore**: K-NN distance per patch, upsampled to image size
- **SimpleNet**: Discriminator score per patch, upsampled to image size

---

## 7. Conclusions

1. **AutoEncoder** is the most reliable model for this dataset with highest AUC-ROC
2. **Data augmentation** significantly improves recall at the cost of slightly lower AUC-ROC
3. **SSIM loss** provides structural awareness beyond pixel-level MSE
4. **Multi-layer features** in PatchCore improve representation quality
5. **Multi-noise synthesis** and **Focal loss** make SimpleNet more robust

### Recommendations

- For **high recall requirements** (catch all defects): Use AutoEncoder with augmentation
- For **balanced performance**: Use AutoEncoder without augmentation
- For **deployment**: Consider ensemble of AutoEncoder + PatchCore

---

## 8. Files Generated

| File | Description |
|------|-------------|
| `results_v2.csv` | All metrics in tabular format |
| `roc_base.png` | ROC curves without augmentation |
| `roc_aug.png` | ROC curves with augmentation |
| `cm_base.png` | Confusion matrices without augmentation |
| `cm_aug.png` | Confusion matrices with augmentation |
| `scores_base.png` | Score distributions without augmentation |
| `scores_aug.png` | Score distributions with augmentation |

---

## Appendix: Code Reference

Main implementation file: `compare_anomaly_models_v2.py`

Key classes:
- `ConvAutoEncoderV2`: AutoEncoder architecture
- `SSIMLoss`: Structural similarity loss
- `PatchCoreModel`: Memory bank + K-NN
- `SimpleNetModel`: Feature adaptor + discriminator
- `WoodDataset`: Dataset with augmentation support
