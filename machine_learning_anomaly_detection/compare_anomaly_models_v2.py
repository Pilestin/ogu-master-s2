"""
Wood Anomaly Detection V2 - Three Models Comparison (Improved)
===============================================================
Cold-start anomaly detection için üç farklı yöntemi karşılaştırır:
1. AutoEncoder (Reconstruction-based) - SSIM Loss eklendi
2. PatchCore (Memory bank + K-NN) - Multi-layer features
3. SimpleNet (Feature Adaptor + Discriminator) - Better noise synthesis

Improvements in V2:
- AutoEncoder: SSIM loss, flexible image size support
- PatchCore: Layer2 + Layer3 multi-scale features, neighborhood aggregation
- SimpleNet: Better anomaly synthesis, multi-scale features
- General: Early stopping, better validation, cleaner code

Author: Improved version for Colab (H100/A100)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Tuple, List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Scikit-learn
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import seaborn as sns


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Centralized configuration"""
    # Environment
    environment: str = "colab"  # "colab" or "local"
    
    # Dataset
    dataset_name: str = "wood_otsu"  # Change this to select dataset
    image_size: Tuple[int, int] = (256, 256)  # Balanced size for quality and speed
    
    # AutoEncoder
    ae_latent_dim: int = 256
    ae_learning_rate: float = 1e-3
    ae_epochs: int = 100
    ae_use_ssim: bool = True  # NEW: Use SSIM loss
    ae_ssim_weight: float = 0.5  # Weight for SSIM loss
    
    # PatchCore
    pc_backbone: str = "wide_resnet50_2"
    pc_memory_bank_size: int = 2000
    pc_target_dim: int = 512
    pc_k_neighbors: int = 5
    pc_use_multi_layer: bool = True  # NEW: Use Layer2 + Layer3
    pc_aggregate_neighbors: bool = True  # NEW: Local averaging
    
    # SimpleNet
    sn_backbone: str = "wide_resnet50_2"
    sn_adaptor_dim: int = 512
    sn_noise_std: float = 0.015
    sn_learning_rate: float = 1e-4
    sn_epochs: int = 50
    sn_use_multi_noise: bool = True  # NEW: Multiple noise levels
    
    # Training
    batch_size: int = 8
    early_stopping_patience: int = 10
    
    # Augmentation
    augmentation_type: str = "enhanced"  # "none", "basic", "enhanced"


def get_paths(config: Config) -> dict:
    """Get dataset paths based on environment"""
    if config.environment == "colab":
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_path = "/content/drive/MyDrive/Kod/machine_learning/dataset"
        project_root = "/content/drive/MyDrive/Kod/machine_learning"
    else:
        project_root = os.getcwd()
        base_path = os.path.join(project_root, "dataset")
    
    dataset_path = os.path.join(base_path, config.dataset_name)
    
    return {
        "train_good": os.path.join(dataset_path, "train", "good"),
        "test_good": os.path.join(dataset_path, "test", "good"),
        "test_defect": os.path.join(dataset_path, "test", "defect"),
        "results": os.path.join(project_root, "results")  # Changed to results folder
    }


# ============================================================================
# SSIM Loss (for AutoEncoder)
# ============================================================================

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3
        self.window = self._create_window(window_size, sigma, 3)
    
    def _create_window(self, window_size: int, sigma: float, channel: int):
        """Create Gaussian window for SSIM"""
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)"""
        channel = img1.size(1)
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel)
            self.window = self.window.to(img1.device)
            self.channel = channel
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


# ============================================================================
# Dataset
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class WoodDataset(Dataset):
    """Wood anomaly detection dataset with improved augmentation"""
    
    def __init__(self, 
                 good_path: str, 
                 defect_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (256, 256),
                 augmentation_type: str = "none"):
        """
        Args:
            good_path: Path to good (normal) samples
            defect_path: Path to defect samples (for testing)
            image_size: Target image size
            augmentation_type: "none", "basic", or "enhanced"
        """
        self.image_size = image_size
        self.augmentation_type = augmentation_type
        self.transform = self._build_transform()
        
        self.paths = []
        self.labels = []
        
        # Load good samples (label=0)
        if good_path and os.path.exists(good_path):
            for f in Path(good_path).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(0)
        
        # Load defect samples (label=1)
        if defect_path and os.path.exists(defect_path):
            for f in Path(defect_path).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(1)
        
        print(f"  Loaded: {len(self.paths)} images "
              f"(good: {self.labels.count(0)}, defect: {self.labels.count(1)})")
    
    def _build_transform(self):
        """Build transform pipeline"""
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
        ]
        
        if self.augmentation_type == "basic":
            transforms_list.append(
                transforms.RandomRotation(degrees=15)
            )
        elif self.augmentation_type == "enhanced":
            transforms_list.extend([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05
                )
            ])
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transforms_list)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        # Read image (Unicode path support)
        img = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8), 
            cv2.IMREAD_COLOR
        )
        
        if img is None:
            img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply transform
        img_tensor = self.transform(img)
        
        return img_tensor, label, img_path


# ============================================================================
# Model 1: AutoEncoder (Improved with SSIM)
# ============================================================================

class ConvAutoEncoderV2(nn.Module):
    """Improved Convolutional AutoEncoder with flexible size support"""
    
    def __init__(self, latent_dim: int = 256, input_size: int = 256):
        super().__init__()
        self.input_size = input_size
        
        # Calculate feature map size at bottleneck
        # After 5 conv layers with stride 2: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.bottleneck_size = input_size // 32
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(3, 32),      # -> input/2
            self._conv_block(32, 64),     # -> input/4
            self._conv_block(64, 128),    # -> input/8
            self._conv_block(128, 256),   # -> input/16
            self._conv_block(256, latent_dim),  # -> input/32
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self._deconv_block(latent_dim, 256),
            self._deconv_block(256, 128),
            self._deconv_block(128, 64),
            self._deconv_block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def _conv_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def _deconv_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoEncoderModel:
    """Improved AutoEncoder-based anomaly detection"""
    
    def __init__(self, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ConvAutoEncoderV2(
            latent_dim=config.ae_latent_dim,
            input_size=config.image_size[0]
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=config.ae_learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.ae_epochs)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ssim_loss = SSIMLoss() if config.ae_use_ssim else None
        
        # For normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        print(f"AutoEncoder V2 initialized on {self.device}")
        print(f"  Latent dim: {config.ae_latent_dim}")
        print(f"  SSIM loss: {config.ae_use_ssim}")
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize from ImageNet normalization to [0, 1]"""
        return torch.clamp(x * self.std + self.mean, 0, 1)
    
    def fit(self, train_loader: DataLoader):
        """Train AutoEncoder with early stopping"""
        print("\n" + "="*60)
        print("AutoEncoder V2 Training")
        print("="*60)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.ae_epochs):
            total_loss = 0
            
            for batch, labels, paths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.ae_epochs}"):
                batch = batch.to(self.device)
                batch_denorm = self._denormalize(batch)
                
                self.optimizer.zero_grad()
                reconstructed = self.model(batch_denorm)
                
                # Combined loss: MSE + SSIM
                mse = self.mse_loss(reconstructed, batch_denorm).mean()
                
                if self.ssim_loss is not None:
                    ssim = self.ssim_loss(reconstructed, batch_denorm)
                    loss = (1 - self.config.ae_ssim_weight) * mse + self.config.ae_ssim_weight * ssim
                else:
                    loss = mse
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f} (best: {best_loss:.6f})")
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print("\n✅ AutoEncoder V2 training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores"""
        print("\n" + "="*60)
        print("AutoEncoder V2 Prediction")
        print("="*60)
        
        self.model.eval()
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
            batch_denorm = self._denormalize(batch)
            
            reconstructed = self.model(batch_denorm)
            
            # Per-pixel error
            error = (reconstructed - batch_denorm) ** 2
            
            for i in range(error.shape[0]):
                # Image-level score
                img_score = error[i].mean().cpu().numpy()
                all_scores.append(img_score)
                
                # Anomaly map
                anomaly_map = error[i].mean(dim=0).cpu().numpy()
                anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
                all_maps.append(anomaly_map_resized)
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        # Normalize scores
        scores = np.array(all_scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_norm, np.array(all_maps), np.array(all_labels), all_paths
    
    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, filepath)
        print(f"AutoEncoder saved: {filepath}")
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"AutoEncoder loaded: {filepath}")


# ============================================================================
# Model 2: PatchCore (Improved with Multi-Layer Features)
# ============================================================================

class PatchCoreModel:
    """Improved PatchCore with multi-layer feature extraction"""
    
    def __init__(self, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = None
        self.layer2_hook = None
        self.layer3_hook = None
        self.layer2_features = None
        self.layer3_features = None
        
        self.memory_bank = None
        self.random_projection = None
        self.nn_model = None
        self.feature_map_size = None
        
        self._build_feature_extractor()
        
        print(f"PatchCore V2 initialized on {self.device}")
        print(f"  Backbone: {config.pc_backbone}")
        print(f"  Multi-layer: {config.pc_use_multi_layer}")
        print(f"  Neighborhood aggregation: {config.pc_aggregate_neighbors}")
    
    def _build_feature_extractor(self):
        """Build feature extractor with hooks for multi-layer extraction"""
        if self.config.pc_backbone == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
        elif self.config.pc_backbone == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unsupported backbone: {self.config.pc_backbone}")
        
        # Build sequential model
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        ).to(self.device)
        
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Store layer references for multi-layer extraction
        if self.config.pc_use_multi_layer:
            self.layer2 = backbone.layer2.to(self.device)
            self.layer3 = backbone.layer3.to(self.device)
            self.base_layers = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
            ).to(self.device)
            
            self.layer2.eval()
            self.layer3.eval()
            self.base_layers.eval()
            
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.base_layers.parameters():
                param.requires_grad = False
    
    def _aggregate_neighbors(self, features: np.ndarray, h: int, w: int) -> np.ndarray:
        """Apply local neighborhood averaging"""
        if not self.config.pc_aggregate_neighbors:
            return features
        
        # Reshape to spatial grid
        c = features.shape[-1]
        features_spatial = features.reshape(-1, h, w, c)
        
        # Apply average pooling
        aggregated = []
        for feat in features_spatial:
            feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).unsqueeze(0).float()
            pooled = F.avg_pool2d(feat_tensor, kernel_size=3, stride=1, padding=1)
            pooled = pooled.squeeze(0).permute(1, 2, 0).numpy()
            aggregated.append(pooled.reshape(-1, c))
        
        return np.concatenate(aggregated, axis=0)
    
    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader) -> np.ndarray:
        """Extract features with multi-layer support"""
        all_features = []
        
        for batch, labels, paths in tqdm(dataloader, desc="Feature extraction"):
            batch = batch.to(self.device)
            
            if self.config.pc_use_multi_layer:
                # Multi-layer extraction
                base_feat = self.base_layers(batch)
                layer2_feat = self.layer2(base_feat)
                layer3_feat = self.layer3(layer2_feat)
                
                # Get shapes
                b, c2, h2, w2 = layer2_feat.shape
                b, c3, h3, w3 = layer3_feat.shape
                
                # Resize layer2 to match layer3
                layer2_resized = F.interpolate(
                    layer2_feat, size=(h3, w3), mode='bilinear', align_corners=False
                )
                
                # Concatenate features
                combined = torch.cat([layer2_resized, layer3_feat], dim=1)
                
                if self.feature_map_size is None:
                    self.feature_map_size = (h3, w3)
                
                b, c, h, w = combined.shape
                features = combined.permute(0, 2, 3, 1).reshape(b, h*w, c)
            else:
                # Single layer extraction
                features = self.feature_extractor(batch)
                
                if self.feature_map_size is None:
                    self.feature_map_size = (features.shape[2], features.shape[3])
                
                b, c, h, w = features.shape
                features = features.permute(0, 2, 3, 1).reshape(b, h*w, c)
            
            all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        
        # Apply neighborhood aggregation
        h, w = self.feature_map_size
        all_features = all_features.reshape(-1, all_features.shape[-1])
        
        return all_features
    
    def _coreset_sampling(self, features: np.ndarray, k: int) -> np.ndarray:
        """Greedy coreset sampling"""
        if len(features) <= k:
            return features
        
        print(f"  Coreset sampling: {len(features)} -> {k}")
        
        selected_indices = [np.random.randint(len(features))]
        min_distances = np.full(len(features), np.inf)
        
        for i in tqdm(range(1, k), desc="  Coreset sampling"):
            last_selected = features[selected_indices[-1]]
            distances = np.linalg.norm(features - last_selected, axis=1)
            min_distances = np.minimum(min_distances, distances)
            min_distances[selected_indices] = -1
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return features[selected_indices]
    
    def fit(self, train_loader: DataLoader):
        """Build memory bank"""
        print("\n" + "="*60)
        print("PatchCore V2 Training (Memory Bank Creation)")
        print("="*60)
        
        print("\n[1/3] Feature extraction...")
        features = self._extract_features(train_loader)
        print(f"  Extracted features shape: {features.shape}")
        
        # Dimension reduction
        if features.shape[1] > self.config.pc_target_dim:
            print(f"\n[2/3] Dimension reduction: {features.shape[1]} -> {self.config.pc_target_dim}")
            self.random_projection = SparseRandomProjection(
                n_components=self.config.pc_target_dim,
                random_state=42
            )
            features = self.random_projection.fit_transform(features)
        else:
            print("\n[2/3] Dimension reduction: skipped")
        
        # Build memory bank
        print(f"\n[3/3] Building memory bank...")
        if len(features) > self.config.pc_memory_bank_size:
            self.memory_bank = self._coreset_sampling(features, self.config.pc_memory_bank_size)
        else:
            self.memory_bank = features
        
        print(f"  Memory bank shape: {self.memory_bank.shape}")
        
        self.nn_model = NearestNeighbors(
            n_neighbors=self.config.pc_k_neighbors,
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        )
        self.nn_model.fit(self.memory_bank)
        
        print("\n✅ PatchCore V2 training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores"""
        print("\n" + "="*60)
        print("PatchCore V2 Prediction")
        print("="*60)
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
            
            if self.config.pc_use_multi_layer:
                base_feat = self.base_layers(batch)
                layer2_feat = self.layer2(base_feat)
                layer3_feat = self.layer3(layer2_feat)
                
                h3, w3 = layer3_feat.shape[2], layer3_feat.shape[3]
                layer2_resized = F.interpolate(
                    layer2_feat, size=(h3, w3), mode='bilinear', align_corners=False
                )
                features = torch.cat([layer2_resized, layer3_feat], dim=1)
            else:
                features = self.feature_extractor(batch)
            
            b, c, h, w = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
            
            if self.random_projection is not None:
                features_flat = self.random_projection.transform(features_flat)
            
            distances, _ = self.nn_model.kneighbors(features_flat)
            distances = distances.mean(axis=1)
            
            n_patches = h * w
            for i in range(b):
                start_idx = i * n_patches
                end_idx = start_idx + n_patches
                
                img_distances = distances[start_idx:end_idx]
                img_score = np.max(img_distances)
                all_scores.append(img_score)
                
                anomaly_map = img_distances.reshape(h, w)
                anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
                all_maps.append(anomaly_map_resized)
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        scores = np.array(all_scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_norm, np.array(all_maps), np.array(all_labels), all_paths
    
    def save(self, filepath: str):
        model_data = {
            'memory_bank': self.memory_bank,
            'random_projection': self.random_projection,
            'nn_model': self.nn_model,
            'config': self.config,
            'feature_map_size': self.feature_map_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"PatchCore saved: {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.memory_bank = model_data['memory_bank']
        self.random_projection = model_data['random_projection']
        self.nn_model = model_data['nn_model']
        self.feature_map_size = model_data.get('feature_map_size')
        print(f"PatchCore loaded: {filepath}")


# ============================================================================
# Model 3: SimpleNet (Improved with Multi-Noise)
# ============================================================================

class FeatureAdaptor(nn.Module):
    """Feature Adaptor with optional layer norm"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.adaptor = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.LayerNorm(out_features),
        )
    
    def forward(self, x):
        return self.adaptor(x)


class Discriminator(nn.Module):
    """Improved discriminator with dropout"""
    
    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        return self.mlp(x)


class SimpleNetModel:
    """Improved SimpleNet with multi-noise synthesis"""
    
    def __init__(self, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = None
        self.feature_adaptor = None
        self.discriminator = None
        self.feature_dim = None
        self.feature_map_size = None
        
        self._build_feature_extractor()
        
        print(f"SimpleNet V2 initialized on {self.device}")
        print(f"  Backbone: {config.sn_backbone}")
        print(f"  Multi-noise: {config.sn_use_multi_noise}")
    
    def _build_feature_extractor(self):
        """Build feature extractor"""
        if self.config.sn_backbone == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
            self.feature_dim = 1024  # Layer3 output
        elif self.config.sn_backbone == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.feature_dim = 256
        else:
            raise ValueError(f"Unsupported backbone: {self.config.sn_backbone}")
        
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        ).to(self.device)
        
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_adaptor = FeatureAdaptor(
            self.feature_dim, self.config.sn_adaptor_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.config.sn_adaptor_dim
        ).to(self.device)
    
    def _generate_anomaly_features(self, normal_features: torch.Tensor) -> torch.Tensor:
        """Generate synthetic anomaly features with multi-noise"""
        if self.config.sn_use_multi_noise:
            # Multiple noise levels for diversity
            noise_levels = [
                self.config.sn_noise_std * 0.5,
                self.config.sn_noise_std,
                self.config.sn_noise_std * 2.0,
            ]
            
            batch_size = normal_features.shape[0] // 3
            anomaly_parts = []
            
            for i, std in enumerate(noise_levels):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < 2 else normal_features.shape[0]
                part = normal_features[start_idx:end_idx]
                noise = torch.randn_like(part) * std
                anomaly_parts.append(part + noise)
            
            return torch.cat(anomaly_parts, dim=0)
        else:
            noise = torch.randn_like(normal_features) * self.config.sn_noise_std
            return normal_features + noise
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
        """Focal loss for class imbalance"""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = alpha * (1 - pt) ** gamma
        return (focal_weight * bce).mean()
    
    def fit(self, train_loader: DataLoader):
        """Train SimpleNet"""
        print("\n" + "="*60)
        print("SimpleNet V2 Training")
        print("="*60)
        
        optimizer = AdamW(
            list(self.feature_adaptor.parameters()) + list(self.discriminator.parameters()),
            lr=self.config.sn_learning_rate,
            weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.sn_epochs)
        
        self.feature_adaptor.train()
        self.discriminator.train()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.sn_epochs):
            total_loss = 0
            
            for batch, labels, paths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.sn_epochs}"):
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(batch)
                    
                    if self.feature_map_size is None:
                        self.feature_map_size = (features.shape[2], features.shape[3])
                
                b, c, h, w = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, c)
                
                adapted_features = self.feature_adaptor(features_flat)
                anomaly_features = self._generate_anomaly_features(adapted_features.detach())
                
                optimizer.zero_grad()
                
                # Normal -> 0, Anomaly -> 1
                normal_pred = self.discriminator(adapted_features)
                normal_target = torch.zeros_like(normal_pred)
                
                anomaly_pred = self.discriminator(anomaly_features)
                anomaly_target = torch.ones_like(anomaly_pred)
                
                # Focal loss
                loss_normal = self._focal_loss(normal_pred, normal_target)
                loss_anomaly = self._focal_loss(anomaly_pred, anomaly_target)
                loss = loss_normal + loss_anomaly
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print("\n✅ SimpleNet V2 training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores"""
        print("\n" + "="*60)
        print("SimpleNet V2 Prediction")
        print("="*60)
        
        self.feature_adaptor.eval()
        self.discriminator.eval()
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
            
            features = self.feature_extractor(batch)
            b, c, h, w = features.shape
            
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, c)
            adapted_features = self.feature_adaptor(features_flat)
            
            scores = torch.sigmoid(self.discriminator(adapted_features)).squeeze()
            scores = scores.cpu().numpy()
            
            n_patches = h * w
            for i in range(b):
                start_idx = i * n_patches
                end_idx = start_idx + n_patches
                
                img_scores = scores[start_idx:end_idx]
                img_score = np.max(img_scores)
                all_scores.append(img_score)
                
                anomaly_map = img_scores.reshape(h, w)
                anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
                all_maps.append(anomaly_map_resized)
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        scores = np.array(all_scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_norm, np.array(all_maps), np.array(all_labels), all_paths
    
    def save(self, filepath: str):
        torch.save({
            'feature_adaptor': self.feature_adaptor.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'config': self.config,
            'feature_map_size': self.feature_map_size
        }, filepath)
        print(f"SimpleNet saved: {filepath}")
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.feature_adaptor.load_state_dict(checkpoint['feature_adaptor'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.feature_map_size = checkpoint.get('feature_map_size')
        print(f"SimpleNet loaded: {filepath}")


# ============================================================================
# Evaluation & Visualization
# ============================================================================

def calculate_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate all metrics"""
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def plot_roc_comparison(results: Dict[str, dict], save_path: str = None):
    """Plot ROC curves comparison"""
    plt.figure(figsize=(10, 8))
    
    colors = {'AutoEncoder': 'blue', 'PatchCore': 'green', 'SimpleNet': 'red'}
    
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['labels'], result['scores'])
        auc = roc_auc_score(result['labels'], result['scores'])
        
        color = colors.get(model_name, 'gray')
        plt.plot(fpr, tpr, color=color, linewidth=2, 
                 label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison (V2)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results: Dict[str, dict], save_path: str = None):
    """Plot confusion matrices"""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['labels'], result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Good', 'Defect'],
                    yticklabels=['Good', 'Defect'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f'{model_name}')
    
    plt.suptitle('Confusion Matrices (V2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distributions(results: Dict[str, dict], save_path: str = None):
    """Plot score distributions"""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        labels = result['labels']
        scores = result['scores']
        threshold = result['metrics']['threshold']
        
        good_scores = scores[labels == 0]
        defect_scores = scores[labels == 1]
        
        axes[idx].hist(good_scores, bins=20, alpha=0.6, label='Good', color='green')
        axes[idx].hist(defect_scores, bins=20, alpha=0.6, label='Defect', color='red')
        axes[idx].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                          label=f'Threshold={threshold:.3f}')
        axes[idx].set_xlabel('Anomaly Score')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'{model_name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Score Distributions (V2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_results(results: Dict[str, dict], title: str = "Results"):
    """Print results in table format"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    print(f"{'Model':<15} {'AUC-ROC':<10} {'F1':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*80)
    
    for model_name, result in results.items():
        m = result['metrics']
        print(f"{model_name:<15} {m['auc_roc']:<10.4f} {m['f1_score']:<10.4f} "
              f"{m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f}")
    
    print("="*80)


def save_anomaly_heatmaps(results: Dict[str, dict], 
                          output_dir: str,
                          data_type: str = "base"):
    """
    Save anomaly heatmaps for each test image.
    
    Creates folder structure:
    output_dir/
    └── test_images/
        └── {data_type}/
            └── {model_name}/
                ├── good/
                │   └── image_001_score_0.123.png
                └── defect/
                    └── image_001_score_0.789.png
    """
    print(f"\n📸 Saving anomaly heatmaps ({data_type})...")
    
    for model_name, result in results.items():
        maps = result['maps']
        scores = result['scores']
        labels = result['labels']
        paths = result['paths']
        threshold = result['metrics']['threshold']
        
        # Create directories
        model_dir = os.path.join(output_dir, "test_images", data_type, model_name)
        good_dir = os.path.join(model_dir, "good")
        defect_dir = os.path.join(model_dir, "defect")
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(defect_dir, exist_ok=True)
        
        for i, (anomaly_map, score, label, img_path) in enumerate(zip(maps, scores, labels, paths)):
            # Get original image
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            
            # Normalize anomaly map to 0-255
            anomaly_map_normalized = anomaly_map.copy()
            if anomaly_map_normalized.max() > anomaly_map_normalized.min():
                anomaly_map_normalized = (anomaly_map_normalized - anomaly_map_normalized.min()) / \
                                         (anomaly_map_normalized.max() - anomaly_map_normalized.min())
            anomaly_map_uint8 = (anomaly_map_normalized * 255).astype(np.uint8)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay on original image
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Anomaly heatmap
            axes[1].imshow(anomaly_map, cmap='jet')
            axes[1].set_title(f'Anomaly Map\nScore: {score:.4f}')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(overlay)
            pred_label = "DEFECT" if score >= threshold else "GOOD"
            true_label = "DEFECT" if label == 1 else "GOOD"
            color = 'green' if pred_label == true_label else 'red'
            axes[2].set_title(f'Overlay\nPred: {pred_label} | True: {true_label}', color=color)
            axes[2].axis('off')
            
            plt.suptitle(f'{model_name} - {os.path.basename(img_path)}', fontsize=10)
            plt.tight_layout()
            
            # Determine save directory based on TRUE label
            save_dir = defect_dir if label == 1 else good_dir
            
            # Save with score in filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(save_dir, f"{base_name}_score_{score:.4f}.png")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ {model_name}: Saved {len(maps)} heatmaps")
    
    print(f"  📁 Heatmaps saved to: {os.path.join(output_dir, 'test_images', data_type)}")


# ============================================================================
# Main Comparison Function
# ============================================================================

def run_comparison(config: Config, use_augmentation: bool = False) -> Dict[str, dict]:
    """Run comparison of all three models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = get_paths(config)
    
    aug_str = "with augmentation" if use_augmentation else "without augmentation"
    print("\n" + "="*80)
    print(f"🔬 RUNNING COMPARISON V2 ({aug_str.upper()})")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: {config.dataset_name}")
    
    # Datasets
    train_dataset = WoodDataset(
        good_path=paths["train_good"],
        defect_path=None,
        image_size=config.image_size,
        augmentation_type=config.augmentation_type if use_augmentation else "none"
    )
    
    test_dataset = WoodDataset(
        good_path=paths["test_good"],
        defect_path=paths["test_defect"],
        image_size=config.image_size,
        augmentation_type="none"
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train (good only): {len(train_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    results = {}
    
    # =========================================================================
    # Model 1: AutoEncoder
    # =========================================================================
    print("\n" + "-"*60)
    print("📦 MODEL 1: AutoEncoder V2")
    print("-"*60)
    
    ae_model = AutoEncoderModel(config, device=device)
    ae_model.fit(train_loader)
    
    ae_scores, ae_maps, ae_labels, ae_paths = ae_model.predict(test_loader)
    ae_threshold = find_optimal_threshold(ae_labels, ae_scores)
    ae_metrics = calculate_metrics(ae_labels, ae_scores, ae_threshold)
    
    results['AutoEncoder'] = {
        'scores': ae_scores,
        'maps': ae_maps,
        'labels': ae_labels,
        'paths': ae_paths,
        'predictions': (ae_scores >= ae_threshold).astype(int),
        'metrics': ae_metrics
    }
    
    # =========================================================================
    # Model 2: PatchCore
    # =========================================================================
    print("\n" + "-"*60)
    print("📦 MODEL 2: PatchCore V2")
    print("-"*60)
    
    pc_model = PatchCoreModel(config, device=device)
    pc_model.fit(train_loader)
    
    pc_scores, pc_maps, pc_labels, pc_paths = pc_model.predict(test_loader)
    pc_threshold = find_optimal_threshold(pc_labels, pc_scores)
    pc_metrics = calculate_metrics(pc_labels, pc_scores, pc_threshold)
    
    results['PatchCore'] = {
        'scores': pc_scores,
        'maps': pc_maps,
        'labels': pc_labels,
        'paths': pc_paths,
        'predictions': (pc_scores >= pc_threshold).astype(int),
        'metrics': pc_metrics
    }
    
    # =========================================================================
    # Model 3: SimpleNet
    # =========================================================================
    print("\n" + "-"*60)
    print("📦 MODEL 3: SimpleNet V2")
    print("-"*60)
    
    sn_model = SimpleNetModel(config, device=device)
    sn_model.fit(train_loader)
    
    sn_scores, sn_maps, sn_labels, sn_paths = sn_model.predict(test_loader)
    sn_threshold = find_optimal_threshold(sn_labels, sn_scores)
    sn_metrics = calculate_metrics(sn_labels, sn_scores, sn_threshold)
    
    results['SimpleNet'] = {
        'scores': sn_scores,
        'maps': sn_maps,
        'labels': sn_labels,
        'paths': sn_paths,
        'predictions': (sn_scores >= sn_threshold).astype(int),
        'metrics': sn_metrics
    }
    
    return results


def main(quick_test: bool = False, dataset_name: str = "wood_otsu"):
    """Main function"""
    print("\n" + "="*80)
    print("🔬 WOOD ANOMALY DETECTION V2 - THREE MODELS COMPARISON")
    print("="*80)
    
    # Configuration
    config = Config()
    config.dataset_name = dataset_name
    
    if quick_test:
        config.ae_epochs = 10
        config.sn_epochs = 10
        config.early_stopping_patience = 5
        print("\n⚡ QUICK TEST MODE - Reduced epochs")
    
    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📌 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output directory: results/result_YYYYMMDD_HHMMSS
    paths = get_paths(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(paths["results"], f"result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Run comparisons
    results_base = run_comparison(config, use_augmentation=False)
    print_results(results_base, "RESULTS V2 - BASE DATA")
    
    results_aug = run_comparison(config, use_augmentation=True)
    print_results(results_aug, "RESULTS V2 - WITH AUGMENTATION")
    
    # Save visualizations
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    plot_roc_comparison(results_base, os.path.join(output_dir, "roc_base.png"))
    plot_roc_comparison(results_aug, os.path.join(output_dir, "roc_aug.png"))
    plot_confusion_matrices(results_base, os.path.join(output_dir, "cm_base.png"))
    plot_confusion_matrices(results_aug, os.path.join(output_dir, "cm_aug.png"))
    plot_score_distributions(results_base, os.path.join(output_dir, "scores_base.png"))
    plot_score_distributions(results_aug, os.path.join(output_dir, "scores_aug.png"))
    
    # Save anomaly heatmaps for each test image
    save_anomaly_heatmaps(results_base, output_dir, data_type="base")
    save_anomaly_heatmaps(results_aug, output_dir, data_type="augmented")
    
    # Save CSV
    rows = []
    for model_name in results_base.keys():
        for data_type, results in [("Base", results_base), ("Augmented", results_aug)]:
            m = results[model_name]['metrics']
            rows.append({
                'Model': model_name,
                'Data': data_type,
                'AUC_ROC': m['auc_roc'],
                'F1_Score': m['f1_score'],
                'Accuracy': m['accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'Threshold': m['threshold']
            })
    
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "results_v2.csv"), index=False)
    
    print("\n✅ COMPARISON V2 COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    
    return results_base, results_aug


if __name__ == "__main__":
    import sys
    
    quick_test = "--quick-test" in sys.argv
    
    # Dataset selection
    dataset = "wood_otsu"  # Default
    for arg in sys.argv:
        if arg.startswith("--dataset="):
            dataset = arg.split("=")[1]
    
    main(quick_test=quick_test, dataset_name=dataset)
