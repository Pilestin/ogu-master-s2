"""
Wood Anomaly Detection V3 - Experiment Framework
=================================================
Hyperparameter testing and comparison framework for cold-start anomaly detection.

Features:
- Grid search for hyperparameter optimization
- Results comparison table (CSV + console)
- Best model selection (auto or manual)
- Final evaluation with heatmaps

Usage:
    # Quick grid search (reduced epochs)
    python compare_anomaly_models_v3.py --mode=grid --quick
    
    # Full grid search
    python compare_anomaly_models_v3.py --mode=grid
    
    # Single model experiment
    python compare_anomaly_models_v3.py --mode=single --model=AutoEncoder
    
    # Final evaluation with specific config
    python compare_anomaly_models_v3.py --mode=final

Author: V3 Experiment Framework
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
import pandas as pd
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

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
# Configuration Classes
# ============================================================================

@dataclass
class BaseConfig:
    """Base configuration"""
    environment: str = "local"
    dataset_name: str = ""
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    augmentation_type: str = "enhanced"
    early_stopping_patience: int = 10


@dataclass
class AutoEncoderConfig:
    """AutoEncoder experiment configuration"""
    latent_dim: int = 256
    learning_rate: float = 1e-3
    epochs: int = 100
    use_ssim: bool = True
    ssim_weight: float = 0.5


@dataclass
class PatchCoreConfig:
    """PatchCore experiment configuration"""
    backbone: str = "wide_resnet50_2"
    memory_bank_size: int = 2000
    target_dim: int = 512
    k_neighbors: int = 5
    use_multi_layer: bool = True
    aggregate_neighbors: bool = True


@dataclass
class SimpleNetConfig:
    """SimpleNet experiment configuration"""
    backbone: str = "wide_resnet50_2"
    adaptor_dim: int = 512
    noise_std: float = 0.015
    learning_rate: float = 1e-4
    epochs: int = 50
    use_multi_noise: bool = True


@dataclass
class EfficientADConfig:
    """EfficientAD experiment configuration"""
    backbone: str = "resnet18"  # Lightweight backbone for speed
    student_dim: int = 384
    autoencoder_dim: int = 384
    learning_rate: float = 1e-4
    epochs: int = 70
    weight_decay: float = 1e-5
    teacher_momentum: float = 0.999


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str = "default"
    base: BaseConfig = field(default_factory=BaseConfig)
    autoencoder: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    simplenet: SimpleNetConfig = field(default_factory=SimpleNetConfig)
    efficientad: EfficientADConfig = field(default_factory=EfficientADConfig)
    
    def to_dict(self) -> dict:
        return {
            'experiment_name': self.experiment_name,
            'base': asdict(self.base),
            'autoencoder': asdict(self.autoencoder),
            'patchcore': asdict(self.patchcore),
            'simplenet': asdict(self.simplenet),
            'efficientad': asdict(self.efficientad)
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        return cls(
            experiment_name=d.get('experiment_name', 'default'),
            base=BaseConfig(**d.get('base', {})),
            autoencoder=AutoEncoderConfig(**d.get('autoencoder', {})),
            patchcore=PatchCoreConfig(**d.get('patchcore', {})),
            simplenet=SimpleNetConfig(**d.get('simplenet', {})),
            efficientad=EfficientADConfig(**d.get('efficientad', {}))
        )


def get_paths(config: BaseConfig) -> dict:
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
        "results": os.path.join(project_root, "results")
    }


# ============================================================================
# SSIM Loss
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
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel)
            self.window = self.window.to(img1.device)
            self.channel = channel
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


# ============================================================================
# Dataset
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class WoodDataset(Dataset):
    """Wood anomaly detection dataset"""
    
    def __init__(self, good_path: str, defect_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (256, 256), augmentation_type: str = "none"):
        self.image_size = image_size
        self.augmentation_type = augmentation_type
        self.transform = self._build_transform()
        self.paths, self.labels = [], []
        
        if good_path and os.path.exists(good_path):
            for f in Path(good_path).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(0)
        
        if defect_path and os.path.exists(defect_path):
            for f in Path(defect_path).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(1)
    
    def _build_transform(self):
        transforms_list = [transforms.ToPILImage(), transforms.Resize(self.image_size)]
        
        if self.augmentation_type == "basic":
            transforms_list.append(transforms.RandomRotation(degrees=15))
        elif self.augmentation_type == "enhanced":
            transforms_list.extend([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
            ])
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms.Compose(transforms_list)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return self.transform(img), label, img_path


# ============================================================================
# Models (Simplified for experiments)
# ============================================================================

class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, input_size: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            self._conv(3, 32), self._conv(32, 64), self._conv(64, 128),
            self._conv(128, 256), self._conv(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            self._deconv(latent_dim, 256), self._deconv(256, 128),
            self._deconv(128, 64), self._deconv(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    
    def _conv(self, i, o):
        return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))
    
    def _deconv(self, i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoderModel:
    def __init__(self, config: AutoEncoderConfig, base_config: BaseConfig, device: str = None):
        self.config = config
        self.base_config = base_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ConvAutoEncoder(config.latent_dim, base_config.image_size[0]).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ssim_loss = SSIMLoss() if config.use_ssim else None
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def _denorm(self, x):
        return torch.clamp(x * self.std + self.mean, 0, 1)
    
    def fit(self, train_loader: DataLoader, verbose: bool = True):
        self.model.train()
        best_loss, patience = float('inf'), 0
        
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch, _, _ in train_loader:
                batch = self._denorm(batch.to(self.device))
                self.optimizer.zero_grad()
                recon = self.model(batch)
                mse = self.mse_loss(recon, batch).mean()
                loss = mse if not self.ssim_loss else (1 - self.config.ssim_weight) * mse + self.config.ssim_weight * self.ssim_loss(recon, batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss, patience = avg_loss, 0
            else:
                patience += 1
            
            if patience >= self.base_config.early_stopping_patience:
                if verbose: print(f"  Early stop at {epoch+1}")
                break
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        self.model.eval()
        all_scores, all_maps, all_labels, all_paths = [], [], [], []
        
        for batch, labels, paths in test_loader:
            batch = self._denorm(batch.to(self.device))
            recon = self.model(batch)
            error = (recon - batch) ** 2
            
            for i in range(error.shape[0]):
                all_scores.append(error[i].mean().cpu().numpy())
                amap = error[i].mean(dim=0).cpu().numpy()
                all_maps.append(cv2.resize(amap, (256, 256)))
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        scores = np.array(all_scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8), np.array(all_maps), np.array(all_labels), all_paths


class PatchCoreModel:
    def __init__(self, config: PatchCoreConfig, base_config: BaseConfig, device: str = None):
        self.config = config
        self.base_config = base_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.memory_bank = None
        self.random_projection = None
        self.nn_model = None
        self.feature_map_size = None
        
        self._build_extractor()
    
    def _build_extractor(self):
        if self.config.backbone == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
        else:
            backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        self.feature_extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        ).to(self.device).eval()
        
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        
        if self.config.use_multi_layer:
            self.base_layers = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
            ).to(self.device).eval()
            self.layer2 = backbone.layer2.to(self.device).eval()
            self.layer3 = backbone.layer3.to(self.device).eval()
            for p in list(self.base_layers.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()):
                p.requires_grad = False
    
    @torch.no_grad()
    def _extract_features(self, loader):
        all_feats = []
        for batch, _, _ in loader:
            batch = batch.to(self.device)
            if self.config.use_multi_layer:
                base = self.base_layers(batch)
                l2 = self.layer2(base)
                l3 = self.layer3(l2)  # FIXED: was self.layer3(self.layer2(base))
                l2_resized = F.interpolate(l2, size=l3.shape[2:], mode='bilinear', align_corners=False)
                feats = torch.cat([l2_resized, l3], dim=1)
            else:
                feats = self.feature_extractor(batch)
            
            # L2 normalize features (important for PatchCore)
            feats = F.normalize(feats, p=2, dim=1)
            
            if self.feature_map_size is None:
                self.feature_map_size = (feats.shape[2], feats.shape[3])
            
            b, c, h, w = feats.shape
            all_feats.append(feats.permute(0, 2, 3, 1).reshape(b, -1, c).cpu().numpy())
        
        return np.concatenate(all_feats, axis=0).reshape(-1, all_feats[0].shape[-1])
    
    def _coreset_sample(self, feats, k):
        if len(feats) <= k:
            return feats
        
        indices = [np.random.randint(len(feats))]
        min_dists = np.full(len(feats), np.inf)
        
        for _ in range(1, k):
            dists = np.linalg.norm(feats - feats[indices[-1]], axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[indices] = -1
            indices.append(np.argmax(min_dists))
        
        return feats[indices]
    
    def fit(self, train_loader, verbose=True):
        feats = self._extract_features(train_loader)
        
        if verbose:
            print(f"    Total patches extracted: {len(feats)}")
        
        if feats.shape[1] > self.config.target_dim:
            self.random_projection = SparseRandomProjection(n_components=self.config.target_dim, random_state=42)
            feats = self.random_projection.fit_transform(feats)
        
        # memory_bank_size = -1 means use all patches (no coreset sampling)
        if self.config.memory_bank_size == -1 or len(feats) <= self.config.memory_bank_size:
            self.memory_bank = feats
            if verbose:
                print(f"    Using all {len(feats)} patches in memory bank")
        else:
            self.memory_bank = self._coreset_sample(feats, self.config.memory_bank_size)
            if verbose:
                print(f"    Coreset sampled to {len(self.memory_bank)} patches")
        
        self.nn_model = NearestNeighbors(n_neighbors=self.config.k_neighbors, algorithm='auto', metric='euclidean', n_jobs=-1)
        self.nn_model.fit(self.memory_bank)
    
    @torch.no_grad()
    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        all_scores, all_maps, all_labels, all_paths = [], [], [], []
        
        for batch, labels, paths in test_loader:
            batch = batch.to(self.device)
            if self.config.use_multi_layer:
                base = self.base_layers(batch)
                l2 = self.layer2(base)
                l3 = self.layer3(l2)  # FIXED: was self.layer3(self.layer2(base))
                l2_resized = F.interpolate(l2, size=l3.shape[2:], mode='bilinear', align_corners=False)
                feats = torch.cat([l2_resized, l3], dim=1)
            else:
                feats = self.feature_extractor(batch)
            
            # L2 normalize features (same as training)
            feats = F.normalize(feats, p=2, dim=1)
            
            b, c, h, w = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
            
            if self.random_projection:
                feats_flat = self.random_projection.transform(feats_flat)
            
            dists, _ = self.nn_model.kneighbors(feats_flat)
            dists = dists.mean(axis=1)
            
            for i in range(b):
                img_dists = dists[i * h * w:(i + 1) * h * w]
                all_scores.append(np.max(img_dists))
                all_maps.append(cv2.resize(img_dists.reshape(h, w), (256, 256)))
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        scores = np.array(all_scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8), np.array(all_maps), np.array(all_labels), all_paths


class SimpleNetModel:
    def __init__(self, config: SimpleNetConfig, base_config: BaseConfig, device: str = None):
        self.config = config
        self.base_config = base_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_map_size = None
        
        if config.backbone == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
            feat_dim = 1024
        else:
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            feat_dim = 256
        
        self.extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        ).to(self.device).eval()
        
        for p in self.extractor.parameters():
            p.requires_grad = False
        
        self.adaptor = nn.Sequential(nn.Linear(feat_dim, config.adaptor_dim, bias=False), nn.LayerNorm(config.adaptor_dim)).to(self.device)
        self.disc = nn.Sequential(
            nn.Linear(config.adaptor_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Linear(128, 1)
        ).to(self.device)
    
    def _gen_anomaly(self, feats):
        if self.config.use_multi_noise:
            stds = [self.config.noise_std * m for m in [0.5, 1.0, 2.0]]
            bs = feats.shape[0] // 3
            parts = [feats[i*bs:(i+1)*bs if i < 2 else feats.shape[0]] + torch.randn_like(feats[i*bs:(i+1)*bs if i < 2 else feats.shape[0]]) * s for i, s in enumerate(stds)]
            return torch.cat(parts, dim=0)
        return feats + torch.randn_like(feats) * self.config.noise_std
    
    def _focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()
    
    def fit(self, train_loader, verbose=True):
        opt = AdamW(list(self.adaptor.parameters()) + list(self.disc.parameters()), lr=self.config.learning_rate, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=self.config.epochs)
        
        self.adaptor.train()
        self.disc.train()
        best_loss, patience = float('inf'), 0
        
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch, _, _ in train_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    feats = self.extractor(batch)
                    if self.feature_map_size is None:
                        self.feature_map_size = (feats.shape[2], feats.shape[3])
                
                b, c, h, w = feats.shape
                flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
                adapted = self.adaptor(flat)
                anomaly = self._gen_anomaly(adapted.detach())
                
                opt.zero_grad()
                loss = self._focal_loss(self.disc(adapted), torch.zeros(adapted.size(0), 1, device=self.device)) + \
                       self._focal_loss(self.disc(anomaly), torch.ones(anomaly.size(0), 1, device=self.device))
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            sched.step()
            if total_loss / len(train_loader) < best_loss:
                best_loss, patience = total_loss / len(train_loader), 0
            else:
                patience += 1
            if patience >= self.base_config.early_stopping_patience:
                break
    
    @torch.no_grad()
    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        self.adaptor.eval()
        self.disc.eval()
        all_scores, all_maps, all_labels, all_paths = [], [], [], []
        
        for batch, labels, paths in test_loader:
            batch = batch.to(self.device)
            feats = self.extractor(batch)
            b, c, h, w = feats.shape
            flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
            scores = torch.sigmoid(self.disc(self.adaptor(flat))).squeeze().cpu().numpy()
            
            for i in range(b):
                img_scores = scores[i * h * w:(i + 1) * h * w]
                all_scores.append(np.max(img_scores))
                all_maps.append(cv2.resize(img_scores.reshape(h, w), (256, 256)))
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        scores = np.array(all_scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8), np.array(all_maps), np.array(all_labels), all_paths


class EfficientADModel:
    """
    EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
    
    Uses a Student-Teacher architecture with a lightweight autoencoder for
    detecting both structural and logical anomalies.
    """
    
    def __init__(self, config: EfficientADConfig, base_config: BaseConfig, device: str = None):
        self.config = config
        self.base_config = base_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_map_size = None
        
        # Build teacher and student networks
        self._build_networks()
    
    def _build_networks(self):
        """Build PDN (Patch Description Network) teacher, student, and autoencoder"""
        # Use ResNet18 as lightweight backbone
        if self.config.backbone == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            feat_dim = 256
        else:
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
            feat_dim = 1024
        
        # Teacher feature extractor (frozen)
        self.teacher = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        ).to(self.device).eval()
        
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Student network (learnable) - same architecture as teacher
        student_backbone = models.resnet18(weights=None) if self.config.backbone == 'resnet18' else models.wide_resnet50_2(weights=None)
        self.student = nn.Sequential(
            student_backbone.conv1, student_backbone.bn1, student_backbone.relu, student_backbone.maxpool,
            student_backbone.layer1, student_backbone.layer2, student_backbone.layer3
        ).to(self.device)
        
        # Lightweight autoencoder for local anomaly detection
        self.autoencoder = nn.Sequential(
            # Encoder
            nn.Conv2d(feat_dim, self.config.autoencoder_dim, 3, 2, 1),
            nn.BatchNorm2d(self.config.autoencoder_dim),
            nn.ReLU(True),
            nn.Conv2d(self.config.autoencoder_dim, self.config.autoencoder_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(self.config.autoencoder_dim // 2),
            nn.ReLU(True),
            # Decoder
            nn.ConvTranspose2d(self.config.autoencoder_dim // 2, self.config.autoencoder_dim, 4, 2, 1),
            nn.BatchNorm2d(self.config.autoencoder_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.config.autoencoder_dim, feat_dim, 4, 2, 1),
        ).to(self.device)
        
        # Projection heads for student to match teacher dimensions
        self.student_head = nn.Sequential(
            nn.Conv2d(feat_dim, self.config.student_dim, 1),
            nn.BatchNorm2d(self.config.student_dim),
            nn.ReLU(True),
            nn.Conv2d(self.config.student_dim, feat_dim, 1)
        ).to(self.device)
    
    def fit(self, train_loader, verbose=True):
        """Train student and autoencoder on normal samples"""
        params = list(self.student.parameters()) + list(self.autoencoder.parameters()) + list(self.student_head.parameters())
        optimizer = AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        self.student.train()
        self.autoencoder.train()
        self.student_head.train()
        
        best_loss, patience = float('inf'), 0
        
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch, _, _ in train_loader:
                batch = batch.to(self.device)
                
                # Get teacher features (frozen)
                with torch.no_grad():
                    teacher_feats = self.teacher(batch)
                    if self.feature_map_size is None:
                        self.feature_map_size = (teacher_feats.shape[2], teacher_feats.shape[3])
                
                # Get student features
                student_feats = self.student(batch)
                student_feats = self.student_head(student_feats)
                
                # Autoencoder reconstruction
                ae_input = teacher_feats.detach()
                ae_output = self.autoencoder(ae_input)
                
                optimizer.zero_grad()
                
                # Student-Teacher distillation loss
                st_loss = F.mse_loss(student_feats, teacher_feats.detach())
                
                # Autoencoder reconstruction loss
                ae_loss = F.mse_loss(ae_output, ae_input)
                
                # Combined loss
                loss = st_loss + ae_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss, patience = avg_loss, 0
            else:
                patience += 1
            
            if patience >= self.base_config.early_stopping_patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1}")
                break
    
    @torch.no_grad()
    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores using combined ST and AE differences"""
        self.student.eval()
        self.autoencoder.eval()
        self.student_head.eval()
        
        all_scores, all_maps, all_labels, all_paths = [], [], [], []
        
        for batch, labels, paths in test_loader:
            batch = batch.to(self.device)
            
            # Teacher features
            teacher_feats = self.teacher(batch)
            
            # Student features
            student_feats = self.student_head(self.student(batch))
            
            # Autoencoder output
            ae_output = self.autoencoder(teacher_feats)
            
            # Calculate anomaly maps
            # ST difference: where student fails to match teacher
            st_diff = (teacher_feats - student_feats) ** 2
            st_diff = st_diff.mean(dim=1)  # Average over channels
            
            # AE difference: reconstruction error
            ae_diff = (teacher_feats - ae_output) ** 2
            ae_diff = ae_diff.mean(dim=1)
            
            # Combined anomaly map (weighted sum)
            anomaly_map = 0.5 * st_diff + 0.5 * ae_diff
            
            b = batch.shape[0]
            for i in range(b):
                amap = anomaly_map[i].cpu().numpy()
                all_scores.append(np.max(amap))
                all_maps.append(cv2.resize(amap, (256, 256)))
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        scores = np.array(all_scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8), np.array(all_maps), np.array(all_labels), all_paths


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_aupro_approximated(anomaly_maps: np.ndarray, labels: np.ndarray, 
                                  num_thresholds: int = 100) -> float:
    """
    Calculate approximated AUPRO (Area Under Per-Region-Overlap curve).
    
    Since we don't have ground truth masks, this uses a threshold-based approach
    to estimate region overlap. For defect images, we assume high anomaly regions
    correspond to actual defects, and measure consistency across thresholds.
    
    Args:
        anomaly_maps: Anomaly heatmaps for each image (N, H, W)
        labels: Ground truth labels (0=good, 1=defect)
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        Approximated AUPRO score (0-1)
    """
    if len(anomaly_maps) == 0 or np.sum(labels == 1) == 0:
        return 0.0
    
    # Normalize all maps to [0, 1]
    all_maps_flat = anomaly_maps.reshape(-1)
    global_min, global_max = all_maps_flat.min(), all_maps_flat.max()
    if global_max - global_min < 1e-8:
        return 0.5
    
    normalized_maps = (anomaly_maps - global_min) / (global_max - global_min + 1e-8)
    
    # Get defect and good maps separately
    defect_maps = normalized_maps[labels == 1]
    good_maps = normalized_maps[labels == 0]
    
    # Calculate PRO-like metric across thresholds
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_scores = []
    
    for thresh in thresholds:
        # For defects: ratio of pixels above threshold (should be high for good detection)
        defect_coverage = np.mean([np.mean(m > thresh) for m in defect_maps]) if len(defect_maps) > 0 else 0
        
        # For good images: ratio of pixels above threshold (should be low - false positives)
        good_fp_rate = np.mean([np.mean(m > thresh) for m in good_maps]) if len(good_maps) > 0 else 0
        
        # PRO score: balance between catching defects and avoiding false positives
        if defect_coverage + (1 - good_fp_rate) > 0:
            pro_score = 2 * defect_coverage * (1 - good_fp_rate) / (defect_coverage + (1 - good_fp_rate) + 1e-8)
        else:
            pro_score = 0
        
        pro_scores.append(pro_score)
    
    # Area under the PRO curve
    aupro = np.trapz(pro_scores, thresholds)
    return float(np.clip(aupro, 0, 1))


def calculate_metrics(y_true, y_scores, threshold=0.5, anomaly_maps=None):
    """Calculate all metrics including AUPRO"""
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    # Calculate AUPRO if anomaly maps are provided
    if anomaly_maps is not None:
        metrics['aupro'] = calculate_aupro_approximated(anomaly_maps, y_true)
    else:
        metrics['aupro'] = 0.0
    
    return metrics


def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return thresholds[np.argmax(tpr - fpr)]


# ============================================================================
# Heatmap Generation (Fixed)
# ============================================================================

def save_heatmaps(model_name: str, scores: np.ndarray, maps: np.ndarray, 
                  labels: np.ndarray, paths: List[str], threshold: float,
                  output_dir: str):
    """Save anomaly heatmaps for each test image"""
    good_dir = os.path.join(output_dir, "heatmaps", model_name, "good")
    defect_dir = os.path.join(output_dir, "heatmaps", model_name, "defect")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)
    
    for i, (anomaly_map, score, label, img_path) in enumerate(zip(maps, scores, labels, paths)):
        # Read original image
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # Ensure anomaly map is 2D and correct size
        if len(anomaly_map.shape) > 2:
            anomaly_map = anomaly_map.mean(axis=0) if anomaly_map.shape[0] <= 3 else anomaly_map[:, :, 0]
        anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (256, 256))
        
        # Normalize
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            anomaly_map_norm = np.zeros_like(anomaly_map)
        
        # Create heatmap
        heatmap_uint8 = (anomaly_map_norm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        im = axes[1].imshow(anomaly_map_norm, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f'Anomaly Map\nScore: {score:.4f}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        axes[2].imshow(overlay)
        pred = "DEFECT" if score >= threshold else "GOOD"
        true = "DEFECT" if label == 1 else "GOOD"
        color = 'green' if pred == true else 'red'
        axes[2].set_title(f'Overlay\nPred: {pred} | True: {true}', color=color)
        axes[2].axis('off')
        
        plt.suptitle(f'{model_name} - {os.path.basename(img_path)}', fontsize=10)
        plt.tight_layout()
        
        # Save
        save_dir = defect_dir if label == 1 else good_dir
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(os.path.join(save_dir, f"{base_name}_score_{score:.4f}.png"), dpi=100, bbox_inches='tight')
        plt.close()


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Runs experiments and collects results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single(self, config: ExperimentConfig, model_type: str, 
                   use_augmentation: bool = True, verbose: bool = True) -> dict:
        """Run single experiment"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        paths = get_paths(config.base)
        
        # Create datasets
        aug_type = config.base.augmentation_type if use_augmentation else "none"
        train_ds = WoodDataset(paths["train_good"], None, config.base.image_size, aug_type)
        test_ds = WoodDataset(paths["test_good"], paths["test_defect"], config.base.image_size, "none")
        
        train_loader = DataLoader(train_ds, batch_size=config.base.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config.base.batch_size, shuffle=False, num_workers=0)
        
        # Create and train model
        if model_type == "AutoEncoder":
            model = AutoEncoderModel(config.autoencoder, config.base, device)
        elif model_type == "PatchCore":
            model = PatchCoreModel(config.patchcore, config.base, device)
        elif model_type == "SimpleNet":
            model = SimpleNetModel(config.simplenet, config.base, device)
        elif model_type == "EfficientAD":
            model = EfficientADModel(config.efficientad, config.base, device)
        else:
            raise ValueError(f"Unknown model: {model_type}")
        
        model.fit(train_loader, verbose=verbose)
        scores, maps, labels, img_paths = model.predict(test_loader)
        
        threshold = find_optimal_threshold(labels, scores)
        # Pass anomaly_maps to calculate_metrics for AUPRO calculation
        metrics = calculate_metrics(labels, scores, threshold, anomaly_maps=maps)
        
        result = {
            'experiment_name': config.experiment_name,
            'model': model_type,
            'augmentation': use_augmentation,
            **metrics,
            'config': config.to_dict(),
            'scores': scores,
            'maps': maps,
            'labels': labels,
            'paths': img_paths
        }
        
        self.results.append(result)
        return result
    
    def run_grid_search(self, model_type: str, param_grid: Dict[str, List[Any]], 
                        base_config: ExperimentConfig, use_augmentation: bool = True,
                        verbose: bool = True) -> List[dict]:
        """Run grid search over parameters"""
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"\n🔬 Grid Search: {model_type}")
        print(f"   Parameters: {param_names}")
        print(f"   Combinations: {len(combinations)}")
        
        results = []
        for i, combo in enumerate(combinations):
            config = ExperimentConfig.from_dict(base_config.to_dict())
            config.experiment_name = f"{model_type}_exp_{i+1}"
            
            # Update config with grid parameters
            for name, value in zip(param_names, combo):
                if model_type == "AutoEncoder":
                    setattr(config.autoencoder, name, value)
                elif model_type == "PatchCore":
                    setattr(config.patchcore, name, value)
                elif model_type == "SimpleNet":
                    setattr(config.simplenet, name, value)
                elif model_type == "EfficientAD":
                    setattr(config.efficientad, name, value)
            
            print(f"\n[{i+1}/{len(combinations)}] {dict(zip(param_names, combo))}")
            result = self.run_single(config, model_type, use_augmentation, verbose=False)
            results.append(result)
            print(f"   AUC: {result['auc_roc']:.4f} | AUPRO: {result['aupro']:.4f} | F1: {result['f1_score']:.4f}")
        
        return results
    
    def get_results_table(self) -> pd.DataFrame:
        """Get results as DataFrame with all parameters"""
        rows = []
        for r in self.results:
            config = r['config']
            base_cfg = config['base']
            
            row = {
                'experiment': r['experiment_name'],
                'model': r['model'],
                'dataset': base_cfg['dataset_name'],
                'augmentation': r['augmentation'],
                'image_size': f"{base_cfg['image_size'][0]}x{base_cfg['image_size'][1]}",
                'AUC_ROC': r['auc_roc'],
                'AUPRO': r.get('aupro', 0.0),
                'F1_Score': r['f1_score'],
                'Accuracy': r['accuracy'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'Threshold': r['threshold']
            }
            
            # Add model-specific parameters
            if r['model'] == 'AutoEncoder':
                ae_cfg = config['autoencoder']
                row['latent_dim'] = ae_cfg['latent_dim']
                row['ssim_weight'] = ae_cfg['ssim_weight']
                row['epochs'] = ae_cfg['epochs']
                row['learning_rate'] = ae_cfg['learning_rate']
            elif r['model'] == 'PatchCore':
                pc_cfg = config['patchcore']
                row['memory_bank'] = pc_cfg['memory_bank_size']
                row['k_neighbors'] = pc_cfg['k_neighbors']
                row['multi_layer'] = pc_cfg['use_multi_layer']
            elif r['model'] == 'SimpleNet':
                sn_cfg = config['simplenet']
                row['noise_std'] = sn_cfg['noise_std']
                row['adaptor_dim'] = sn_cfg['adaptor_dim']
                row['epochs'] = sn_cfg['epochs']
                row['multi_noise'] = sn_cfg['use_multi_noise']
            elif r['model'] == 'EfficientAD':
                ead_cfg = config['efficientad']
                row['backbone'] = ead_cfg['backbone']
                row['student_dim'] = ead_cfg['student_dim']
                row['autoencoder_dim'] = ead_cfg['autoencoder_dim']
                row['epochs'] = ead_cfg['epochs']
            
            rows.append(row)
        return pd.DataFrame(rows)
    
    def save_results(self, filename: str = "experiment_results.csv"):
        """Save results to CSV"""
        df = self.get_results_table()
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        print(f"✓ Results saved: {filename}")
        return df
    
    def get_best_result(self, metric: str = 'auc_roc') -> dict:
        """Get best result by metric"""
        return max(self.results, key=lambda x: x[metric])
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all experiments"""
        n_results = len(self.results)
        if n_results == 0:
            return
        
        # Group by model type
        models = {}
        for r in self.results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        # Plot best result per model
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(models.items()):
            best = max(results, key=lambda x: x['auc_roc'])
            y_true = best['labels']
            y_pred = (best['scores'] >= best['threshold']).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Good', 'Defect'], yticklabels=['Good', 'Defect'])
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
            axes[idx].set_title(f'{model_name}\nAUC: {best["auc_roc"]:.3f} | F1: {best["f1_score"]:.3f}')
        
        plt.suptitle('Confusion Matrices (Best per Model)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrices saved")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all experiments"""
        if len(self.results) == 0:
            return
        
        # Group by model
        models = {}
        for r in self.results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        plt.figure(figsize=(10, 8))
        colors = {'AutoEncoder': 'blue', 'PatchCore': 'green', 'SimpleNet': 'red', 'EfficientAD': 'purple'}
        
        for model_name, results in models.items():
            best = max(results, key=lambda x: x['auc_roc'])
            fpr, tpr, _ = roc_curve(best['labels'], best['scores'])
            plt.plot(fpr, tpr, color=colors.get(model_name, 'gray'), linewidth=2,
                    label=f'{model_name} (AUC={best["auc_roc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (Best per Model)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ ROC curves saved")
    
    def plot_score_distributions(self):
        """Plot score distributions for best models"""
        if len(self.results) == 0:
            return
        
        models = {}
        for r in self.results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(models.items()):
            best = max(results, key=lambda x: x['auc_roc'])
            scores = best['scores']
            labels = best['labels']
            threshold = best['threshold']
            
            good_scores = scores[labels == 0]
            defect_scores = scores[labels == 1]
            
            axes[idx].hist(good_scores, bins=15, alpha=0.6, label='Good', color='green')
            axes[idx].hist(defect_scores, bins=15, alpha=0.6, label='Defect', color='red')
            axes[idx].axvline(threshold, color='black', linestyle='--', linewidth=2,
                             label=f'Threshold={threshold:.3f}')
            axes[idx].set_xlabel('Anomaly Score')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'{model_name}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Score Distributions (Best per Model)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Score distributions saved")
    
    def plot_metrics_comparison(self):
        """Plot metrics comparison bar chart"""
        if len(self.results) == 0:
            return
        
        models = {}
        for r in self.results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        # Get best per model
        best_results = {name: max(results, key=lambda x: x['auc_roc']) 
                       for name, results in models.items()}
        
        metrics = ['auc_roc', 'aupro', 'f1_score', 'accuracy', 'precision', 'recall']
        x = np.arange(len(metrics))
        width = 0.20  # Narrower bars for 4 models
        
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {'AutoEncoder': 'steelblue', 'PatchCore': 'forestgreen', 'SimpleNet': 'indianred', 'EfficientAD': 'purple'}
        
        for i, (model_name, result) in enumerate(best_results.items()):
            values = [result.get(m, 0.0) for m in metrics]
            ax.bar(x + i*width, values, width, label=model_name, color=colors.get(model_name, 'gray'))
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics Comparison (Best per Model)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['AUC-ROC', 'AUPRO', 'F1 Score', 'Accuracy', 'Precision', 'Recall'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (model_name, result) in enumerate(best_results.items()):
            values = [result.get(m, 0.0) for m in metrics]
            for j, v in enumerate(values):
                ax.text(x[j] + i*width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Metrics comparison saved")
    
    def generate_summary_report(self):
        """Generate a detailed summary report"""
        if len(self.results) == 0:
            return
        
        # Find best overall and best per model
        best_overall = self.get_best_result('auc_roc')
        best_config = best_overall['config']
        
        models = {}
        for r in self.results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        best_per_model = {name: max(results, key=lambda x: x['auc_roc']) 
                         for name, results in models.items()}
        
        report = []
        report.append("=" * 80)
        report.append("📊 EXPERIMENT SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal Experiments: {len(self.results)}")
        report.append(f"Output Directory: {self.output_dir}")
        report.append(f"Dataset: {best_config['base']['dataset_name']}")
        report.append(f"Image Size: {best_config['base']['image_size']}")
        
        report.append("\n" + "-" * 40)
        report.append("🏆 BEST OVERALL RESULT")
        report.append("-" * 40)
        report.append(f"  Experiment: {best_overall['experiment_name']}")
        report.append(f"  Model:      {best_overall['model']}")
        report.append(f"  AUC-ROC:    {best_overall['auc_roc']:.4f}")
        report.append(f"  AUPRO:      {best_overall.get('aupro', 0.0):.4f}")
        report.append(f"  F1 Score:   {best_overall['f1_score']:.4f}")
        report.append(f"  Accuracy:   {best_overall['accuracy']:.4f}")
        report.append(f"  Precision:  {best_overall['precision']:.4f}")
        report.append(f"  Recall:     {best_overall['recall']:.4f}")
        report.append(f"  Threshold:  {best_overall['threshold']:.4f}")
        
        # Add best model parameters
        report.append("\n  Parameters:")
        if best_overall['model'] == 'AutoEncoder':
            ae = best_config['autoencoder']
            report.append(f"    latent_dim: {ae['latent_dim']}")
            report.append(f"    ssim_weight: {ae['ssim_weight']}")
            report.append(f"    epochs: {ae['epochs']}")
            report.append(f"    learning_rate: {ae['learning_rate']}")
        elif best_overall['model'] == 'PatchCore':
            pc = best_config['patchcore']
            report.append(f"    memory_bank_size: {pc['memory_bank_size']}")
            report.append(f"    k_neighbors: {pc['k_neighbors']}")
            report.append(f"    use_multi_layer: {pc['use_multi_layer']}")
        elif best_overall['model'] == 'SimpleNet':
            sn = best_config['simplenet']
            report.append(f"    noise_std: {sn['noise_std']}")
            report.append(f"    adaptor_dim: {sn['adaptor_dim']}")
            report.append(f"    epochs: {sn['epochs']}")
            report.append(f"    use_multi_noise: {sn['use_multi_noise']}")
        elif best_overall['model'] == 'EfficientAD':
            ead = best_config['efficientad']
            report.append(f"    backbone: {ead['backbone']}")
            report.append(f"    student_dim: {ead['student_dim']}")
            report.append(f"    autoencoder_dim: {ead['autoencoder_dim']}")
            report.append(f"    epochs: {ead['epochs']}")
        
        report.append("\n" + "-" * 40)
        report.append("📈 BEST PER MODEL")
        report.append("-" * 40)
        
        for model_name, result in best_per_model.items():
            cfg = result['config']
            report.append(f"\n  [{model_name}]")
            report.append(f"    Experiment: {result['experiment_name']}")
            report.append(f"    AUC-ROC:    {result['auc_roc']:.4f}")
            report.append(f"    AUPRO:      {result.get('aupro', 0.0):.4f}")
            report.append(f"    F1 Score:   {result['f1_score']:.4f}")
            report.append(f"    Accuracy:   {result['accuracy']:.4f}")
            report.append(f"    Precision:  {result['precision']:.4f}")
            report.append(f"    Recall:     {result['recall']:.4f}")
        
        report.append("\n" + "-" * 40)
        report.append("📁 GENERATED FILES")
        report.append("-" * 40)
        report.append("  - experiment_results.csv")
        report.append("  - confusion_matrices.png")
        report.append("  - roc_curves.png")
        report.append("  - score_distributions.png")
        report.append("  - metrics_comparison.png")
        report.append("  - heatmaps/ (for best model)")
        report.append("  - summary_report.txt")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        print("✓ Summary report saved")
        
        return report_text
    
    def save_all_visualizations(self):
        """Generate and save all visualizations"""
        print("\n" + "="*80)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*80)
        
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_score_distributions()
        self.plot_metrics_comparison()
        self.generate_summary_report()
        self.save_best_config()
    
    def save_best_config(self, metric: str = 'auc_roc'):
        """Save best model configuration to JSON"""
        if len(self.results) == 0:
            return
        
        best = self.get_best_result(metric)
        config = best['config']
        
        # Add metrics to config
        best_config = {
            'experiment_name': best['experiment_name'],
            'model_type': best['model'],
            'metrics': {
                'auc_roc': best['auc_roc'],
                'f1_score': best['f1_score'],
                'accuracy': best['accuracy'],
                'precision': best['precision'],
                'recall': best['recall'],
                'threshold': best['threshold']
            },
            'config': config
        }
        
        config_path = os.path.join(self.output_dir, 'best_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, default=str)
        print(f"✓ Best config saved: best_config.json")
    
    def save_best_model(self, metric: str = 'auc_roc'):
        """
        Retrain and save the best model to file.
        Note: This retrains the model with best config since we don't store model weights during grid search.
        """
        if len(self.results) == 0:
            return
        
        best = self.get_best_result(metric)
        config = ExperimentConfig.from_dict(best['config'])
        model_type = best['model']
        
        print(f"\n💾 Saving best model: {best['experiment_name']}")
        print(f"   Retraining with best configuration...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        paths = get_paths(config.base)
        
        # Create datasets
        train_ds = WoodDataset(paths["train_good"], None, config.base.image_size, config.base.augmentation_type)
        train_loader = DataLoader(train_ds, batch_size=config.base.batch_size, shuffle=True, num_workers=0)
        
        # Create and train model
        if model_type == "AutoEncoder":
            model = AutoEncoderModel(config.autoencoder, config.base, device)
        elif model_type == "PatchCore":
            model = PatchCoreModel(config.patchcore, config.base, device)
        elif model_type == "SimpleNet":
            model = SimpleNetModel(config.simplenet, config.base, device)
        elif model_type == "EfficientAD":
            model = EfficientADModel(config.efficientad, config.base, device)
        else:
            raise ValueError(f"Unknown model: {model_type}")
        
        model.fit(train_loader, verbose=True)
        
        # Save model
        model_path = os.path.join(self.output_dir, f'best_model_{model_type.lower()}.pkl')
        
        # For AutoEncoder, save the nn.Module state dict
        if model_type == "AutoEncoder":
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'config': best['config'],
                'metrics': {
                    'auc_roc': best['auc_roc'],
                    'aupro': best.get('aupro', 0.0),
                    'f1_score': best['f1_score'],
                    'threshold': best['threshold']
                }
            }, model_path)
        else:
            # For PatchCore, SimpleNet, and EfficientAD, save with pickle
            save_data = {
                'model_type': model_type,
                'config': best['config'],
                'metrics': {
                    'auc_roc': best['auc_roc'],
                    'aupro': best.get('aupro', 0.0),
                    'f1_score': best['f1_score'],
                    'threshold': best['threshold']
                }
            }
            if model_type == "PatchCore":
                save_data['memory_bank'] = model.memory_bank
                save_data['random_projection'] = model.random_projection
            elif model_type == "SimpleNet":
                save_data['adaptor_state'] = model.adaptor.state_dict()
                save_data['disc_state'] = model.disc.state_dict()
            elif model_type == "EfficientAD":
                save_data['student_state'] = model.student.state_dict()
                save_data['autoencoder_state'] = model.autoencoder.state_dict()
                save_data['student_head_state'] = model.student_head.state_dict()
            
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
        
        print(f"✓ Best model saved: {os.path.basename(model_path)}")
    
    def save_best_heatmaps(self, metric: str = 'auc_roc'):
        """Save heatmaps for best model"""
        best = self.get_best_result(metric)
        print(f"\n📸 Generating heatmaps for best model: {best['experiment_name']}")

        save_heatmaps(
            best['model'], best['scores'], best['maps'],
            best['labels'], best['paths'], best['threshold'],
            self.output_dir
        )
        print(f"✓ Heatmaps saved")
    
    def save_all_best_heatmaps(self, metric: str = 'auc_roc'):
        """Save heatmaps for best result of EACH model type"""
        if len(self.results) == 0:
            return
        
        # Group results by model type
        models = {}
        for r in self.results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        print(f"\n📸 Generating heatmaps for best result of each model...")
        print(f"   Models found: {list(models.keys())}")
        
        for model_name, results in models.items():
            # Find best result for this model
            best = max(results, key=lambda x: x[metric])
            print(f"\n   [{model_name}] Best: {best['experiment_name']} (AUC: {best['auc_roc']:.4f})")
            
            save_heatmaps(
                model_name, best['scores'], best['maps'],
                best['labels'], best['paths'], best['threshold'],
                self.output_dir
            )
            print(f"   ✓ {model_name} heatmaps saved")
        
        print(f"\n✅ All model heatmaps saved to: {self.output_dir}/heatmaps/")


# ============================================================================
# Main Functions
# ============================================================================

def run_quick_experiments():
    """Quick experiments with reduced parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"experiment_{timestamp}")
    
    print("\n" + "="*80)
    print("🔬 QUICK EXPERIMENT MODE")
    print("="*80)
    
    # Base config with reduced epochs
    base = ExperimentConfig()
    base.base.environment = "colab"  # Change as needed
    base.autoencoder.epochs = 20
    base.simplenet.epochs = 15
    
    runner = ExperimentRunner(output_dir)
    
    # AutoEncoder grid
    runner.run_grid_search("AutoEncoder", {
        'latent_dim': [128, 256],
        'ssim_weight': [0.3, 0.5, 0.7]
    }, base)
    
    # PatchCore grid (-1 = use all patches)
    runner.run_grid_search("PatchCore", {
        'memory_bank_size': [2000, 5000, -1],
        'k_neighbors': [3, 5, 9]
    }, base)
    
    # SimpleNet grid
    runner.run_grid_search("SimpleNet", {
        'noise_std': [0.01, 0.015, 0.02]
    }, base)
    
    # EfficientAD grid
    base.efficientad.epochs = 25  # Reduced for quick mode
    runner.run_grid_search("EfficientAD", {
        'student_dim': [256, 384],
        'autoencoder_dim': [256, 384]
    }, base)
    
    # Save and display results
    df = runner.save_results()
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    print(df.sort_values('AUC_ROC', ascending=False).to_string(index=False))
    
    # Generate all visualizations
    runner.save_all_visualizations()
    runner.save_all_best_heatmaps()  # Save heatmaps for all 4 models
    
    print(f"\n✅ Experiments complete! Results in: {output_dir}")
    return runner


def run_full_experiments():
    """Full experiments with all parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"experiment_full_{timestamp}")
    
    print("\n" + "="*80)
    print("🔬 FULL EXPERIMENT MODE")
    print("="*80)
    
    base = ExperimentConfig()
    base.base.environment = "colab"
    
    runner = ExperimentRunner(output_dir)
    
    # AutoEncoder
    runner.run_grid_search("AutoEncoder", {
        'latent_dim': [128, 256, 512],
        'ssim_weight': [0.3, 0.5, 0.7],
        'learning_rate': [1e-3, 5e-4]
    }, base)
    
    # PatchCore
    runner.run_grid_search("PatchCore", {
        'memory_bank_size': [1000, 2000, 3000],
        'k_neighbors': [3, 5, 9],
        'use_multi_layer': [True, False]
    }, base)
    
    # SimpleNet
    runner.run_grid_search("SimpleNet", {
        'noise_std': [0.01, 0.015, 0.02],
        'adaptor_dim': [256, 512],
        'use_multi_noise': [True, False]
    }, base)
    
    # EfficientAD
    runner.run_grid_search("EfficientAD", {
        'student_dim': [256, 384, 512],
        'autoencoder_dim': [256, 384, 512],
        'backbone': ['resnet18']
    }, base)
    
    df = runner.save_results()
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY (Top 10)")
    print("="*80)
    print(df.sort_values('AUC_ROC', ascending=False).head(10).to_string(index=False))
    
    # Generate all visualizations
    runner.save_all_visualizations()
    runner.save_all_best_heatmaps()  # Save heatmaps for all 4 models
    
    print(f"\n✅ Full experiments complete! Results in: {output_dir}")
    return runner


def run_single_model(model_type: str):
    """Run experiments for single model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"experiment_{model_type.lower()}_{timestamp}")
    
    print(f"\n🔬 Single Model Experiment: {model_type}")
    
    base = ExperimentConfig()
    base.base.environment = "colab"
    base.autoencoder.epochs = 30
    base.simplenet.epochs = 20
    
    runner = ExperimentRunner(output_dir)
    
    if model_type == "AutoEncoder":
        runner.run_grid_search("AutoEncoder", {
            'latent_dim': [128, 256, 512],
            'ssim_weight': [0.3, 0.5, 0.7]
        }, base)
    elif model_type == "PatchCore":
        runner.run_grid_search("PatchCore", {
            'memory_bank_size': [1000, 2000, 3000],
            'k_neighbors': [3, 5, 9]
        }, base)
    elif model_type == "SimpleNet":
        runner.run_grid_search("SimpleNet", {
            'noise_std': [0.01, 0.015, 0.02],
            'adaptor_dim': [256, 512]
        }, base)
    elif model_type == "EfficientAD":
        base.efficientad.epochs = 30
        runner.run_grid_search("EfficientAD", {
            'student_dim': [256, 384, 512],
            'autoencoder_dim': [256, 384, 512]
        }, base)
    
    df = runner.save_results()
    print("\n📊 Results:")
    print(df.sort_values('AUC_ROC', ascending=False).to_string(index=False))
    
    # Generate all visualizations
    runner.save_all_visualizations()
    runner.save_all_best_heatmaps()  # Save heatmaps for all models
    return runner


def main():
    """Main entry point"""
    import sys
    
    print("\n" + "="*80)
    print("🔬 WOOD ANOMALY DETECTION V3 - EXPERIMENT FRAMEWORK")
    print("="*80)
    
    # Parse arguments
    mode = "quick"  # default
    model = None
    
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
        elif arg.startswith("--model="):
            model = arg.split("=")[1]
        elif arg == "--quick":
            mode = "quick"
        elif arg == "--full":
            mode = "full"
    
    # Run appropriate mode
    if mode == "quick":
        run_quick_experiments()
    elif mode == "full":
        run_full_experiments()
    elif mode == "single" and model:
        run_single_model(model)
    else:
        print("Usage:")
        print("  --mode=quick    Quick experiments (reduced epochs)")
        print("  --mode=full     Full experiments (all parameters)")
        print("  --mode=single --model=AutoEncoder|PatchCore|SimpleNet|EfficientAD")


if __name__ == "__main__":
    main()
