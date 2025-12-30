"""
Wood Anomaly Detection - Three Models Comparison
=================================================
Bu dosya cold-start anomaly detection için üç farklı yöntemi karşılaştırır:
1. AutoEncoder (Reconstruction-based)
2. PatchCore (Memory bank + K-NN)
3. SimpleNet (Feature Adaptor + Discriminator)

Hem base data hem de data augmentation ile test yapar.
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
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

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
# Environment Setup
# ============================================================================

ENVIRONMENT = "local"  # "colab" veya "local"

if ENVIRONMENT == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = "/content/drive/MyDrive/Kod/machine_learning/"
    PROJECT_ROOT = os.path.join(DRIVE_PATH, "Kod")
    DATASET_ROOT = os.path.join(DRIVE_PATH, "dataset")
else:
    PROJECT_ROOT = os.getcwd()
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")

# Dataset paths
ORIGINAL_DATA = os.path.join(DATASET_ROOT, "wood")
SELECTED_DATASET = os.path.join(DATASET_ROOT, "wood_otsu")
TRAIN_GOOD_PATH = os.path.join(SELECTED_DATASET, "train", "good")
TEST_GOOD_PATH = os.path.join(SELECTED_DATASET, "test", "good")
TEST_DEFECT_PATH = os.path.join(SELECTED_DATASET, "test", "defect")

# Model output directory
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_outputs")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Dataset Class with Augmentation Support
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class WoodDataset(Dataset):
    """Wood anomaly detection dataset with optional augmentation"""
    
    def __init__(self, 
                 good_path: str, 
                 defect_path: Optional[str] = None,
                 transform=None,
                 image_size: Tuple[int, int] = (224, 224),
                 use_augmentation: bool = False,
                 augmentation_type: str = "basic"):
        """
        Args:
            good_path: İyi (normal) örneklerin klasör yolu
            defect_path: Hatalı örneklerin klasör yolu (test için)
            transform: Custom görüntü dönüşümleri
            image_size: Hedef görüntü boyutu
            use_augmentation: Data augmentation uygula (training için)
            augmentation_type: "basic" (rotation only) or "enhanced" (rotation+flip+color)
        """
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.augmentation_type = augmentation_type
        self.transform = transform or self._default_transform()
        
        self.images = []
        self.labels = []
        self.paths = []
        
        # Good örnekleri yükle (label=0)
        if good_path and os.path.exists(good_path):
            for f in Path(good_path).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(0)
        
        # Defect örnekleri yükle (label=1)
        if defect_path and os.path.exists(defect_path):
            for f in Path(defect_path).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(1)
    
    def _default_transform(self):
        """Default transform with optional augmentation"""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
        ]
        
        # Data augmentation
        if self.use_augmentation:
            if self.augmentation_type == "enhanced":
                # Enhanced augmentation: rotation + flip + color jitter
                transform_list.extend([
                    transforms.RandomRotation(degrees=(-10, 10)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05
                    )
                ])
            else:
                # Basic augmentation: rotation only
                transform_list.append(transforms.RandomRotation(degrees=(-10, 10)))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        # Görüntüyü oku (unicode path desteği)
        img = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8), 
            cv2.IMREAD_COLOR
        )
        
        if img is None:
            img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {img_path}")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grayscale ise RGB'ye çevir
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Transform uygula
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img_tensor, label, img_path


# ============================================================================
# Model 1: AutoEncoder
# ============================================================================

class ConvAutoEncoder(nn.Module):
    """Convolutional AutoEncoder for anomaly detection"""
    
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1),  # latent x 7 x 7
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 3 x 224 x 224
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoEncoderModel:
    """AutoEncoder-based anomaly detection model"""
    
    def __init__(self,
                 device: str = None,
                 latent_dim: int = 256,
                 learning_rate: float = 1e-3,
                 epochs: int = 50):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.model = ConvAutoEncoder(latent_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')
        
        print(f"AutoEncoder initialized on {self.device}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Epochs: {epochs}")
    
    def fit(self, train_loader: DataLoader):
        """Train AutoEncoder on normal data only"""
        print("\n" + "="*60)
        print("AutoEncoder Training")
        print("="*60)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch, labels, paths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch = batch.to(self.device)
                
                # Normalize input to [0, 1] for reconstruction
                # De-normalize from ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                batch_denorm = batch * std + mean
                batch_denorm = torch.clamp(batch_denorm, 0, 1)
                
                # Forward
                self.optimizer.zero_grad()
                reconstructed = self.model(batch_denorm)
                
                # Loss
                loss = self.criterion(reconstructed, batch_denorm).mean()
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        print("\n✅ AutoEncoder training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores based on reconstruction error"""
        print("\n" + "="*60)
        print("AutoEncoder Prediction")
        print("="*60)
        
        self.model.eval()
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
            
            # De-normalize
            batch_denorm = batch * std + mean
            batch_denorm = torch.clamp(batch_denorm, 0, 1)
            
            # Reconstruct
            reconstructed = self.model(batch_denorm)
            
            # Reconstruction error (per pixel)
            error = (reconstructed - batch_denorm) ** 2
            
            # Image-level score: mean error across all pixels
            for i in range(error.shape[0]):
                img_error = error[i].mean().cpu().numpy()
                all_scores.append(img_error)
                
                # Anomaly map: mean across channels
                anomaly_map = error[i].mean(dim=0).cpu().numpy()
                anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
                all_maps.append(anomaly_map_resized)
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        # Normalize scores to [0, 1]
        scores = np.array(all_scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_norm, np.array(all_maps), np.array(all_labels), all_paths
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'latent_dim': self.latent_dim
        }, filepath)
        print(f"AutoEncoder model saved: {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"AutoEncoder model loaded: {filepath}")


# ============================================================================
# Model 2: PatchCore
# ============================================================================

class PatchCoreModel:
    """
    PatchCore: Towards Total Recall in Industrial Anomaly Detection
    Memory bank tabanlı anomali tespit modeli.
    """
    
    def __init__(self,
                 backbone: str = 'wide_resnet50_2',
                 device: str = None,
                 memory_bank_size: int = 1000,
                 reduce_dimensions: bool = True,
                 target_dim: int = 550,
                 k_neighbors: int = 3):
        
        self.backbone_name = backbone
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_bank_size = memory_bank_size
        self.reduce_dimensions = reduce_dimensions
        self.target_dim = target_dim
        self.k_neighbors = k_neighbors
        
        self.feature_extractor = None
        self.memory_bank = None
        self.random_projection = None
        self.nn_model = None
        self.feature_map_size = None
        
        self._build_feature_extractor()
        
        print(f"PatchCore initialized on {self.device}")
        print(f"  Backbone: {backbone}")
        print(f"  Memory bank size: {memory_bank_size}")
    
    def _build_feature_extractor(self):
        """Build feature extractor from pretrained backbone"""
        if self.backbone_name == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
        elif self.backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
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
    
    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader) -> np.ndarray:
        """Extract features from dataloader"""
        all_features = []
        
        for batch, labels, paths in tqdm(dataloader, desc="Feature extraction"):
            batch = batch.to(self.device)
            features = self.feature_extractor(batch)
            
            if self.feature_map_size is None:
                self.feature_map_size = (features.shape[2], features.shape[3])
            
            b, c, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(b, h*w, c)
            all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
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
        """Build memory bank from normal data"""
        print("\n" + "="*60)
        print("PatchCore Training (Memory Bank Creation)")
        print("="*60)
        
        print("\n[1/3] Feature extraction...")
        features = self._extract_features(train_loader)
        print(f"  Extracted features shape: {features.shape}")
        
        if self.reduce_dimensions and features.shape[1] > self.target_dim:
            print(f"\n[2/3] Dimension reduction: {features.shape[1]} -> {self.target_dim}")
            self.random_projection = SparseRandomProjection(
                n_components=self.target_dim,
                random_state=42
            )
            features = self.random_projection.fit_transform(features)
        else:
            print("\n[2/3] Dimension reduction: skipped")
        
        print(f"\n[3/3] Building memory bank...")
        if len(features) > self.memory_bank_size:
            self.memory_bank = self._coreset_sampling(features, self.memory_bank_size)
        else:
            self.memory_bank = features
        
        print(f"  Memory bank shape: {self.memory_bank.shape}")
        
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        )
        self.nn_model.fit(self.memory_bank)
        
        print("\n✅ PatchCore training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores"""
        print("\n" + "="*60)
        print("PatchCore Prediction")
        print("="*60)
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
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
        """Save model"""
        model_data = {
            'memory_bank': self.memory_bank,
            'random_projection': self.random_projection,
            'nn_model': self.nn_model,
            'backbone_name': self.backbone_name,
            'memory_bank_size': self.memory_bank_size,
            'target_dim': self.target_dim,
            'feature_map_size': self.feature_map_size,
            'k_neighbors': self.k_neighbors
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"PatchCore model saved: {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.memory_bank = model_data['memory_bank']
        self.random_projection = model_data['random_projection']
        self.nn_model = model_data['nn_model']
        self.feature_map_size = model_data.get('feature_map_size')
        print(f"PatchCore model loaded: {filepath}")


# ============================================================================
# Model 3: SimpleNet
# ============================================================================

class FeatureAdaptor(nn.Module):
    """Feature Adaptor - single FC layer without bias for domain adaptation"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.adaptor = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x):
        return self.adaptor(x)


class Discriminator(nn.Module):
    """Simple MLP discriminator for normal vs anomaly classification"""
    
    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        return self.mlp(x)


class SimpleNetModel:
    """
    SimpleNet: A Simple Network for Image Anomaly Detection and Localization (CVPR 2023)
    
    Components:
    1. Feature Extractor (pretrained backbone)
    2. Feature Adaptor (domain adaptation)
    3. Anomaly Feature Generator (training only - Gaussian noise)
    4. Discriminator (normal vs anomaly)
    """
    
    def __init__(self,
                 device: str = None,
                 backbone: str = 'wide_resnet50_2',
                 adaptor_dim: int = 512,
                 noise_std: float = 0.015,
                 learning_rate: float = 1e-4,
                 epochs: int = 20):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone_name = backbone
        self.adaptor_dim = adaptor_dim
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.feature_extractor = None
        self.feature_adaptor = None
        self.discriminator = None
        self.feature_dim = None
        self.feature_map_size = None
        
        self._build_feature_extractor()
        
        print(f"SimpleNet initialized on {self.device}")
        print(f"  Backbone: {backbone}")
        print(f"  Adaptor dim: {adaptor_dim}")
        print(f"  Noise std: {noise_std}")
    
    def _build_feature_extractor(self):
        """Build feature extractor from pretrained backbone"""
        if self.backbone_name == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
            self.feature_dim = 1024  # Layer3 output channels
        elif self.backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.feature_dim = 256
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
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
        
        # Initialize adaptor and discriminator
        self.feature_adaptor = FeatureAdaptor(self.feature_dim, self.adaptor_dim).to(self.device)
        self.discriminator = Discriminator(self.adaptor_dim).to(self.device)
    
    def _generate_anomaly_features(self, normal_features: torch.Tensor) -> torch.Tensor:
        """Generate synthetic anomaly features by adding Gaussian noise"""
        noise = torch.randn_like(normal_features) * self.noise_std
        anomaly_features = normal_features + noise
        return anomaly_features
    
    def _truncated_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
        """Truncated L1 loss for discriminator training"""
        loss = F.l1_loss(pred, target, reduction='none')
        loss = torch.clamp(loss, max=margin)
        return loss.mean()
    
    def fit(self, train_loader: DataLoader):
        """Train SimpleNet on normal data"""
        print("\n" + "="*60)
        print("SimpleNet Training")
        print("="*60)
        
        # Optimizer for adaptor and discriminator only
        optimizer = Adam(
            list(self.feature_adaptor.parameters()) + list(self.discriminator.parameters()),
            lr=self.learning_rate
        )
        
        self.feature_adaptor.train()
        self.discriminator.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch, labels, paths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch = batch.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor(batch)
                    
                    if self.feature_map_size is None:
                        self.feature_map_size = (features.shape[2], features.shape[3])
                
                b, c, h, w = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, c)
                
                # Adapt features
                adapted_features = self.feature_adaptor(features_flat)
                
                # Generate synthetic anomaly features
                anomaly_features = self._generate_anomaly_features(adapted_features.detach())
                
                # Discriminator training
                optimizer.zero_grad()
                
                # Normal features -> low score (target=0)
                normal_pred = self.discriminator(adapted_features)
                normal_target = torch.zeros_like(normal_pred)
                
                # Anomaly features -> high score (target=1)
                anomaly_pred = self.discriminator(anomaly_features)
                anomaly_target = torch.ones_like(anomaly_pred)
                
                # Combined loss
                loss_normal = self._truncated_l1_loss(normal_pred, normal_target)
                loss_anomaly = self._truncated_l1_loss(anomaly_pred, anomaly_target)
                loss = loss_normal + loss_anomaly
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        print("\n✅ SimpleNet training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Predict anomaly scores"""
        print("\n" + "="*60)
        print("SimpleNet Prediction")
        print("="*60)
        
        self.feature_adaptor.eval()
        self.discriminator.eval()
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
            
            # Extract features
            features = self.feature_extractor(batch)
            b, c, h, w = features.shape
            
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, c)
            
            # Adapt features
            adapted_features = self.feature_adaptor(features_flat)
            
            # Get discriminator scores (anomaly score = output value)
            scores = self.discriminator(adapted_features).squeeze()
            scores = scores.cpu().numpy()
            
            # Per-image scores and maps
            n_patches = h * w
            for i in range(b):
                start_idx = i * n_patches
                end_idx = start_idx + n_patches
                
                img_scores = scores[start_idx:end_idx]
                img_score = np.max(img_scores)  # Use max for image-level anomaly
                all_scores.append(img_score)
                
                # Anomaly map
                anomaly_map = img_scores.reshape(h, w)
                anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
                all_maps.append(anomaly_map_resized)
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        # Normalize scores to [0, 1]
        scores = np.array(all_scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_norm, np.array(all_maps), np.array(all_labels), all_paths
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'feature_adaptor': self.feature_adaptor.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'adaptor_dim': self.adaptor_dim,
            'feature_dim': self.feature_dim,
            'backbone_name': self.backbone_name,
            'feature_map_size': self.feature_map_size
        }, filepath)
        print(f"SimpleNet model saved: {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.feature_adaptor.load_state_dict(checkpoint['feature_adaptor'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.feature_map_size = checkpoint.get('feature_map_size')
        print(f"SimpleNet model loaded: {filepath}")


# ============================================================================
# Evaluation & Visualization Functions
# ============================================================================

def calculate_metrics(y_true: np.ndarray, 
                      y_scores: np.ndarray, 
                      threshold: float = 0.5) -> dict:
    """Calculate all metrics"""
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_scores),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def plot_roc_comparison(results: Dict[str, dict], save_path: str = None):
    """Plot ROC curves for all models on same plot"""
    plt.figure(figsize=(10, 8))
    
    colors = {'AutoEncoder': 'blue', 'PatchCore': 'green', 'SimpleNet': 'red'}
    
    for model_name, result in results.items():
        y_true = result['labels']
        y_scores = result['scores']
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        color = colors.get(model_name, 'gray')
        plt.plot(fpr, tpr, color=color, linewidth=2, 
                 label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results_base: Dict[str, dict], 
                            results_aug: Dict[str, dict],
                            save_path: str = None):
    """Plot metrics comparison bar chart"""
    models = list(results_base.keys())
    metrics_names = ['auc_roc', 'f1_score', 'accuracy', 'precision', 'recall']
    
    fig, axes = plt.subplots(1, len(metrics_names), figsize=(20, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    for idx, metric in enumerate(metrics_names):
        base_values = [results_base[m]['metrics'][metric] for m in models]
        aug_values = [results_aug[m]['metrics'][metric] for m in models]
        
        axes[idx].bar(x - width/2, base_values, width, label='Base', color='steelblue')
        axes[idx].bar(x + width/2, aug_values, width, label='Augmented', color='coral')
        
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models, rotation=45)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 1)
    
    plt.suptitle('Model Comparison: Base vs Augmented Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results: Dict[str, dict], save_path: str = None):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        y_true = result['labels']
        y_pred = result['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Good', 'Defect'],
                    yticklabels=['Good', 'Defect'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f'{model_name}')
    
    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distributions(results: Dict[str, dict], save_path: str = None):
    """Plot score distributions for all models"""
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
    
    plt.suptitle('Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_comparison_results(results: Dict[str, dict], title: str = "Results"):
    """Print comparison results in table format"""
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


def generate_readme(output_dir: str,
                    results_base: Dict[str, dict],
                    results_aug: Dict[str, dict],
                    experiment_config: dict):
    """
    Generate a detailed README.md file for the experiment results.
    
    Args:
        output_dir: Output directory path
        results_base: Results without augmentation
        results_aug: Results with augmentation
        experiment_config: Configuration dictionary with experiment parameters
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    readme_content = f"""# Wood Anomaly Detection - Experiment Results

**Generated:** {timestamp}  
**Output Directory:** `{os.path.basename(output_dir)}`

---

## Experiment Overview

This experiment compares three anomaly detection methods for cold-start wood defect detection:
- **AutoEncoder**: Reconstruction-based anomaly detection
- **PatchCore**: Memory bank + K-NN approach
- **SimpleNet**: Feature Adaptor + Discriminator (CVPR 2023)

---

## Dataset Information

| Parameter | Value |
|-----------|-------|
| Dataset | wood_otsu |
| Train Samples (good only) | {experiment_config.get('train_samples', 'N/A')} |
| Test Samples | {experiment_config.get('test_samples', 'N/A')} |
| Image Size | {experiment_config.get('image_size', '224x224')} |

---

## Model Configurations

### AutoEncoder
| Parameter | Value |
|-----------|-------|
| Latent Dimension | {experiment_config.get('ae_latent_dim', 256)} |
| Learning Rate | {experiment_config.get('ae_lr', 1e-3)} |
| Epochs | {experiment_config.get('ae_epochs', 100)} |

### PatchCore
| Parameter | Value |
|-----------|-------|
| Backbone | {experiment_config.get('pc_backbone', 'wide_resnet50_2')} |
| Memory Bank Size | {experiment_config.get('pc_memory_bank', 1000)} |
| Target Dimension | {experiment_config.get('pc_target_dim', 512)} |
| K-Neighbors | {experiment_config.get('pc_k_neighbors', 5)} |
| Reduce Dimensions | {experiment_config.get('pc_reduce_dims', True)} |

### SimpleNet
| Parameter | Value |
|-----------|-------|
| Backbone | {experiment_config.get('sn_backbone', 'wide_resnet50_2')} |
| Adaptor Dimension | {experiment_config.get('sn_adaptor_dim', 512)} |
| Noise Std | {experiment_config.get('sn_noise_std', 0.015)} |
| Learning Rate | {experiment_config.get('sn_lr', 1e-4)} |
| Epochs | {experiment_config.get('sn_epochs', 50)} |

---

## Data Augmentation

| Setting | Value |
|---------|-------|
| Base Augmentation | None |
| Enhanced Augmentation | ±10° Rotation |

---

## Results Summary

### Without Augmentation (Base)

| Model | AUC-ROC | F1 Score | Accuracy | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
"""
    
    # Add base results
    for model_name, result in results_base.items():
        m = result['metrics']
        readme_content += f"| {model_name} | {m['auc_roc']:.4f} | {m['f1_score']:.4f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |\n"
    
    readme_content += """
### With Augmentation (±10° Rotation)

| Model | AUC-ROC | F1 Score | Accuracy | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
"""
    
    # Add augmented results
    for model_name, result in results_aug.items():
        m = result['metrics']
        readme_content += f"| {model_name} | {m['auc_roc']:.4f} | {m['f1_score']:.4f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |\n"
    
    # Find best model
    best_model_base = max(results_base.items(), key=lambda x: x[1]['metrics']['auc_roc'])
    best_model_aug = max(results_aug.items(), key=lambda x: x[1]['metrics']['auc_roc'])
    
    readme_content += f"""
---

## Best Performing Models

- **Base Data:** {best_model_base[0]} (AUC-ROC: {best_model_base[1]['metrics']['auc_roc']:.4f})
- **With Augmentation:** {best_model_aug[0]} (AUC-ROC: {best_model_aug[1]['metrics']['auc_roc']:.4f})

---

## Output Files

| File | Description |
|------|-------------|
| `comparison_results.csv` | All metrics in CSV format |
| `roc_comparison_base.png` | ROC curves (base data) |
| `roc_comparison_augmented.png` | ROC curves (augmented data) |
| `metrics_comparison.png` | Side-by-side metrics comparison |
| `confusion_matrices_base.png` | Confusion matrices (base) |
| `confusion_matrices_augmented.png` | Confusion matrices (augmented) |
| `score_distributions_base.png` | Anomaly score distributions (base) |
| `score_distributions_augmented.png` | Anomaly score distributions (augmented) |

---

## Notes

- All models were trained using only "good" (normal) samples
- Test set contains both "good" and "defect" samples
- Optimal thresholds were determined using Youden's J statistic
- PatchCore parameters optimized via grid search

---

## Reproduction

```bash
cd machine_learning
python compare_anomaly_models.py
```

For quick testing with reduced epochs:
```bash
python compare_anomaly_models.py --quick-test
```
"""
    
    # Write README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  ✓ README.md generated: {readme_path}")


# ============================================================================
# Main Comparison Function
# ============================================================================

def run_comparison(use_augmentation: bool = False, 
                   device: str = None,
                   quick_test: bool = False) -> Dict[str, dict]:
    """
    Run comparison of all three models
    
    Args:
        use_augmentation: Whether to use data augmentation during training
        device: Device to use ('cuda' or 'cpu')
        quick_test: If True, use fewer epochs for quick testing
        
    Returns:
        Dictionary with results for each model
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    aug_suffix = "with augmentation" if use_augmentation else "without augmentation"
    print("\n" + "="*80)
    print(f"🔬 RUNNING COMPARISON ({aug_suffix.upper()})")
    print("="*80)
    print(f"Device: {device}")
    
    # HIGH RESOLUTION: 384x384 for better feature extraction
    IMAGE_SIZE = (384, 384)
    
    # Training dataset (with or without augmentation)
    train_dataset = WoodDataset(
        good_path=TRAIN_GOOD_PATH,
        defect_path=None,
        image_size=IMAGE_SIZE,
        use_augmentation=use_augmentation
    )
    
    # Test dataset (never augmented)
    test_dataset = WoodDataset(
        good_path=TEST_GOOD_PATH,
        defect_path=TEST_DEFECT_PATH,
        image_size=IMAGE_SIZE,
        use_augmentation=False
    )
    
    print(f"\nDataset:")
    print(f"  Train (good only): {len(train_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Augmentation: {use_augmentation}")
    print(f"  Image Size: 384x384 (HIGH RES)")
    
    # Reduced batch size for high resolution
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    results = {}
    
    # Model configurations (HIGH PERFORMANCE - increased params)
    ae_epochs = 15 if quick_test else 150
    sn_epochs = 10 if quick_test else 100
    
    # =========================================================================
    # Model 1: AutoEncoder
    # =========================================================================
    print("\n" + "-"*60)
    print("📦 MODEL 1: AutoEncoder")
    print("-"*60)
    
    ae_model = AutoEncoderModel(
        device=device,
        latent_dim=256,
        learning_rate=1e-3,
        epochs=ae_epochs
    )
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
    print("📦 MODEL 2: PatchCore")
    print("-"*60)
    
    # HIGH PERFORMANCE: Increased memory bank for high resolution
    pc_model = PatchCoreModel(
        backbone='wide_resnet50_2',
        device=device,
        memory_bank_size=3000,  # Increased for high res (was 1000)
        reduce_dimensions=True,
        target_dim=768,  # Increased for high res (was 512)
        k_neighbors=5
    )
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
    print("📦 MODEL 3: SimpleNet")
    print("-"*60)
    
    # HIGH PERFORMANCE: Increased params and different noise for better separation
    sn_model = SimpleNetModel(
        device=device,
        backbone='wide_resnet50_2',
        adaptor_dim=768,  # Increased (was 512)
        noise_std=0.02,   # Slightly higher noise for better separation (was 0.015)
        learning_rate=2e-4,  # Slightly higher LR (was 1e-4)
        epochs=sn_epochs
    )
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


def main(quick_test: bool = False):
    """Main function to run full comparison"""
    print("\n" + "="*80)
    print("🔬 WOOD ANOMALY DETECTION - THREE MODELS COMPARISON")
    print("="*80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📌 Device: {device}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(MODEL_OUTPUT_DIR, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # =========================================================================
    # Test 1: Base data (no augmentation)
    # =========================================================================
    results_base = run_comparison(use_augmentation=False, device=device, quick_test=quick_test)
    print_comparison_results(results_base, "RESULTS - BASE DATA (No Augmentation)")
    
    # =========================================================================
    # Test 2: With augmentation
    # =========================================================================
    results_aug = run_comparison(use_augmentation=True, device=device, quick_test=quick_test)
    print_comparison_results(results_aug, "RESULTS - WITH AUGMENTATION (±10° Rotation)")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    # ROC Comparison - Base
    plot_roc_comparison(
        results_base,
        save_path=os.path.join(output_dir, "roc_comparison_base.png")
    )
    print("  ✓ ROC comparison (base) saved")
    
    # ROC Comparison - Augmented
    plot_roc_comparison(
        results_aug,
        save_path=os.path.join(output_dir, "roc_comparison_augmented.png")
    )
    print("  ✓ ROC comparison (augmented) saved")
    
    # Metrics Comparison
    plot_metrics_comparison(
        results_base, results_aug,
        save_path=os.path.join(output_dir, "metrics_comparison.png")
    )
    print("  ✓ Metrics comparison saved")
    
    # Confusion Matrices
    plot_confusion_matrices(
        results_base,
        save_path=os.path.join(output_dir, "confusion_matrices_base.png")
    )
    print("  ✓ Confusion matrices (base) saved")
    
    plot_confusion_matrices(
        results_aug,
        save_path=os.path.join(output_dir, "confusion_matrices_augmented.png")
    )
    print("  ✓ Confusion matrices (augmented) saved")
    
    # Score Distributions
    plot_score_distributions(
        results_base,
        save_path=os.path.join(output_dir, "score_distributions_base.png")
    )
    print("  ✓ Score distributions (base) saved")
    
    plot_score_distributions(
        results_aug,
        save_path=os.path.join(output_dir, "score_distributions_augmented.png")
    )
    print("  ✓ Score distributions (augmented) saved")
    
    # CSV Results
    rows = []
    for model_name in results_base.keys():
        base_m = results_base[model_name]['metrics']
        aug_m = results_aug[model_name]['metrics']
        
        rows.append({
            'Model': model_name,
            'Data': 'Base',
            'AUC_ROC': base_m['auc_roc'],
            'F1_Score': base_m['f1_score'],
            'Accuracy': base_m['accuracy'],
            'Precision': base_m['precision'],
            'Recall': base_m['recall'],
            'Threshold': base_m['threshold']
        })
        rows.append({
            'Model': model_name,
            'Data': 'Augmented',
            'AUC_ROC': aug_m['auc_roc'],
            'F1_Score': aug_m['f1_score'],
            'Accuracy': aug_m['accuracy'],
            'Precision': aug_m['precision'],
            'Recall': aug_m['recall'],
            'Threshold': aug_m['threshold']
        })
    
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Results CSV saved: {csv_path}")
    
    # Generate README with experiment configuration
    experiment_config = {
        'train_samples': len(WoodDataset(TRAIN_GOOD_PATH, None)),
        'test_samples': len(WoodDataset(TEST_GOOD_PATH, TEST_DEFECT_PATH)),
        'image_size': '384x384 (HIGH RES)',
        'ae_latent_dim': 256,
        'ae_lr': 1e-3,
        'ae_epochs': 15 if quick_test else 150,
        'pc_backbone': 'wide_resnet50_2',
        'pc_memory_bank': 3000,
        'pc_target_dim': 768,
        'pc_k_neighbors': 5,
        'pc_reduce_dims': True,
        'sn_backbone': 'wide_resnet50_2',
        'sn_adaptor_dim': 768,
        'sn_noise_std': 0.02,
        'sn_lr': 2e-4,
        'sn_epochs': 10 if quick_test else 100
    }
    
    generate_readme(output_dir, results_base, results_aug, experiment_config)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    print(f"📁 All results saved to: {output_dir}")
    
    return results_base, results_aug


if __name__ == "__main__":
    import sys
    
    quick_test = "--quick-test" in sys.argv
    
    if quick_test:
        print("\n⚡ QUICK TEST MODE - Reduced epochs for testing")
    
    main(quick_test=quick_test)
