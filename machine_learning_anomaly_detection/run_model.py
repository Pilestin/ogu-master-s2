"""
PatchCore Anomaly Detection for Wood Defect Detection
======================================================
Bu dosya sadece "good" örneklerden öğrenerek anomali tespiti yapar.
PatchCore modeli kullanılarak ahşap üzerindeki hataları tespit eder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Tuple, List, Optional

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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
# Dataset Class
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class WoodDataset(Dataset):
    """Wood anomaly detection dataset"""
    
    def __init__(self, 
                 good_path: str, 
                 defect_path: Optional[str] = None,
                 transform=None,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            good_path: İyi (normal) örneklerin klasör yolu
            defect_path: Hatalı örneklerin klasör yolu (test için)
            transform: Görüntü dönüşümleri
            image_size: Hedef görüntü boyutu
        """
        self.image_size = image_size
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
        """Default ImageNet normalization"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
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
            # Fallback: standard okuma
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
# PatchCore Model
# ============================================================================

class PatchCoreModel:
    """
    PatchCore: Towards Total Recall in Industrial Anomaly Detection
    
    Memory bank tabanlı anomali tespit modeli.
    Sadece normal örneklerden öğrenir ve test zamanında
    K-NN ile anomali skoru hesaplar.
    """
    
    def __init__(self,
                 backbone: str = 'wide_resnet50_2',
                 device: str = None,
                 memory_bank_size: int = 1000,
                 reduce_dimensions: bool = True,
                 target_dim: int = 550,
                 k_neighbors: int = 3):
        """
        Args:
            backbone: Feature extractor backbone ('wide_resnet50_2' veya 'resnet18')
            device: 'cuda' veya 'cpu'
            memory_bank_size: Coreset boyutu
            reduce_dimensions: Random projection ile boyut azaltma
            target_dim: Hedef feature boyutu
            k_neighbors: K-NN için komşu sayısı
        """
        self.backbone_name = backbone
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_bank_size = memory_bank_size
        self.reduce_dimensions = reduce_dimensions
        self.target_dim = target_dim
        self.k_neighbors = k_neighbors
        
        # Model components
        self.feature_extractor = None
        self.memory_bank = None
        self.random_projection = None
        self.nn_model = None
        
        # Feature map boyutu (predict'te kullanılacak)
        self.feature_map_size = None
        
        self._build_feature_extractor()
        
        print(f"PatchCore initialized on {self.device}")
        print(f"  Backbone: {backbone}")
        print(f"  Memory bank size: {memory_bank_size}")
    
    def _build_feature_extractor(self):
        """Pretrained backbone'dan feature extractor oluştur"""
        if self.backbone_name == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1')
        elif self.backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # Layer2 ve Layer3'ü kullan (multi-scale features)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        ).to(self.device)
        
        # Eval mode, gradyan hesaplama yok
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader) -> np.ndarray:
        """DataLoader'dan feature çıkar"""
        all_features = []
        
        for batch, labels, paths in tqdm(dataloader, desc="Feature extraction"):
            batch = batch.to(self.device)
            
            # Forward pass
            features = self.feature_extractor(batch)
            
            # [B, C, H, W] -> feature map boyutunu kaydet
            if self.feature_map_size is None:
                self.feature_map_size = (features.shape[2], features.shape[3])
            
            # [B, C, H, W] -> [B, H*W, C]
            b, c, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(b, h*w, c)
            
            all_features.append(features.cpu().numpy())
        
        # [N_images, H*W, C] -> [N_images * H*W, C]
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(-1, all_features.shape[-1])
        
        return all_features
    
    def _coreset_sampling(self, features: np.ndarray, k: int) -> np.ndarray:
        """
        Greedy coreset sampling ile representative patches seç.
        Büyük veri setlerinden verimli örnekleme yapar.
        """
        if len(features) <= k:
            return features
        
        print(f"  Coreset sampling: {len(features)} -> {k}")
        
        # İlk nokta rastgele
        selected_indices = [np.random.randint(len(features))]
        min_distances = np.full(len(features), np.inf)
        
        for i in tqdm(range(1, k), desc="  Coreset sampling"):
            # Son eklenen noktaya olan mesafeleri hesapla
            last_selected = features[selected_indices[-1]]
            distances = np.linalg.norm(features - last_selected, axis=1)
            
            # Minimum mesafeleri güncelle
            min_distances = np.minimum(min_distances, distances)
            
            # Zaten seçilenleri maskele
            min_distances[selected_indices] = -1
            
            # En uzak noktayı seç
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return features[selected_indices]
    
    def fit(self, train_loader: DataLoader):
        """
        Normal verilerle memory bank oluştur.
        
        Args:
            train_loader: Sadece normal (good) örnekleri içeren DataLoader
        """
        print("\n" + "="*60)
        print("PatchCore Training (Memory Bank Creation)")
        print("="*60)
        
        # Feature extraction
        print("\n[1/3] Feature extraction...")
        features = self._extract_features(train_loader)
        print(f"  Extracted features shape: {features.shape}")
        
        # Dimension reduction
        if self.reduce_dimensions and features.shape[1] > self.target_dim:
            print(f"\n[2/3] Dimension reduction: {features.shape[1]} -> {self.target_dim}")
            self.random_projection = SparseRandomProjection(
                n_components=self.target_dim,
                random_state=42
            )
            features = self.random_projection.fit_transform(features)
        else:
            print("\n[2/3] Dimension reduction: skipped")
        
        # Coreset sampling
        print(f"\n[3/3] Building memory bank...")
        if len(features) > self.memory_bank_size:
            self.memory_bank = self._coreset_sampling(features, self.memory_bank_size)
        else:
            self.memory_bank = features
        
        print(f"  Memory bank shape: {self.memory_bank.shape}")
        
        # K-NN model
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        )
        self.nn_model.fit(self.memory_bank)
        
        print("\n✅ Training completed!")
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Test verisi için anomaly score ve heatmap hesapla.
        
        Returns:
            anomaly_scores: Her görüntü için anomaly skoru
            anomaly_maps: Pixel-wise anomaly haritaları
            labels: Gerçek etiketler
        """
        print("\n" + "="*60)
        print("PatchCore Prediction")
        print("="*60)
        
        all_scores = []
        all_maps = []
        all_labels = []
        all_paths = []
        
        for batch, labels, paths in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(self.device)
            
            # Feature extraction
            features = self.feature_extractor(batch)
            b, c, h, w = features.shape
            
            # [B, C, H, W] -> [B*H*W, C]
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
            
            # Dimension reduction
            if self.random_projection is not None:
                features_flat = self.random_projection.transform(features_flat)
            
            # K-NN mesafeleri
            distances, _ = self.nn_model.kneighbors(features_flat)
            distances = distances.mean(axis=1)  # K komşunun ortalaması
            
            # Her görüntü için skorları ayır
            n_patches = h * w
            for i in range(b):
                start_idx = i * n_patches
                end_idx = start_idx + n_patches
                
                img_distances = distances[start_idx:end_idx]
                
                # Image-level score (max anomaly)
                img_score = np.max(img_distances)
                all_scores.append(img_score)
                
                # Anomaly map
                anomaly_map = img_distances.reshape(h, w)
                anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
                all_maps.append(anomaly_map_resized)
            
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
        
        # Skorları normalize et [0, 1]
        scores = np.array(all_scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_norm, np.array(all_maps), np.array(all_labels), all_paths
    
    def save(self, filepath: str):
        """Modeli kaydet"""
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
        print(f"Model saved: {filepath}")
    
    def load(self, filepath: str):
        """Modeli yükle"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.memory_bank = model_data['memory_bank']
        self.random_projection = model_data['random_projection']
        self.nn_model = model_data['nn_model']
        self.feature_map_size = model_data.get('feature_map_size')
        print(f"Model loaded: {filepath}")


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_metrics(y_true: np.ndarray, 
                      y_scores: np.ndarray, 
                      threshold: float = 0.5) -> dict:
    """Tüm metrikleri hesapla"""
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
    """Youden's J statistic ile optimal threshold bul"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          save_path: str = None):
    """Confusion matrix görselleştir"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good', 'Defect'],
                yticklabels=['Good', 'Defect'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - PatchCore')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                   y_scores: np.ndarray, 
                   save_path: str = None):
    """ROC eğrisi çiz"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'PatchCore (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_score_distribution(y_true: np.ndarray, 
                            y_scores: np.ndarray,
                            threshold: float = 0.5,
                            save_path: str = None):
    """Anomaly score dağılımını göster"""
    good_scores = y_scores[y_true == 0]
    defect_scores = y_scores[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(good_scores, bins=20, alpha=0.6, label='Good', color='green')
    plt.hist(defect_scores, bins=20, alpha=0.6, label='Defect', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_anomaly_heatmaps(images_paths: List[str],
                                anomaly_maps: np.ndarray,
                                labels: np.ndarray,
                                scores: np.ndarray,
                                n_samples: int = 6,
                                save_path: str = None):
    """Anomaly heatmap'leri görselleştir (seçili örnekler)"""
    # Her sınıftan örnekler seç
    good_indices = np.where(labels == 0)[0]
    defect_indices = np.where(labels == 1)[0]
    
    n_good = min(n_samples // 2, len(good_indices))
    n_defect = min(n_samples - n_good, len(defect_indices))
    
    # En yüksek skorlu defect'leri seç
    defect_scores = scores[defect_indices]
    top_defect_idx = defect_indices[np.argsort(defect_scores)[-n_defect:]]
    
    # En düşük skorlu good'ları seç
    good_scores = scores[good_indices]
    top_good_idx = good_indices[np.argsort(good_scores)[:n_good]]
    
    selected_indices = np.concatenate([top_good_idx, top_defect_idx])
    
    n_cols = 3
    n_rows = len(selected_indices)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        # Dosya adını al
        file_name = Path(images_paths[idx]).stem
        
        # Orijinal görüntüyü oku
        img = cv2.imdecode(
            np.fromfile(images_paths[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        anomaly_map = anomaly_maps[idx]
        label = "Good" if labels[idx] == 0 else "Defect"
        score = scores[idx]
        
        # Normalize heatmap
        heatmap_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap_norm), 
            cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
        # Original - dosya adı ile
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{file_name}\n[{label}] Score: {score:.4f}', fontsize=9)
        axes[i, 0].axis('off')
        
        # Heatmap
        axes[i, 1].imshow(heatmap_norm, cmap='jet')
        axes[i, 1].set_title('Anomaly Heatmap', fontsize=9)
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay', fontsize=9)
        axes[i, 2].axis('off')
    
    plt.suptitle('Anomaly Detection Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_all_heatmaps(images_paths: List[str],
                      anomaly_maps: np.ndarray,
                      labels: np.ndarray,
                      scores: np.ndarray,
                      output_dir: str,
                      samples_per_page: int = 6):
    """
    Tüm test örnekleri için ısı haritalarını kaydet.
    Her sayfada samples_per_page adet örnek olacak şekilde kaydet.
    
    Args:
        images_paths: Görüntü dosya yolları
        anomaly_maps: Anomali haritaları
        labels: Gerçek etiketler (0=good, 1=defect)
        scores: Anomali skorları
        output_dir: Çıktı klasörü
        samples_per_page: Her sayfadaki örnek sayısı
    """
    heatmaps_dir = os.path.join(output_dir, "all_heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    n_samples = len(images_paths)
    n_pages = (n_samples + samples_per_page - 1) // samples_per_page
    
    print(f"\n📸 Saving heatmaps for all {n_samples} test samples...")
    print(f"   Output: {heatmaps_dir}")
    print(f"   Pages: {n_pages} ({samples_per_page} samples per page)")
    
    for page in range(n_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, n_samples)
        page_indices = list(range(start_idx, end_idx))
        
        n_rows = len(page_indices)
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(page_indices):
            # Dosya adını al
            file_name = Path(images_paths[idx]).stem
            
            # Orijinal görüntüyü oku
            img = cv2.imdecode(
                np.fromfile(images_paths[idx], dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            
            anomaly_map = anomaly_maps[idx]
            label = "Good" if labels[idx] == 0 else "Defect"
            score = scores[idx]
            
            # Normalize heatmap
            heatmap_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            heatmap_color = cv2.applyColorMap(
                np.uint8(255 * heatmap_norm), 
                cv2.COLORMAP_JET
            )
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
            
            # Original - dosya adı ile
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'{file_name}\n[{label}] Score: {score:.4f}', fontsize=9)
            axes[i, 0].axis('off')
            
            # Heatmap
            axes[i, 1].imshow(heatmap_norm, cmap='jet')
            axes[i, 1].set_title('Anomaly Heatmap', fontsize=9)
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay', fontsize=9)
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Anomaly Heatmaps - Page {page + 1}/{n_pages}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Sayfayı kaydet
        save_path = os.path.join(heatmaps_dir, f"heatmaps_page_{page + 1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Bellek temizliği için kapat
        
        print(f"   ✓ Page {page + 1}/{n_pages} saved")
    
    print(f"   ✅ All heatmaps saved to: {heatmaps_dir}")


def print_results(metrics: dict):
    """Sonuçları formatla yazdır"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS - PatchCore")
    print("="*60)
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"  F1 Score    : {metrics['f1_score']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Threshold   : {metrics['threshold']:.4f}")
    print("="*60)


# ============================================================================
# Grid Search for Hyperparameter Optimization
# ============================================================================

def run_single_experiment(config: dict, train_loader, test_loader, device: str) -> dict:
    """
    Tek bir konfigürasyon için model eğitip değerlendir.
    
    Args:
        config: Model konfigürasyonu
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: 'cuda' veya 'cpu'
    
    Returns:
        Sonuçlar dict'i
    """
    try:
        # Model oluştur
        model = PatchCoreModel(
            backbone=config['backbone'],
            device=device,
            memory_bank_size=config['memory_bank_size'],
            reduce_dimensions=config['reduce_dimensions'],
            target_dim=config.get('target_dim', 550),
            k_neighbors=config['k_neighbors']
        )
        
        # Eğit
        model.fit(train_loader)
        
        # Prediction
        scores, anomaly_maps, labels, paths = model.predict(test_loader)
        
        # Optimal threshold
        optimal_threshold = find_optimal_threshold(labels, scores)
        
        # Metrics
        metrics = calculate_metrics(labels, scores, threshold=optimal_threshold)
        
        result = {
            **config,
            'auc_roc': metrics['auc_roc'],
            'f1_score': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'threshold': optimal_threshold,
            'status': 'success'
        }
        
        return result, model, scores, anomaly_maps, labels, paths
        
    except Exception as e:
        result = {
            **config,
            'auc_roc': 0,
            'f1_score': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'threshold': 0,
            'status': f'error: {str(e)}'
        }
        return result, None, None, None, None, None


def run_grid_search():
    """
    Kapsamlı grid search ile en iyi parametreleri bul.
    Tüm kombinasyonları test eder ve sonuçları kaydeder.
    """
    import pandas as pd
    from datetime import datetime
    import itertools
    
    print("\n" + "="*70)
    print("🔍 PATCHCORE HYPERPARAMETER GRID SEARCH")
    print("="*70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📌 Device: {device}")
    
    # Grid search output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_output_dir = os.path.join(MODEL_OUTPUT_DIR, f"grid_search_{timestamp}")
    os.makedirs(search_output_dir, exist_ok=True)
    print(f"📁 Output directory: {search_output_dir}")
    
    # ========================================================================
    # HYPERPARAMETER GRID - Tüm test edilecek değerler
    # ========================================================================
    
    param_grid = {
        'backbone': ['wide_resnet50_2', 'resnet50'],  # 'resnet18' daha hızlı ama daha az güçlü
        'memory_bank_size': [1000, 2000, 3000],
        'reduce_dimensions': [True, False],
        'target_dim': [512, 1024],  # reduce_dimensions=True ise kullanılır
        'k_neighbors': [1, 3, 5, 9],
        'image_size': [(224, 224), (256, 256)],
    }
    
    # Tüm kombinasyonları oluştur
    all_configs = []
    for backbone in param_grid['backbone']:
        for memory_bank_size in param_grid['memory_bank_size']:
            for reduce_dims in param_grid['reduce_dimensions']:
                for k_neighbors in param_grid['k_neighbors']:
                    for image_size in param_grid['image_size']:
                        if reduce_dims:
                            for target_dim in param_grid['target_dim']:
                                all_configs.append({
                                    'backbone': backbone,
                                    'memory_bank_size': memory_bank_size,
                                    'reduce_dimensions': reduce_dims,
                                    'target_dim': target_dim,
                                    'k_neighbors': k_neighbors,
                                    'image_size': image_size
                                })
                        else:
                            all_configs.append({
                                'backbone': backbone,
                                'memory_bank_size': memory_bank_size,
                                'reduce_dimensions': reduce_dims,
                                'target_dim': None,
                                'k_neighbors': k_neighbors,
                                'image_size': image_size
                            })
    
    total_experiments = len(all_configs)
    print(f"\n📊 Total experiments to run: {total_experiments}")
    print(f"\n⚙️  Parameter Grid:")
    for key, values in param_grid.items():
        print(f"   {key}: {values}")
    
    # Results storage
    all_results = []
    best_result = None
    best_model = None
    best_auc = 0
    
    # ========================================================================
    # RUN EXPERIMENTS
    # ========================================================================
    
    for exp_idx, config in enumerate(all_configs):
        print(f"\n{'='*70}")
        print(f"🧪 Experiment {exp_idx + 1}/{total_experiments}")
        print(f"{'='*70}")
        print(f"   Backbone: {config['backbone']}")
        print(f"   Memory Bank: {config['memory_bank_size']}")
        print(f"   Reduce Dims: {config['reduce_dimensions']}")
        if config['reduce_dimensions']:
            print(f"   Target Dim: {config['target_dim']}")
        print(f"   K-Neighbors: {config['k_neighbors']}")
        print(f"   Image Size: {config['image_size']}")
        
        # Dataset oluştur (image_size'a göre)
        train_dataset = WoodDataset(
            good_path=TRAIN_GOOD_PATH,
            defect_path=None,
            image_size=config['image_size']
        )
        
        test_dataset = WoodDataset(
            good_path=TEST_GOOD_PATH,
            defect_path=TEST_DEFECT_PATH,
            image_size=config['image_size']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Experiment çalıştır
        result, model, scores, anomaly_maps, labels, paths = run_single_experiment(
            config, train_loader, test_loader, device
        )
        
        # Config'i string'e çevir (image_size tuple olduğu için)
        result['image_size'] = str(config['image_size'])
        all_results.append(result)
        
        if result['status'] == 'success':
            print(f"\n   📊 Results:")
            print(f"      AUC-ROC: {result['auc_roc']:.4f}")
            print(f"      F1 Score: {result['f1_score']:.4f}")
            print(f"      Accuracy: {result['accuracy']:.4f}")
            
            # Best model güncelle
            if result['auc_roc'] > best_auc:
                best_auc = result['auc_roc']
                best_result = result
                best_model = model
                best_scores = scores
                best_maps = anomaly_maps
                best_labels = labels
                best_paths = paths
                print(f"      ⭐ NEW BEST MODEL!")
        else:
            print(f"\n   ❌ Error: {result['status']}")
        
        # Ara sonuçları kaydet (her 5 experimentte bir)
        if (exp_idx + 1) % 5 == 0 or exp_idx == total_experiments - 1:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(search_output_dir, "grid_search_results.csv"), index=False)
            print(f"\n   💾 Intermediate results saved ({exp_idx + 1}/{total_experiments})")
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 GRID SEARCH COMPLETED!")
    print("="*70)
    
    # Results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('auc_roc', ascending=False)
    
    # CSV kaydet
    csv_path = os.path.join(search_output_dir, "grid_search_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved to: {csv_path}")
    
    # Top 10 sonuç
    print("\n" + "="*70)
    print("🏆 TOP 10 CONFIGURATIONS (by AUC-ROC)")
    print("="*70)
    
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n#{top_10.index.get_loc(idx) + 1}:")
        print(f"   Backbone: {row['backbone']}, Memory: {row['memory_bank_size']}")
        print(f"   Reduce: {row['reduce_dimensions']}, Target: {row['target_dim']}, K: {row['k_neighbors']}")
        print(f"   Image Size: {row['image_size']}")
        print(f"   AUC: {row['auc_roc']:.4f}, F1: {row['f1_score']:.4f}, Acc: {row['accuracy']:.4f}")
    
    # Best model kaydet
    if best_model is not None:
        print("\n" + "="*70)
        print("⭐ BEST MODEL")
        print("="*70)
        print(f"   Backbone: {best_result['backbone']}")
        print(f"   Memory Bank: {best_result['memory_bank_size']}")
        print(f"   Reduce Dims: {best_result['reduce_dimensions']}")
        print(f"   Target Dim: {best_result['target_dim']}")
        print(f"   K-Neighbors: {best_result['k_neighbors']}")
        print(f"   Image Size: {best_result['image_size']}")
        print(f"\n   📊 Metrics:")
        print(f"      AUC-ROC: {best_result['auc_roc']:.4f}")
        print(f"      F1 Score: {best_result['f1_score']:.4f}")
        print(f"      Accuracy: {best_result['accuracy']:.4f}")
        print(f"      Precision: {best_result['precision']:.4f}")
        print(f"      Recall: {best_result['recall']:.4f}")
        
        # Best model kaydet
        best_model_path = os.path.join(search_output_dir, "best_model.pkl")
        best_model.save(best_model_path)
        
        # Best model için görselleştirmeler
        optimal_threshold = best_result['threshold']
        y_pred = (best_scores >= optimal_threshold).astype(int)
        
        # Confusion Matrix
        plot_confusion_matrix(
            best_labels, y_pred,
            save_path=os.path.join(search_output_dir, "best_confusion_matrix.png")
        )
        
        # ROC Curve
        plot_roc_curve(
            best_labels, best_scores,
            save_path=os.path.join(search_output_dir, "best_roc_curve.png")
        )
        
        # Score Distribution
        plot_score_distribution(
            best_labels, best_scores, optimal_threshold,
            save_path=os.path.join(search_output_dir, "best_score_distribution.png")
        )
        
        # Heatmaps
        save_all_heatmaps(
            best_paths, best_maps, best_labels, best_scores,
            output_dir=search_output_dir,
            samples_per_page=6
        )
        
        # Best config kaydet
        with open(os.path.join(search_output_dir, "best_config.txt"), 'w') as f:
            f.write("BEST PATCHCORE CONFIGURATION\n")
            f.write("="*50 + "\n\n")
            for key, value in best_result.items():
                f.write(f"{key}: {value}\n")
    
    # Visualization: Performance vs Parameters
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: AUC by Backbone
    plt.subplot(2, 3, 1)
    for backbone in param_grid['backbone']:
        subset = results_df[results_df['backbone'] == backbone]
        plt.scatter([backbone] * len(subset), subset['auc_roc'], alpha=0.6, s=50)
    plt.xlabel('Backbone')
    plt.ylabel('AUC-ROC')
    plt.title('AUC by Backbone')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: AUC by Memory Bank Size
    plt.subplot(2, 3, 2)
    for mb_size in param_grid['memory_bank_size']:
        subset = results_df[results_df['memory_bank_size'] == mb_size]
        plt.scatter([mb_size] * len(subset), subset['auc_roc'], alpha=0.6, s=50)
    plt.xlabel('Memory Bank Size')
    plt.ylabel('AUC-ROC')
    plt.title('AUC by Memory Bank Size')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: AUC by K-Neighbors
    plt.subplot(2, 3, 3)
    for k in param_grid['k_neighbors']:
        subset = results_df[results_df['k_neighbors'] == k]
        plt.scatter([k] * len(subset), subset['auc_roc'], alpha=0.6, s=50)
    plt.xlabel('K-Neighbors')
    plt.ylabel('AUC-ROC')
    plt.title('AUC by K-Neighbors')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: AUC by Reduce Dimensions
    plt.subplot(2, 3, 4)
    for rd in param_grid['reduce_dimensions']:
        subset = results_df[results_df['reduce_dimensions'] == rd]
        plt.scatter([str(rd)] * len(subset), subset['auc_roc'], alpha=0.6, s=50)
    plt.xlabel('Reduce Dimensions')
    plt.ylabel('AUC-ROC')
    plt.title('AUC by Reduce Dimensions')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: F1 vs AUC scatter
    plt.subplot(2, 3, 5)
    plt.scatter(results_df['auc_roc'], results_df['f1_score'], alpha=0.6, s=50)
    plt.xlabel('AUC-ROC')
    plt.ylabel('F1 Score')
    plt.title('F1 vs AUC Correlation')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Top 10 comparison bar chart
    plt.subplot(2, 3, 6)
    top_5 = results_df.head(5)
    x_labels = [f"#{i+1}" for i in range(len(top_5))]
    plt.bar(x_labels, top_5['auc_roc'], alpha=0.7, label='AUC-ROC')
    plt.bar(x_labels, top_5['f1_score'], alpha=0.5, label='F1 Score')
    plt.xlabel('Configuration Rank')
    plt.ylabel('Score')
    plt.title('Top 5 Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(search_output_dir, "parameter_analysis.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Grid search completed!")
    print(f"📁 All results saved to: {search_output_dir}")
    
    return results_df, best_result


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Ana çalıştırma fonksiyonu"""
    print("\n" + "="*60)
    print("🔧 PatchCore Anomaly Detection - Wood Defect")
    print("="*60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📌 Device: {device}")
    
    # Dataset bilgileri
    print(f"\n📁 Dataset Paths:")
    print(f"  Selected Dataset: {SELECTED_DATASET}")
    print(f"  Train Good: {TRAIN_GOOD_PATH}")
    print(f"  Test Good: {TEST_GOOD_PATH}")
    print(f"  Test Defect: {TEST_DEFECT_PATH}")
    
    # Datasets oluştur
    print("\n📦 Loading datasets...")
    
    train_dataset = WoodDataset(
        good_path=TRAIN_GOOD_PATH,
        defect_path=None,  # Sadece good örnekler
        image_size=(224, 224)
    )
    
    test_dataset = WoodDataset(
        good_path=TEST_GOOD_PATH,
        defect_path=TEST_DEFECT_PATH,
        image_size=(224, 224)
    )
    
    print(f"  Train samples (good only): {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=0
    )
    
    # PatchCore Model
    model = PatchCoreModel(
        backbone='wide_resnet50_2',
        device=device,
        memory_bank_size=2000,
        reduce_dimensions=False,
        target_dim=1024,
        k_neighbors=5
    )
    
    # Training
    model.fit(train_loader)
    
    # Prediction
    scores, anomaly_maps, labels, paths = model.predict(test_loader)
    
    # Optimal threshold
    optimal_threshold = find_optimal_threshold(labels, scores)
    print(f"\n🎯 Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    
    # Metrics
    metrics = calculate_metrics(labels, scores, threshold=optimal_threshold)
    print_results(metrics)
    
    # Predictions
    y_pred = (scores >= optimal_threshold).astype(int)
    
    # Visualizations
    print("\n📈 Generating visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(
        labels, y_pred,
        save_path=os.path.join(MODEL_OUTPUT_DIR, "confusion_matrix.png")
    )
    
    # ROC Curve
    plot_roc_curve(
        labels, scores,
        save_path=os.path.join(MODEL_OUTPUT_DIR, "roc_curve.png")
    )
    
    # Score Distribution
    plot_score_distribution(
        labels, scores, optimal_threshold,
        save_path=os.path.join(MODEL_OUTPUT_DIR, "score_distribution.png")
    )
    
    # Anomaly Heatmaps (selected samples)
    visualize_anomaly_heatmaps(
        paths, anomaly_maps, labels, scores,
        n_samples=6,
        save_path=os.path.join(MODEL_OUTPUT_DIR, "anomaly_heatmaps.png")
    )
    
    # ALL heatmaps for report (6 samples per page)
    save_all_heatmaps(
        paths, anomaly_maps, labels, scores,
        output_dir=MODEL_OUTPUT_DIR,
        samples_per_page=6
    )
    
    # Model kaydet
    model_path = os.path.join(MODEL_OUTPUT_DIR, "patchcore_model.pkl")
    model.save(model_path)
    
    print("\n✅ All done!")
    print(f"📁 Results saved to: {MODEL_OUTPUT_DIR}")



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--grid-search":
        # Grid search modu
        run_grid_search()
    else:
        # Normal mod
        main()

