"""
PaDiM Model for Wood Anomaly Detection
Feature-based anomaly detection using patch distribution modeling
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from utils import (load_image, calculate_metrics, plot_metrics, 
                  visualize_anomaly_detection, save_model_results,
                  preprocess_for_anomaly_detection)

class PaDiMModel:
    """PaDiM anomaly detection model"""
    
    def __init__(self, backbone='resnet18', reduce_dimensions=True, target_dim=100):
        self.backbone_name = backbone
        self.reduce_dimensions = reduce_dimensions
        self.target_dim = target_dim
        self.feature_extractor = None
        self.random_projection = None
        self.patch_means = None
        self.patch_covariances = None
        self.patch_inv_covariances = None
        self.image_size = None
        self.patch_size = None
        
        self._build_feature_extractor()
    
    def _build_feature_extractor(self):
        """Feature extractor oluştur"""
        if self.backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            # İlk 3 layer kullan (layer1, layer2, layer3)
            self.feature_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3
            )
            self.feature_dimensions = [64, 128, 256]  # Her layer'ın çıkış channel sayısı
        
        # Eval mode
        self.feature_extractor.eval()
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _extract_features(self, images):
        """Deep features çıkar"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(device)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        features_list = []
        
        with torch.no_grad():
            for img in images:
                # Görüntüyü uint8'e çevir
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Transform uygula
                img_tensor = transform(img_uint8).unsqueeze(0).to(device)
                
                # Features çıkar
                features = self.feature_extractor(img_tensor)
                features = features.cpu().numpy().squeeze()
                
                # Spatial dimension'ları koru
                features_resized = cv2.resize(
                    features.transpose(1, 2, 0), 
                    (28, 28)  # 224/8 = 28 (resnet layer3 output size)
                ).transpose(2, 0, 1)
                
                features_list.append(features_resized)
        
        return np.array(features_list)
    
    def _extract_patches(self, features):
        """Feature map'lerden patch'leri çıkar"""
        batch_size, channels, height, width = features.shape
        
        # Her spatial location için feature vector
        patches = features.reshape(batch_size, channels, -1)  # (B, C, H*W)
        patches = patches.transpose(0, 2, 1)  # (B, H*W, C)
        patches = patches.reshape(-1, channels)  # (B*H*W, C)
        
        return patches
    
    def fit(self, train_images):
        """Normal verilerle modeli eğit"""
        print("PaDiM modeli eğitiliyor...")
        print(f"Training verisi: {train_images.shape}")
        
        # Features çıkar
        print("Features çıkarılıyor...")
        features = self._extract_features(train_images)
        print(f"Feature shape: {features.shape}")
        
        # Patch'leri çıkar
        patches = self._extract_patches(features)
        print(f"Patch shape: {patches.shape}")
        
        # Dimension reduction
        if self.reduce_dimensions and patches.shape[1] > self.target_dim:
            print(f"Boyut azaltma: {patches.shape[1]} -> {self.target_dim}")
            self.random_projection = SparseRandomProjection(
                n_components=self.target_dim, random_state=42
            )
            patches = self.random_projection.fit_transform(patches)
        
        # Her spatial location için mean ve covariance hesapla
        print("Patch dağılımları hesaplanıyor...")
        n_patches_per_image = features.shape[2] * features.shape[3]  # H*W
        n_images = len(train_images)
        
        self.patch_means = []
        self.patch_covariances = []
        self.patch_inv_covariances = []
        
        for i in range(n_patches_per_image):
            # Bu spatial location'daki tüm patch'ler
            patch_vectors = patches[i::n_patches_per_image]  # Her n_patches_per_image'de bir al
            
            # Mean ve covariance
            mean = np.mean(patch_vectors, axis=0)
            cov = np.cov(patch_vectors.T) + np.eye(patch_vectors.shape[1]) * 1e-6
            
            self.patch_means.append(mean)
            self.patch_covariances.append(cov)
            
            # Inverse covariance (Mahalanobis distance için)
            try:
                inv_cov = np.linalg.inv(cov)
                self.patch_inv_covariances.append(inv_cov)
            except:
                # Eğer singular ise regularize et
                inv_cov = np.linalg.pinv(cov)
                self.patch_inv_covariances.append(inv_cov)
        
        self.patch_means = np.array(self.patch_means)
        self.patch_covariances = np.array(self.patch_covariances)
        self.patch_inv_covariances = np.array(self.patch_inv_covariances)
        
        print(f"Model eğitimi tamamlandı!")
        print(f"Patch means shape: {self.patch_means.shape}")
        print(f"Patch covariances shape: {self.patch_covariances.shape}")
    
    def predict(self, test_images):
        """Test görüntüleri için anomaly score hesapla"""
        print("PaDiM tahmin yapıyor...")
        
        # Features çıkar
        features = self._extract_features(test_images)
        
        # Patch'leri çıkar
        patches = self._extract_patches(features)
        
        # Dimension reduction (eğitimde kullanıldıysa)
        if self.random_projection is not None:
            patches = self.random_projection.transform(patches)
        
        # Her görüntü için anomaly score hesapla
        n_patches_per_image = features.shape[2] * features.shape[3]
        n_images = len(test_images)
        
        anomaly_scores = []
        anomaly_maps = []
        
        for img_idx in range(n_images):
            patch_scores = []
            
            for patch_idx in range(n_patches_per_image):
                # Bu görüntüdeki bu spatial location'daki patch
                global_patch_idx = img_idx * n_patches_per_image + patch_idx
                patch_vector = patches[global_patch_idx]
                
                # Mahalanobis distance hesapla
                mean = self.patch_means[patch_idx]
                inv_cov = self.patch_inv_covariances[patch_idx]
                
                try:
                    diff = patch_vector - mean
                    score = np.sqrt(diff.T @ inv_cov @ diff)
                except:
                    # Fallback: Euclidean distance
                    score = np.linalg.norm(patch_vector - mean)
                
                patch_scores.append(score)
            
            # Görüntü seviyesinde anomaly score (max patch score)
            img_score = np.max(patch_scores)
            anomaly_scores.append(img_score)
            
            # Anomaly map oluştur
            h, w = features.shape[2], features.shape[3]
            anomaly_map = np.array(patch_scores).reshape(h, w)
            
            # Test görüntüsünün boyutuna resize et
            anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
            anomaly_maps.append(anomaly_map_resized)
        
        return np.array(anomaly_scores), np.array(anomaly_maps)

def load_dataset(data_dir, target_size=(256, 256)):
    """Dataset'i yükle"""
    train_good_dir = os.path.join(data_dir, "train", "good")
    test_good_dir = os.path.join(data_dir, "test", "good")
    test_defect_dir = os.path.join(data_dir, "test", "defect")
    
    # Training verisi
    train_images = []
    train_files = [f for f in os.listdir(train_good_dir) if f.endswith('.bmp')]
    
    print(f"Training verisi yükleniyor: {len(train_files)} dosya...")
    for filename in train_files:
        img_path = os.path.join(train_good_dir, filename)
        img = load_image(img_path, target_size)
        if img is not None:
            img = preprocess_for_anomaly_detection(img, target_size)
            train_images.append(img)
    
    # Test verisi
    test_images = []
    test_labels = []
    
    # Good örnekler
    test_good_files = [f for f in os.listdir(test_good_dir) if f.endswith('.bmp')]
    print(f"Test good verisi yükleniyor: {len(test_good_files)} dosya...")
    for filename in test_good_files:
        img_path = os.path.join(test_good_dir, filename)
        img = load_image(img_path, target_size)
        if img is not None:
            img = preprocess_for_anomaly_detection(img, target_size)
            test_images.append(img)
            test_labels.append(0)
    
    # Defect örnekler
    test_defect_files = [f for f in os.listdir(test_defect_dir) if f.endswith('.bmp')]
    print(f"Test defect verisi yükleniyor: {len(test_defect_files)} dosya...")
    for filename in test_defect_files:
        img_path = os.path.join(test_defect_dir, filename)
        img = load_image(img_path, target_size)
        if img is not None:
            img = preprocess_for_anomaly_detection(img, target_size)
            test_images.append(img)
            test_labels.append(1)
    
    return np.array(train_images), np.array(test_images), np.array(test_labels)

def main():
    """Ana fonksiyon"""
    print("=== PADIM ANOMALY DETECTION ===\n")
    
    # Dataset yükle
    data_dir = "wood"
    target_size = (256, 256)
    
    train_images, test_images, test_labels = load_dataset(data_dir, target_size)
    
    print(f"\nDataset Bilgileri:")
    print(f"Training set: {train_images.shape}")
    print(f"Test set: {test_images.shape}")
    print(f"Test labels: {test_labels.shape}")
    print(f"Normal samples: {np.sum(test_labels == 0)}")
    print(f"Anomaly samples: {np.sum(test_labels == 1)}")
    
    # Model oluştur
    model = PaDiMModel(
        backbone='resnet18',
        reduce_dimensions=True,
        target_dim=100
    )
    
    # Modeli eğit
    print("\n" + "="*50)
    print("MODEL EĞİTİMİ BAŞLIYOR...")
    print("="*50)
    
    model.fit(train_images)
    
    # Test seti üzerinde tahmin yap
    print("\n" + "="*50)
    print("TEST SONUÇLARI HESAPLANIYOR...")
    print("="*50)
    
    anomaly_scores, anomaly_maps = model.predict(test_images)
    
    # Skorları normalize et
    anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Metrikleri hesapla
    results = calculate_metrics(test_labels, anomaly_scores_norm, threshold=0.5)
    
    # Sonuçları görselleştir
    plot_metrics(results, "PaDiM")
    
    # Anomali tespit örneklerini göster
    visualize_anomaly_detection(
        test_images, test_labels, anomaly_scores_norm, anomaly_maps,
        "PaDiM", num_samples=6
    )
    
    # Sonuçları kaydet
    save_model_results("PaDiM", results)
    
    print(f"\n=== PADIM SONUÇLARI ===")
    print(f"AUC Score: {results['auc_score']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
