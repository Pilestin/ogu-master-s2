"""
PATCHCORE Model for Wood Anomaly Detection
State-of-the-art memory bank approach for anomaly detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
from utils import (load_image, calculate_metrics, plot_metrics, 
                  visualize_anomaly_detection, save_model_results,
                  preprocess_for_anomaly_detection)

class PatchCoreModel:
    """PatchCore anomaly detection model"""
    
    def __init__(self, backbone='wide_resnet50_2', memory_bank_size=1000, 
                 reduce_dimensions=True, target_dim=550):
        self.backbone_name = backbone
        self.memory_bank_size = memory_bank_size
        self.reduce_dimensions = reduce_dimensions
        self.target_dim = target_dim
        self.feature_extractor = None
        self.memory_bank = None
        self.random_projection = None
        self.nn_model = None
        
        self._build_feature_extractor()
    
    def _build_feature_extractor(self):
        """Feature extractor oluştur"""
        if self.backbone_name == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(pretrained=True)
            # Layer2 ve Layer3 feature'larını kullan
            self.feature_extractor = nn.ModuleDict({
                'layer1': nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    backbone.layer1
                ),
                'layer2': backbone.layer2,
                'layer3': backbone.layer3
            })
        elif self.backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            self.feature_extractor = nn.ModuleDict({
                'layer1': nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    backbone.layer1
                ),
                'layer2': backbone.layer2,
                'layer3': backbone.layer3
            })
        
        # Eval mode
        for module in self.feature_extractor.values():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    def _extract_features(self, images):
        """Multi-scale features çıkar"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for module in self.feature_extractor.values():
            module.to(device)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        
        with torch.no_grad():
            for img in images:
                # Görüntüyü uint8'e çevir
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Transform uygula
                img_tensor = transform(img_uint8).unsqueeze(0).to(device)
                
                # Multi-scale features
                x = self.feature_extractor['layer1'](img_tensor)
                feat1 = x
                
                x = self.feature_extractor['layer2'](x)
                feat2 = x
                
                x = self.feature_extractor['layer3'](x)
                feat3 = x
                
                # Feature'ları aynı boyuta getir ve birleştir
                feat2_resized = nn.functional.interpolate(
                    feat2, size=feat3.shape[-2:], mode='bilinear', align_corners=False
                )
                feat1_resized = nn.functional.interpolate(
                    feat1, size=feat3.shape[-2:], mode='bilinear', align_corners=False
                )
                
                # Concatenate features
                combined_features = torch.cat([feat1_resized, feat2_resized, feat3], dim=1)
                combined_features = combined_features.cpu().numpy().squeeze()
                
                all_features.append(combined_features)
        
        return np.array(all_features)
    
    def _extract_patches(self, features):
        """Feature map'lerden patch'leri çıkar"""
        batch_size, channels, height, width = features.shape
        
        # Her spatial location için feature vector
        patches = features.reshape(batch_size, channels, -1)  # (B, C, H*W)
        patches = patches.transpose(0, 2, 1)  # (B, H*W, C)
        patches = patches.reshape(-1, channels)  # (B*H*W, C)
        
        return patches
    
    def _coreset_sampling(self, features, k):
        """Coreset sampling ile representative patches seç"""
        print(f"Coreset sampling: {len(features)} -> {k}")
        
        if len(features) <= k:
            return features
        
        # Greedy coreset sampling
        # İlk nokta rastgele seç
        selected_indices = [np.random.randint(len(features))]
        selected_features = [features[selected_indices[0]]]
        
        for i in range(1, k):
            if i % 100 == 0:
                print(f"Coreset progress: {i}/{k}")
            
            # Her nokta için en yakın seçili noktaya olan mesafeyi hesapla
            distances = []
            for j, feature in enumerate(features):
                if j in selected_indices:
                    distances.append(0)  # Zaten seçilmiş
                else:
                    # En yakın seçili noktaya olan mesafe
                    min_dist = float('inf')
                    for selected_feature in selected_features:
                        dist = np.linalg.norm(feature - selected_feature)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            
            # En büyük mesafeye sahip noktayı seç
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            selected_features.append(features[next_idx])
        
        return np.array(selected_features)
    
    def fit(self, train_images):
        """Normal verilerle memory bank oluştur"""
        print("PatchCore modeli eğitiliyor...")
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
        
        # Coreset sampling ile memory bank oluştur
        if len(patches) > self.memory_bank_size:
            self.memory_bank = self._coreset_sampling(patches, self.memory_bank_size)
        else:
            self.memory_bank = patches
        
        print(f"Memory bank oluşturuldu: {self.memory_bank.shape}")
        
        # Nearest neighbors model
        self.nn_model = NearestNeighbors(
            n_neighbors=1, 
            algorithm='auto', 
            metric='euclidean'
        )
        self.nn_model.fit(self.memory_bank)
        
        print("PatchCore eğitimi tamamlandı!")
    
    def predict(self, test_images):
        """Test görüntüleri için anomaly score hesapla"""
        print("PatchCore tahmin yapıyor...")
        
        # Features çıkar
        features = self._extract_features(test_images)
        
        # Patch'leri çıkar
        patches = self._extract_patches(features)
        
        # Dimension reduction (eğitimde kullanıldıysa)
        if self.random_projection is not None:
            patches = self.random_projection.transform(patches)
        
        # Her patch için en yakın memory bank elementine olan mesafeyi hesapla
        distances, _ = self.nn_model.kneighbors(patches)
        distances = distances.squeeze()
        
        # Her görüntü için anomaly score hesapla
        n_patches_per_image = features.shape[2] * features.shape[3]
        n_images = len(test_images)
        
        anomaly_scores = []
        anomaly_maps = []
        
        for img_idx in range(n_images):
            start_idx = img_idx * n_patches_per_image
            end_idx = start_idx + n_patches_per_image
            
            img_distances = distances[start_idx:end_idx]
            
            # Görüntü seviyesinde score (max patch distance)
            img_score = np.max(img_distances)
            anomaly_scores.append(img_score)
            
            # Anomaly map oluştur
            h, w = features.shape[2], features.shape[3]
            anomaly_map = img_distances.reshape(h, w)
            
            # Test görüntüsünün boyutuna resize et
            anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
            anomaly_maps.append(anomaly_map_resized)
        
        return np.array(anomaly_scores), np.array(anomaly_maps)
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        model_data = {
            'memory_bank': self.memory_bank,
            'random_projection': self.random_projection,
            'nn_model': self.nn_model,
            'backbone_name': self.backbone_name,
            'memory_bank_size': self.memory_bank_size,
            'reduce_dimensions': self.reduce_dimensions,
            'target_dim': self.target_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """Modeli yükle"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.memory_bank = model_data['memory_bank']
        self.random_projection = model_data['random_projection']
        self.nn_model = model_data['nn_model']
        self.backbone_name = model_data['backbone_name']
        self.memory_bank_size = model_data['memory_bank_size']
        self.reduce_dimensions = model_data['reduce_dimensions']
        self.target_dim = model_data['target_dim']
        
        print(f"Model yüklendi: {filepath}")

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
    print("=== PATCHCORE ANOMALY DETECTION ===\n")
    
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
    model = PatchCoreModel(
        backbone='resnet18',  # Daha hızlı test için
        memory_bank_size=500,  # Küçük dataset için
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
    plot_metrics(results, "PatchCore")
    
    # Anomali tespit örneklerini göster
    visualize_anomaly_detection(
        test_images, test_labels, anomaly_scores_norm, anomaly_maps,
        "PatchCore", num_samples=6
    )
    
    # Sonuçları kaydet
    save_model_results("PatchCore", results)
    model.save_model("patchcore_model.pkl")
    
    print(f"\n=== PATCHCORE SONUÇLARI ===")
    print(f"AUC Score: {results['auc_score']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("="*35)

if __name__ == "__main__":
    main()
