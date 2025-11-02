"""
Utilities for Wood Anomaly Detection Project
Genel yardımcı fonksiyonlar ve veri yükleme araçları
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

class WoodDataset(Dataset):
    """Ahşap görüntüleri için PyTorch Dataset sınıfı"""
    
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        if is_train:
            # Sadece good örnekleri yükle (train)
            good_dir = os.path.join(root_dir, "train", "good")
            for img_name in os.listdir(good_dir):
                if img_name.endswith('.bmp'):
                    self.image_paths.append(os.path.join(good_dir, img_name))
                    self.labels.append(0)  # 0 = normal
        else:
            # Test: hem good hem defect yükle
            good_dir = os.path.join(root_dir, "test", "good")
            defect_dir = os.path.join(root_dir, "test", "defect")
            
            for img_name in os.listdir(good_dir):
                if img_name.endswith('.bmp'):
                    self.image_paths.append(os.path.join(good_dir, img_name))
                    self.labels.append(0)  # 0 = normal
                    
            for img_name in os.listdir(defect_dir):
                if img_name.endswith('.bmp'):
                    self.image_paths.append(os.path.join(defect_dir, img_name))
                    self.labels.append(1)  # 1 = anomaly
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükle
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def load_image(image_path, target_size=(256, 256)):
    """Görüntü yükle ve boyutlandır"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image

def normalize_image(image):
    """Görüntüyü 0-1 aralığına normalize et"""
    return image.astype(np.float32) / 255.0

def create_anomaly_map(original, reconstructed):
    """Anomali haritası oluştur"""
    diff = np.abs(original - reconstructed)
    anomaly_map = np.mean(diff, axis=-1)  # RGB kanallarının ortalaması
    return anomaly_map

def calculate_metrics(y_true, y_scores, threshold=0.5):
    """Model değerlendirme metriklerini hesapla"""
    y_pred = (y_scores > threshold).astype(int)
    
    # Temel metrikler
    auc_score = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    
    # ROC eğrisi
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Precision-Recall eğrisi
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    results = {
        'auc_score': auc_score,
        'f1_score': f1,
        'y_pred': y_pred,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return results

def plot_metrics(results, model_name="Model"):
    """Değerlendirme metriklerini görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Değerlendirme Sonuçları', fontsize=16)
    
    # ROC Curve
    axes[0, 0].plot(results['fpr'], results['tpr'], 'b-', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title(f'ROC Curve (AUC = {results["auc_score"]:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    axes[0, 1].plot(results['recall'], results['precision'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Metrik değerleri text olarak göster
    metrics_text = f"""
    AUC Score: {results['auc_score']:.3f}
    F1 Score: {results['f1_score']:.3f}
    
    Detailed Classification Report:
    {results['classification_report']}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, 
                     verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_anomaly_detection(images, labels, anomaly_scores, anomaly_maps, 
                               model_name="Model", num_samples=6):
    """Anomali tespit sonuçlarını görselleştir"""
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 12))
    fig.suptitle(f'{model_name} - Anomaly Detection Results', fontsize=16)
    
    # Farklı kategorilerden örnekler seç
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    half_samples = num_samples // 2
    selected_normal = np.random.choice(normal_indices, half_samples, replace=False)
    selected_anomaly = np.random.choice(anomaly_indices, half_samples, replace=False)
    selected_indices = np.concatenate([selected_normal, selected_anomaly])
    
    for i, idx in enumerate(selected_indices):
        # Orijinal görüntü
        axes[0, i].imshow(images[idx])
        axes[0, i].set_title(f'Original\nLabel: {"Normal" if labels[idx] == 0 else "Anomaly"}')
        axes[0, i].axis('off')
        
        # Anomali haritası
        if anomaly_maps is not None:
            axes[1, i].imshow(anomaly_maps[idx], cmap='hot')
            axes[1, i].set_title(f'Anomaly Map\nScore: {anomaly_scores[idx]:.3f}')
        else:
            axes[1, i].text(0.5, 0.5, f'Score: {anomaly_scores[idx]:.3f}', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title('Anomaly Score')
        axes[1, i].axis('off')
        
        # Threshold uygulanmış sonuç
        prediction = "ANOMALY" if anomaly_scores[idx] > 0.5 else "NORMAL"
        color = 'red' if prediction == "ANOMALY" else 'green'
        axes[2, i].text(0.5, 0.5, prediction, ha='center', va='center', 
                       transform=axes[2, i].transAxes, fontsize=16, 
                       color=color, fontweight='bold')
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
        
        # Doğru/Yanlış göstergesi
        is_correct = (labels[idx] == 1 and anomaly_scores[idx] > 0.5) or \
                    (labels[idx] == 0 and anomaly_scores[idx] <= 0.5)
        result_text = "✓ CORRECT" if is_correct else "✗ WRONG"
        result_color = 'green' if is_correct else 'red'
        axes[3, i].text(0.5, 0.5, result_text, ha='center', va='center',
                       transform=axes[3, i].transAxes, fontsize=14,
                       color=result_color, fontweight='bold')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_model_results(model_name, results, save_dir="results"):
    """Model sonuçlarını dosyaya kaydet"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Sonuçları text dosyasına kaydet
    with open(os.path.join(save_dir, f"{model_name}_results.txt"), 'w') as f:
        f.write(f"=== {model_name} Sonuçları ===\n\n")
        f.write(f"AUC Score: {results['auc_score']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
    
    print(f"Sonuçlar kaydedildi: {save_dir}/{model_name}_results.txt")

def preprocess_for_anomaly_detection(image, target_size=(256, 256)):
    """Anomali tespiti için görüntü ön işleme"""
    # Boyutlandır
    image = cv2.resize(image, target_size)
    
    # Normalize et
    image = image.astype(np.float32) / 255.0
    
    return image
