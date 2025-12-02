"""
Utility Functions for Anomaly Detection Models
Yardımcı fonksiyonlar ve ortak kullanılan işlemler
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve
)
import json
import pandas as pd


def load_image(image_path, target_size=(256, 256)):
    """
    Görüntü dosyasını yükler ve RGB formatına çevirir
    
    Args:
        image_path: Görüntü dosyasının yolu
        target_size: Hedef boyut (width, height)
    
    Returns:
        RGB formatında numpy array veya None
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Görüntü yüklenirken hata: {image_path} - {e}")
        return None


def normalize_image(image):
    """
    Görüntüyü 0-1 aralığına normalize eder
    
    Args:
        image: Numpy array (0-255)
    
    Returns:
        Normalize edilmiş görüntü (0-1)
    """
    return image.astype(np.float32) / 255.0


def preprocess_for_anomaly_detection(image, target_size=(256, 256)):
    """
    Anomali tespiti için görüntü ön işleme
    
    Args:
        image: RGB görüntü
        target_size: Hedef boyut
    
    Returns:
        Normalize edilmiş ve resize edilmiş görüntü
    """
    # Resize
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize (0-1)
    image = normalize_image(image)
    
    return image


def create_anomaly_map(original, reconstructed):
    """
    Orijinal ve yeniden oluşturulmuş görüntüler arasındaki farkı hesaplar
    
    Args:
        original: Orijinal görüntü
        reconstructed: Yeniden oluşturulmuş görüntü
    
    Returns:
        Anomali haritası
    """
    # MSE hesapla (her piksel için)
    diff = np.mean(np.square(original - reconstructed), axis=-1)
    
    # Normalize et
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    return diff


def calculate_metrics(y_true, y_scores, threshold=0.5):
    """
    Sınıflandırma metriklerini hesaplar
    
    Args:
        y_true: Gerçek etiketler (0: normal, 1: anomaly)
        y_scores: Anomali skorları (0-1 arası)
        threshold: Karar eşiği
    
    Returns:
        Metrik sözlüğü
    """
    # Binary predictions
    y_pred = (y_scores >= threshold).astype(int)
    
    # Metrikler
    results = {
        'auc_score': roc_auc_score(y_true, y_scores),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'threshold': threshold,
        'y_scores': y_scores.tolist() if isinstance(y_scores, np.ndarray) else y_scores,
        'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else y_true
    }
    
    return results


def plot_metrics(results, model_name="Model"):
    """
    Metrik sonuçlarını görselleştirir
    
    Args:
        results: calculate_metrics fonksiyonundan dönen sözlük
        model_name: Model adı
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[0].set_title(f'{model_name} - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Değerleri yazdır
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), 
                        ha='center', va='center', color='red')
    
    # ROC Curve
    y_true = np.array(results['y_true'])
    y_scores = np.array(results['y_scores'])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    axes[1].plot(fpr, tpr, label=f'AUC = {results["auc_score"]:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{model_name} - ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Metrikleri yazdır
    print(f"\n=== {model_name} PERFORMANS METRİKLERİ ===")
    print(f"AUC Score:    {results['auc_score']:.4f}")
    print(f"F1 Score:     {results['f1_score']:.4f}")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Threshold:    {results['threshold']:.4f}")
    print("="*45)


def visualize_anomaly_detection(images, labels, scores, anomaly_maps, 
                                model_name="Model", num_samples=6):
    """
    Anomali tespit sonuçlarını görselleştirir
    
    Args:
        images: Test görüntüleri
        labels: Gerçek etiketler
        scores: Anomali skorları
        anomaly_maps: Anomali haritaları
        model_name: Model adı
        num_samples: Gösterilecek örnek sayısı
    """
    # Her sınıftan eşit sayıda örnek seç
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    n_per_class = num_samples // 2
    
    selected_indices = []
    if len(normal_indices) > 0:
        selected_indices.extend(np.random.choice(normal_indices, 
                                                min(n_per_class, len(normal_indices)), 
                                                replace=False))
    if len(anomaly_indices) > 0:
        selected_indices.extend(np.random.choice(anomaly_indices, 
                                                 min(n_per_class, len(anomaly_indices)), 
                                                 replace=False))
    
    n_samples = len(selected_indices)
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_idx in enumerate(selected_indices):
        # Orijinal görüntü
        axes[idx, 0].imshow(images[img_idx])
        label_text = "Normal" if labels[img_idx] == 0 else "Anomaly"
        axes[idx, 0].set_title(f'{label_text} | Score: {scores[img_idx]:.3f}')
        axes[idx, 0].axis('off')
        
        # Anomaly map
        axes[idx, 1].imshow(anomaly_maps[img_idx], cmap='jet')
        axes[idx, 1].set_title('Anomaly Map')
        axes[idx, 1].axis('off')
    
    plt.suptitle(f'{model_name} - Anomaly Detection Results', fontsize=14)
    plt.tight_layout()
    plt.show()


def save_model_results(model_name, results, output_dir="results"):
    """
    Model sonuçlarını dosyaya kaydeder
    
    Args:
        model_name: Model adı
        results: Metrik sonuçları
        output_dir: Çıktı dizini
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON olarak kaydet
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    
    # Numpy array'leri liste'ye çevir
    results_copy = results.copy()
    
    with open(output_file, 'w') as f:
        json.dump(results_copy, f, indent=4)
    
    print(f"\n✅ Sonuçlar kaydedildi: {output_file}")


def compare_models(results_list, model_names):
    """
    Birden fazla modelin performansını karşılaştırır
    
    Args:
        results_list: Model sonuçları listesi
        model_names: Model isimleri listesi
    """
    metrics_to_compare = ['auc_score', 'f1_score', 'precision', 'recall', 'accuracy']
    
    # DataFrame oluştur
    comparison_data = []
    for model_name, results in zip(model_names, results_list):
        row = {'Model': model_name}
        for metric in metrics_to_compare:
            row[metric] = results.get(metric, 0)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Görselleştirme
    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(18, 4))
    
    for idx, metric in enumerate(metrics_to_compare):
        axes[idx].bar(df['Model'], df[metric])
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Tablo yazdır
    print("\n=== MODEL KARŞILAŞTIRMA TABLOSU ===")
    print(df.to_string(index=False))
    print("="*60)
    
    return df
