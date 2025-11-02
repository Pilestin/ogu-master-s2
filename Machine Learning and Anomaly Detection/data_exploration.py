"""
Ahşap Anomali Tespiti - Veri Keşfi ve Analizi
Bu script dataset'i analiz eder ve örnek görüntüleri görselleştirir.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from pathlib import Path

# Dataset yolları
DATASET_ROOT = "dataset/wood"
TRAIN_GOOD_PATH = os.path.join(DATASET_ROOT, "train", "good")
TEST_GOOD_PATH = os.path.join(DATASET_ROOT, "test", "good")
TEST_DEFECT_PATH = os.path.join(DATASET_ROOT, "test", "defect")

def load_image(image_path):
    """Görüntü yükle ve RGB formatına çevir"""
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None

def analyze_image_stats(image_paths, label):
    """Görüntü istatistiklerini analiz et"""
    heights, widths, channels = [], [], []
    file_sizes = []
    
    for path in image_paths:
        img = load_image(path)
        if img is not None:
            h, w, c = img.shape
            heights.append(h)
            widths.append(w)
            channels.append(c)
            file_sizes.append(os.path.getsize(path) / (1024*1024))  # MB
    
    stats = {
        'label': label,
        'count': len(image_paths),
        'height_mean': np.mean(heights),
        'height_std': np.std(heights),
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
        'channels': channels[0] if channels else 0,
        'file_size_mb_mean': np.mean(file_sizes),
        'file_size_mb_std': np.std(file_sizes)
    }
    
    return stats

def visualize_samples(image_paths, title, num_samples=6):
    """Örnek görüntüleri görselleştir"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    selected_paths = np.random.choice(image_paths, min(num_samples, len(image_paths)), replace=False)
    
    for i, path in enumerate(selected_paths):
        row, col = i // 3, i % 3
        img = load_image(path)
        if img is not None:
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{os.path.basename(path)}", fontsize=8)
            axes[row, col].axis('off')
    
    # Boş subplot'ları gizle
    for i in range(len(selected_paths), 6):
        row, col = i // 3, i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def compute_image_histogram(image_paths, label, num_samples=10):
    """Görüntülerin histogramlarını hesapla"""
    selected_paths = np.random.choice(image_paths, min(num_samples, len(image_paths)), replace=False)
    
    r_hist, g_hist, b_hist = [], [], []
    
    for path in selected_paths:
        img = load_image(path)
        if img is not None:
            r_hist.append(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten())
            g_hist.append(cv2.calcHist([img], [1], None, [256], [0, 256]).flatten())
            b_hist.append(cv2.calcHist([img], [2], None, [256], [0, 256]).flatten())
    
    # Ortalama histogram
    r_mean = np.mean(r_hist, axis=0)
    g_mean = np.mean(g_hist, axis=0)
    b_mean = np.mean(b_hist, axis=0)
    
    plt.figure(figsize=(12, 4))
    plt.plot(r_mean, color='red', alpha=0.7, label='Red')
    plt.plot(g_mean, color='green', alpha=0.7, label='Green')
    plt.plot(b_mean, color='blue', alpha=0.7, label='Blue')
    plt.title(f'Ortalama Renk Histogramı - {label}')
    plt.xlabel('Pixel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Ana fonksiyon"""
    print("=== AHŞAP ANOMALİ TESPİTİ - VERİ ANALİZİ ===\n")
    
    # Dosya yollarını topla
    train_good_files = [os.path.join(TRAIN_GOOD_PATH, f) for f in os.listdir(TRAIN_GOOD_PATH) if f.endswith('.bmp')]
    test_good_files = [os.path.join(TEST_GOOD_PATH, f) for f in os.listdir(TEST_GOOD_PATH) if f.endswith('.bmp')]
    test_defect_files = [os.path.join(TEST_DEFECT_PATH, f) for f in os.listdir(TEST_DEFECT_PATH) if f.endswith('.bmp')]
    
    # İstatistikleri hesapla
    stats_train_good = analyze_image_stats(train_good_files, "Train Good")
    stats_test_good = analyze_image_stats(test_good_files, "Test Good")
    stats_test_defect = analyze_image_stats(test_defect_files, "Test Defect")
    
    # İstatistikleri yazdır
    print("Dataset İstatistikleri:")
    print("-" * 50)
    for stats in [stats_train_good, stats_test_good, stats_test_defect]:
        print(f"\n{stats['label']}:")
        print(f"  Dosya sayısı: {stats['count']}")
        print(f"  Ortalama boyut: {stats['width_mean']:.0f} x {stats['height_mean']:.0f}")
        print(f"  Kanal sayısı: {stats['channels']}")
        print(f"  Ortalama dosya boyutu: {stats['file_size_mb_mean']:.2f} ± {stats['file_size_mb_std']:.2f} MB")
    
    print("\n" + "="*60)
    print("GÖRSELLEŞTİRME BAŞLIYOR...")
    print("="*60)
    
    # Örnek görüntüleri göster
    visualize_samples(train_good_files, "Training Set - Good Samples")
    visualize_samples(test_good_files, "Test Set - Good Samples")
    visualize_samples(test_defect_files, "Test Set - Defect Samples")
    
    # Histogramları göster
    compute_image_histogram(train_good_files, "Train Good")
    compute_image_histogram(test_good_files, "Test Good")
    compute_image_histogram(test_defect_files, "Test Defect")
    
    print("\n=== VERİ ANALİZİ TAMAMLANDI ===")
    
    # Bir örnek görüntü yükleyip boyutunu kontrol et
    sample_img = load_image(train_good_files[0])
    if sample_img is not None:
        print(f"\nÖrnek görüntü boyutu: {sample_img.shape}")
        print(f"Pixel değer aralığı: {sample_img.min()} - {sample_img.max()}")

if __name__ == "__main__":
    main()
