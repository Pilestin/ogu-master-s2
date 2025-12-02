""" 
Öncesinde preprocess aşamasından geçirilmiş resimler (dataset/wood_...) 
alınarak modeller ile test edilir. 

Kullanım:
    python start_process.py --dataset wood_otsu_clahe --model autoencoder
    python start_process.py --dataset wood_otsu_clahe_gamma --model padim
    python start_process.py --dataset wood_otsu_sobel_clahe --model patchcore
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime

# Model import'ları
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.model1_autoencoder import ConvolutionalAutoencoder, load_dataset as load_dataset_ae
from models.model2_padim import PaDiMModel, load_dataset as load_dataset_padim
from models.model3_patchcore import PatchCoreModel, load_dataset as load_dataset_patchcore

from utils import (
    calculate_metrics, 
    plot_metrics, 
    visualize_anomaly_detection,
    save_model_results,
    compare_models
)


# Mevcut dataset'ler
AVAILABLE_DATASETS = [
    'wood',
    'wood_otsu_clahe',
    'wood_otsu_clahe_gamma',
    'wood_otsu_sobel_clahe',
    'wood_otsu_clahe_log',
    'wood_otsu_clahe_wb'
]

# Mevcut modeller
AVAILABLE_MODELS = {
    'autoencoder': {
        'class': ConvolutionalAutoencoder,
        'load_dataset': load_dataset_ae,
        'name': 'Convolutional Autoencoder'
    },
    'padim': {
        'class': PaDiMModel,
        'load_dataset': load_dataset_padim,
        'name': 'PaDiM'
    },
    'patchcore': {
        'class': PatchCoreModel,
        'load_dataset': load_dataset_patchcore,
        'name': 'PatchCore'
    }
}


def get_dataset_path(dataset_name):
    """Dataset yolunu döndürür"""
    base_path = "dataset"
    dataset_path = os.path.join(base_path, dataset_name)
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset bulunamadı: {dataset_path}")
    
    return dataset_path


def run_autoencoder(dataset_path, target_size=(256, 256), epochs=50, batch_size=8):
    """
    Autoencoder modelini çalıştırır
    
    Args:
        dataset_path: Dataset dizini
        target_size: Görüntü boyutu
        epochs: Eğitim epoch sayısı
        batch_size: Batch size
    
    Returns:
        results: Sonuç sözlüğü
    """
    print("\n" + "="*60)
    print("AUTOENCODER MODEL ÇALIŞTIRILIYOR")
    print("="*60)
    
    # Dataset yükle
    train_images, test_images, test_labels = load_dataset_ae(dataset_path, target_size)
    
    print(f"\nDataset Bilgileri:")
    print(f"  Training set: {train_images.shape}")
    print(f"  Test set: {test_images.shape}")
    print(f"  Normal samples: {np.sum(test_labels == 0)}")
    print(f"  Anomaly samples: {np.sum(test_labels == 1)}")
    
    # Model oluştur
    autoencoder = ConvolutionalAutoencoder(
        input_shape=(target_size[0], target_size[1], 3),
        latent_dim=128
    )
    
    model = autoencoder.build_model()
    print(f"\nModel yapısı oluşturuldu.")
    
    # Modeli eğit
    print("\n" + "-"*60)
    print("MODEL EĞİTİMİ BAŞLIYOR...")
    print("-"*60)
    
    history = autoencoder.train(
        train_images,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Eğitim geçmişini görselleştir
    autoencoder.plot_training_history()
    
    # Test seti üzerinde tahmin yap
    print("\n" + "-"*60)
    print("TEST SONUÇLARI HESAPLANIYOR...")
    print("-"*60)
    
    anomaly_scores, anomaly_maps, reconstructed = autoencoder.predict_anomaly_scores(test_images)
    
    # Skorları normalize et (0-1 aralığına)
    anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Metrikleri hesapla
    results = calculate_metrics(test_labels, anomaly_scores_norm, threshold=0.5)
    
    # Sonuçları görselleştir
    plot_metrics(results, "Autoencoder")
    
    # Anomali tespit örneklerini göster
    visualize_anomaly_detection(
        test_images, test_labels, anomaly_scores_norm, anomaly_maps,
        "Autoencoder", num_samples=6
    )
    
    return results


def run_padim(dataset_path, target_size=(256, 256)):
    """
    PaDiM modelini çalıştırır
    
    Args:
        dataset_path: Dataset dizini
        target_size: Görüntü boyutu
    
    Returns:
        results: Sonuç sözlüğü
    """
    print("\n" + "="*60)
    print("PADIM MODEL ÇALIŞTIRILIYOR")
    print("="*60)
    
    # Dataset yükle
    train_images, test_images, test_labels = load_dataset_padim(dataset_path, target_size)
    
    print(f"\nDataset Bilgileri:")
    print(f"  Training set: {train_images.shape}")
    print(f"  Test set: {test_images.shape}")
    print(f"  Normal samples: {np.sum(test_labels == 0)}")
    print(f"  Anomaly samples: {np.sum(test_labels == 1)}")
    
    # Model oluştur
    model = PaDiMModel(
        backbone='resnet18',
        reduce_dimensions=True,
        target_dim=100
    )
    
    # Modeli eğit
    print("\n" + "-"*60)
    print("MODEL EĞİTİMİ BAŞLIYOR...")
    print("-"*60)
    
    model.fit(train_images)
    
    # Test seti üzerinde tahmin yap
    print("\n" + "-"*60)
    print("TEST SONUÇLARI HESAPLANIYOR...")
    print("-"*60)
    
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
    
    return results


def run_patchcore(dataset_path, target_size=(256, 256), memory_bank_size=250):
    """
    PatchCore modelini çalıştırır
    
    Args:
        dataset_path: Dataset dizini
        target_size: Görüntü boyutu
        memory_bank_size: Memory bank boyutu
    
    Returns:
        results: Sonuç sözlüğü
    """
    print("\n" + "="*60)
    print("PATCHCORE MODEL ÇALIŞTIRILIYOR")
    print("="*60)
    
    # Dataset yükle
    train_images, test_images, test_labels = load_dataset_patchcore(dataset_path, target_size)
    
    print(f"\nDataset Bilgileri:")
    print(f"  Training set: {train_images.shape}")
    print(f"  Test set: {test_images.shape}")
    print(f"  Normal samples: {np.sum(test_labels == 0)}")
    print(f"  Anomaly samples: {np.sum(test_labels == 1)}")
    
    # Model oluştur
    model = PatchCoreModel(
        backbone='resnet18',
        memory_bank_size=memory_bank_size,
        reduce_dimensions=True,
        target_dim=100
    )
    
    # Modeli eğit
    print("\n" + "-"*60)
    print("MODEL EĞİTİMİ BAŞLIYOR...")
    print("-"*60)
    
    model.fit(train_images)
    
    # Test seti üzerinde tahmin yap
    print("\n" + "-"*60)
    print("TEST SONUÇLARI HESAPLANIYOR...")
    print("-"*60)
    
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
    
    return results


def start_process(dataset_name, model_name, **kwargs):
    """
    Ana işlem fonksiyonu
    
    Args:
        dataset_name: Kullanılacak dataset adı
        model_name: Kullanılacak model adı
        **kwargs: Ek parametreler
    
    Returns:
        results: Sonuç sözlüğü
    """
    # Dataset kontrolü
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"⚠️  Uyarı: '{dataset_name}' listede yok ama deneniyor...")
    
    # Model kontrolü
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Geçersiz model: {model_name}. Mevcut modeller: {list(AVAILABLE_MODELS.keys())}")
    
    # Dataset yolunu al
    dataset_path = get_dataset_path(dataset_name)
    
    print("\n" + "="*60)
    print(f"BAŞLANGIÇ BİLGİLERİ")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Model: {AVAILABLE_MODELS[model_name]['name']}")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Modeli çalıştır
    results = None
    
    if model_name == 'autoencoder':
        results = run_autoencoder(
            dataset_path, 
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 8)
        )
    elif model_name == 'padim':
        results = run_padim(dataset_path)
    elif model_name == 'patchcore':
        results = run_patchcore(
            dataset_path,
            memory_bank_size=kwargs.get('memory_bank_size', 100)
        )
    
    # Sonuçları kaydet
    if results:
        output_dir = f"results/{dataset_name}"
        save_model_results(AVAILABLE_MODELS[model_name]['name'], results, output_dir)
        
        # Özet yazdır
        print("\n" + "="*60)
        print(f"SONUÇ ÖZETİ - {AVAILABLE_MODELS[model_name]['name']}")
        print("="*60)
        print(f"Dataset: {dataset_name}")
        print(f"AUC Score:    {results['auc_score']:.4f}")
        print(f"F1 Score:     {results['f1_score']:.4f}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        print(f"Accuracy:     {results['accuracy']:.4f}")
        print("="*60)
    
    return results


def main():
    """Ana fonksiyon - command line argümanlarını parse eder"""
    parser = argparse.ArgumentParser(
        description='Ahşap anomali tespiti için model eğitimi ve testi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python start_process.py --dataset wood_otsu_clahe --model autoencoder
  python start_process.py --dataset wood_otsu_clahe_gamma --model padim
  python start_process.py --dataset wood_otsu_sobel_clahe --model patchcore --memory-bank-size 1000
  python start_process.py --dataset wood_otsu_clahe --model autoencoder --epochs 100 --batch-size 4
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help=f'Kullanılacak dataset adı. Önerilen: {", ".join(AVAILABLE_DATASETS)}'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help='Kullanılacak model adı'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Autoencoder için epoch sayısı (varsayılan: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Autoencoder için batch size (varsayılan: 8)'
    )
    
    parser.add_argument(
        '--memory-bank-size',
        type=int,
        default=100,
        help='PatchCore için memory bank boyutu (varsayılan: 100)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='Mevcut dataset\'leri listele'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Mevcut modelleri listele'
    )
    
    args = parser.parse_args()
    
    # Liste komutları
    if args.list_datasets:
        print("\n=== MEVCUT DATASET'LER ===")
        for ds in AVAILABLE_DATASETS:
            path = os.path.join("dataset", ds)
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {ds}")
        return
    
    if args.list_models:
        print("\n=== MEVCUT MODELLER ===")
        for key, info in AVAILABLE_MODELS.items():
            print(f"  • {key}: {info['name']}")
        return
    
    # Ana işlemi başlat
    try:
        results = start_process(
            dataset_name=args.dataset,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            memory_bank_size=args.memory_bank_size
        )
        
        print("\n✅ İşlem başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


