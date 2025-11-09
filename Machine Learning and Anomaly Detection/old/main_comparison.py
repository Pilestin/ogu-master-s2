"""
Ana script - Tüm anomali tespit modellerini çalıştır ve karşılaştır
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Model scriptlerini import et
from model1_autoencoder import ConvolutionalAutoencoder, load_dataset as load_dataset_ae
from model2_padim import PaDiMModel, load_dataset as load_dataset_padim
from model3_patchcore import PatchCoreModel, load_dataset as load_dataset_patchcore
from utils import calculate_metrics, plot_metrics, save_model_results

def run_model_comparison():
    """Tüm modelleri çalıştır ve karşılaştır"""
    print("="*60)
    print("AHŞAP ANOMALİ TESPİTİ - MODEL KARŞILAŞTIRMASI")
    print("="*60)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    data_dir = "wood"
    target_size = (256, 256)
    
    # Sonuçları saklamak için
    results_summary = {
        'Model': [],
        'AUC Score': [],
        'F1 Score': [],
        'Eğitim Süresi (dk)': [],
        'Inference Süresi (sn)': [],
        'Açıklama': []
    }
    
    # 1. AUTOENCODER MODELİ
    print("\n" + "="*50)
    print("1. AUTOENCODER MODELİ ÇALIŞIYOR...")
    print("="*50)
    
    try:
        start_time = time.time()
        
        # Dataset yükle
        train_images, test_images, test_labels = load_dataset_ae(data_dir, target_size)
        
        # Model oluştur ve eğit
        autoencoder = ConvolutionalAutoencoder(
            input_shape=(target_size[0], target_size[1], 3),
            latent_dim=128
        )
        autoencoder.build_model()
        
        # Eğitim
        train_start = time.time()
        autoencoder.train(
            train_images,
            validation_split=0.2,
            epochs=20,  # Hızlı test için
            batch_size=4
        )
        train_time = (time.time() - train_start) / 60  # dakika
        
        # Test
        inference_start = time.time()
        anomaly_scores, anomaly_maps, _ = autoencoder.predict_anomaly_scores(test_images)
        inference_time = time.time() - inference_start  # saniye
        
        # Skorları normalize et
        anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Metrikleri hesapla
        ae_results = calculate_metrics(test_labels, anomaly_scores_norm, threshold=0.5)
        
        # Sonuçları kaydet
        results_summary['Model'].append('Autoencoder')
        results_summary['AUC Score'].append(ae_results['auc_score'])
        results_summary['F1 Score'].append(ae_results['f1_score'])
        results_summary['Eğitim Süresi (dk)'].append(train_time)
        results_summary['Inference Süresi (sn)'].append(inference_time)
        results_summary['Açıklama'].append('CNN tabanlı reconstruction error')
        
        # Model kaydet
        autoencoder.save_model("autoencoder_final.h5")
        save_model_results("Autoencoder", ae_results)
        
        print(f"✓ Autoencoder tamamlandı - AUC: {ae_results['auc_score']:.3f}, F1: {ae_results['f1_score']:.3f}")
        
    except Exception as e:
        print(f"✗ Autoencoder hatası: {str(e)}")
        results_summary['Model'].append('Autoencoder')
        results_summary['AUC Score'].append(0.0)
        results_summary['F1 Score'].append(0.0)
        results_summary['Eğitim Süresi (dk)'].append(0.0)
        results_summary['Inference Süresi (sn)'].append(0.0)
        results_summary['Açıklama'].append(f'Hata: {str(e)[:50]}...')
    
    # 2. PADIM MODELİ
    print("\n" + "="*50)
    print("2. PADIM MODELİ ÇALIŞIYOR...")
    print("="*50)
    
    try:
        start_time = time.time()
        
        # Dataset yükle
        train_images, test_images, test_labels = load_dataset_padim(data_dir, target_size)
        
        # Model oluştur ve eğit
        train_start = time.time()
        padim_model = PaDiMModel(
            backbone='resnet18',
            reduce_dimensions=True,
            target_dim=100
        )
        padim_model.fit(train_images)
        train_time = (time.time() - train_start) / 60  # dakika
        
        # Test
        inference_start = time.time()
        anomaly_scores, anomaly_maps = padim_model.predict(test_images)
        inference_time = time.time() - inference_start  # saniye
        
        # Skorları normalize et
        anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Metrikleri hesapla
        padim_results = calculate_metrics(test_labels, anomaly_scores_norm, threshold=0.5)
        
        # Sonuçları kaydet
        results_summary['Model'].append('PaDiM')
        results_summary['AUC Score'].append(padim_results['auc_score'])
        results_summary['F1 Score'].append(padim_results['f1_score'])
        results_summary['Eğitim Süresi (dk)'].append(train_time)
        results_summary['Inference Süresi (sn)'].append(inference_time)
        results_summary['Açıklama'].append('Feature-based Mahalanobis distance')
        
        save_model_results("PaDiM", padim_results)
        
        print(f"✓ PaDiM tamamlandı - AUC: {padim_results['auc_score']:.3f}, F1: {padim_results['f1_score']:.3f}")
        
    except Exception as e:
        print(f"✗ PaDiM hatası: {str(e)}")
        results_summary['Model'].append('PaDiM')
        results_summary['AUC Score'].append(0.0)
        results_summary['F1 Score'].append(0.0)
        results_summary['Eğitim Süresi (dk)'].append(0.0)
        results_summary['Inference Süresi (sn)'].append(0.0)
        results_summary['Açıklama'].append(f'Hata: {str(e)[:50]}...')
    
    # 3. PATCHCORE MODELİ
    print("\n" + "="*50)
    print("3. PATCHCORE MODELİ ÇALIŞIYOR...")
    print("="*50)
    
    try:
        start_time = time.time()
        
        # Dataset yükle
        train_images, test_images, test_labels = load_dataset_patchcore(data_dir, target_size)
        
        # Model oluştur ve eğit
        train_start = time.time()
        patchcore_model = PatchCoreModel(
            backbone='resnet18',
            memory_bank_size=500,
            reduce_dimensions=True,
            target_dim=100
        )
        patchcore_model.fit(train_images)
        train_time = (time.time() - train_start) / 60  # dakika
        
        # Test
        inference_start = time.time()
        anomaly_scores, anomaly_maps = patchcore_model.predict(test_images)
        inference_time = time.time() - inference_start  # saniye
        
        # Skorları normalize et
        anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Metrikleri hesapla
        patchcore_results = calculate_metrics(test_labels, anomaly_scores_norm, threshold=0.5)
        
        # Sonuçları kaydet
        results_summary['Model'].append('PatchCore')
        results_summary['AUC Score'].append(patchcore_results['auc_score'])
        results_summary['F1 Score'].append(patchcore_results['f1_score'])
        results_summary['Eğitim Süresi (dk)'].append(train_time)
        results_summary['Inference Süresi (sn)'].append(inference_time)
        results_summary['Açıklama'].append('Memory bank with coreset sampling')
        
        # Model kaydet
        patchcore_model.save_model("patchcore_final.pkl")
        save_model_results("PatchCore", patchcore_results)
        
        print(f"✓ PatchCore tamamlandı - AUC: {patchcore_results['auc_score']:.3f}, F1: {patchcore_results['f1_score']:.3f}")
        
    except Exception as e:
        print(f"✗ PatchCore hatası: {str(e)}")
        results_summary['Model'].append('PatchCore')
        results_summary['AUC Score'].append(0.0)
        results_summary['F1 Score'].append(0.0)
        results_summary['Eğitim Süresi (dk)'].append(0.0)
        results_summary['Inference Süresi (sn)'].append(0.0)
        results_summary['Açıklama'].append(f'Hata: {str(e)[:50]}...')
    
    # SONUÇLARI KARŞILAŞTIR
    print("\n" + "="*60)
    print("MODEL KARŞILAŞTIRMA SONUÇLARI")
    print("="*60)
    
    # DataFrame oluştur
    df_results = pd.DataFrame(results_summary)
    print("\nDetaylı Sonuçlar:")
    print("-" * 80)
    print(df_results.to_string(index=False))
    
    # En iyi modeli bul
    best_auc_idx = df_results['AUC Score'].idxmax()
    best_f1_idx = df_results['F1 Score'].idxmax()
    
    print(f"\n📊 En yüksek AUC Score: {df_results.loc[best_auc_idx, 'Model']} ({df_results.loc[best_auc_idx, 'AUC Score']:.4f})")
    print(f"📊 En yüksek F1 Score: {df_results.loc[best_f1_idx, 'Model']} ({df_results.loc[best_f1_idx, 'F1 Score']:.4f})")
    
    # Görselleştir
    plot_comparison_results(df_results)
    
    # Sonuçları CSV'ye kaydet
    df_results.to_csv("model_comparison_results.csv", index=False)
    print(f"\n💾 Sonuçlar kaydedildi: model_comparison_results.csv")
    
    print(f"\n⏱️ Toplam süre: {time.time() - start_time:.1f} saniye")
    print(f"🏁 Tamamlanma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def plot_comparison_results(df):
    """Model karşılaştırma sonuçlarını görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Karşılaştırma Sonuçları', fontsize=16)
    
    # AUC Score karşılaştırması
    axes[0, 0].bar(df['Model'], df['AUC Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('AUC Score Karşılaştırması')
    axes[0, 0].set_ylabel('AUC Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(df['AUC Score']):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # F1 Score karşılaştırması
    axes[0, 1].bar(df['Model'], df['F1 Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_title('F1 Score Karşılaştırması')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(df['F1 Score']):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Eğitim süresi karşılaştırması
    axes[1, 0].bar(df['Model'], df['Eğitim Süresi (dk)'], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    axes[1, 0].set_title('Eğitim Süresi (dakika)')
    axes[1, 0].set_ylabel('Süre (dk)')
    for i, v in enumerate(df['Eğitim Süresi (dk)']):
        axes[1, 0].text(i, v + max(df['Eğitim Süresi (dk)']) * 0.02, f'{v:.1f}', ha='center', va='bottom')
    
    # Inference süresi karşılaştırması
    axes[1, 1].bar(df['Model'], df['Inference Süresi (sn)'], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    axes[1, 1].set_title('Inference Süresi (saniye)')
    axes[1, 1].set_ylabel('Süre (sn)')
    for i, v in enumerate(df['Inference Süresi (sn)']):
        axes[1, 1].text(i, v + max(df['Inference Süresi (sn)']) * 0.02, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_model_comparison()
