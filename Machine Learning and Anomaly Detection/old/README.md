# Ahşap Yüzey Anomali Tespiti Projesi

Bu proje, ahşap yüzeylerde meydana gelen yüzey hatalarının (defect) tespit edilmesi için denetimsiz (unsupervised) öğrenme modelleri geliştirmektedir.

## 📁 Proje Yapısı

```
Dataset/
├── wood/                          # Ana dataset klasörü
│   ├── train/
│   │   └── good/                  # Eğitim verisi (90 adet normal örnek)
│   └── test/
│       ├── good/                  # Test verisi normal örnekler (10 adet)
│       └── defect/                # Test verisi anomali örnekler (36 adet)
├── data_exploration.py            # Veri analizi ve görselleştirme
├── data_preprocessing.py          # ⭐ Gelişmiş veri önişleme (YENİ!)
├── utils.py                       # Genel yardımcı fonksiyonlar
├── model1_autoencoder.py          # Autoencoder tabanlı anomali tespiti
├── model2_padim.py                # PaDiM tabanlı anomali tespiti
├── model3_patchcore.py            # PatchCore tabanlı anomali tespiti
├── main_comparison.py             # Tüm modelleri karşılaştır
├── requirements.txt               # Python gereksinimler
├── PREPROCESSING_GUIDE.md         # Detaylı önişleme rehberi
└── README.md                      # Bu dosya
```

## 🚀 Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Dataset'i uygun klasör yapısında yerleştirin.

## 📊 Kullanım

### 1. Veri Keşfi ve Analizi
```bash
python data_exploration.py
```
Dataset hakkında detaylı istatistikler ve görselleştirmeler sağlar.

### 2. Veri Önişleme (YENİ!)
```bash
python data_preprocessing.py
```
**Özellikler:**
- 🔧 Otomatik kenar kırpma (gereksiz bölgeleri temizler)
- 🔇 Gürültü azaltma (Bilateral, Gaussian, Median, NLM)
- 📊 Kontrast iyileştirme (CLAHE, Histogram Equalization)
- 🔍 Keskinleştirme (detay vurgulama)
- 📏 Normalizasyon (MinMax, Z-Score, Robust)
- 🔄 Veri artırma (Data Augmentation) - Training için
- ⚙️ Morfolojik işlemler (Opening, Closing)

Script interaktif olarak çalışır ve farklı konfigürasyonlar sunar:
- **Minimal**: Hızlı, temel işlemler
- **Balanced**: Dengeli, önerilen konfigürasyon
- **Aggressive**: Detaylı, kapsamlı işlemler

Önişlenmiş veriler `dataset/wood_preprocessed/` klasörüne kaydedilir.

### 3. Tek Model Çalıştırma

**Autoencoder:**
```bash
python model1_autoencoder.py
```

**PaDiM:**
```bash
python model2_padim.py
```

**PatchCore:**
```bash
python model3_patchcore.py
```

### 3. Tüm Modelleri Karşılaştır
```bash
python main_comparison.py
```
Tüm modelleri çalıştırır ve performanslarını karşılaştırır.

## 🧠 Modeller

### 1. Autoencoder
- **Yaklaşım:** Derin öğrenme tabanlı reconstruction error
- **Özellikler:** CNN encoder-decoder mimarisi
- **Avantajlar:** Güçlü feature learning, end-to-end eğitim
- **Dezavantajlar:** Eğitim süresi, GPU gereksinimi

### 2. PaDiM (Patch Distribution Modeling)
- **Yaklaşım:** Feature-based Mahalanobis distance
- **Özellikler:** Pre-trained ResNet features, Gaussian dağılım modeli
- **Avantajlar:** Hızlı inference, etkili anomali lokalizasyonu
- **Dezavantajlar:** Memory kullanımı, feature dimensionality

### 3. PatchCore
- **Yaklaşım:** Memory bank with coreset sampling
- **Özellikler:** Multi-scale features, nearest neighbor search
- **Avantajlar:** State-of-the-art performans, robust
- **Dezavantajlar:** Memory bank boyutu, computational cost

## 📈 Değerlendirme Metrikleri

- **AUC Score (ROC-AUC):** Genel ayırt edicilik performansı
- **F1 Score:** Precision ve Recall dengesini ölçer
- **Hata Lokalizasyonu:** Anomali haritalarının görsel doğruluğu

## 📋 Sonuçlar

Modeller çalıştırıldıktan sonra aşağıdaki dosyalar oluşturulur:

- `model_comparison_results.csv` - Detaylı karşılaştırma tablosu
- `model_comparison_chart.png` - Görsel karşılaştırma grafikleri
- `results/` klasörü - Her model için detaylı sonuçlar
- Model dosyaları (`.h5`, `.pkl`)

## 🔧 Teknik Detaylar

### Görüntü Ön İşleme
- Boyut: 4096x1000 → 256x256 (resize)
- Normalizasyon: 0-1 aralığı
- Format: RGB

### Eğitim Parametreleri
- **Autoencoder:** 20-30 epoch, batch_size=4, Adam optimizer
- **PaDiM:** Pre-trained ResNet18, 100 dim reduction
- **PatchCore:** Memory bank=500, ResNet18 backbone

### Donanım Gereksinimleri
- **CPU:** Minimum 4 core (tüm modeller için)
- **RAM:** Minimum 8GB (16GB önerilen)
- **GPU:** Opsiyonel (Autoencoder için hızlandırır)

## 📚 Literatür

Projede kullanılan temel yaklaşımlar:

1. **Autoencoder:** Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders
2. **PaDiM:** Defard, T., et al. (2021). PaDiM: a Patch Distribution Modeling Framework
3. **PatchCore:** Roth, K., et al. (2022). Towards Total Recall in Industrial Anomaly Detection

## 🎯 Kullanım Alanları

- Endüstriyel kalite kontrol
- Ahşap üretim hattı otomasyonu
- Yüzey defekt tespiti
- Görsel muayene sistemleri

## ⚠️ Önemli Notlar

1. İlk çalıştırmada pre-trained modeller indirilir (internet gerekli)
2. Modeller GPU kullanımı için CUDA kurulumu gerektirebilir
3. Büyük dataset'ler için memory kullanımına dikkat edin
4. Sonuçlar sisteme bağlı olarak değişebilir (reproducibility için seed ayarlanmıştır)

## 📞 Destek

Projede karşılaşılan sorunlar için:
1. Requirements'ın doğru kurulduğundan emin olun
2. Dataset yapısının doğru olduğunu kontrol edin
3. Memory/GPU sorunları için batch size'ı azaltın
