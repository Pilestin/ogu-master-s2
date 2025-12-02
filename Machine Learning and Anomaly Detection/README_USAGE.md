# Ahşap Anomali Tespiti - Kullanım Kılavuzu

Bu proje, ahşap görüntülerinde kusurları tespit etmek için 3 farklı derin öğrenme modelini kullanır.

## 📁 Proje Yapısı

```
Machine Learning and Anomaly Detection/
├── dataset/                          # Veri setleri
│   ├── wood/                         # Orijinal veri seti
│   ├── wood_otsu_clahe/             # Önişlenmiş veri setleri
│   ├── wood_otsu_clahe_gamma/
│   ├── wood_otsu_sobel_clahe/
│   └── ...
├── models/                           # Model dosyaları
│   ├── model1_autoencoder.py        # Convolutional Autoencoder
│   ├── model2_padim.py              # PaDiM Model
│   └── model3_patchcore.py          # PatchCore Model
├── results/                          # Sonuç dosyaları
├── utils.py                          # Yardımcı fonksiyonlar
├── start_process.py                  # Ana çalıştırma scripti
└── main_preprocess.py               # Önişleme scripti
```

## 🚀 Kullanım

### 1. Mevcut Dataset'leri Listele

```bash
python start_process.py --list-datasets
```

### 2. Mevcut Modelleri Listele

```bash
python start_process.py --list-models
```

### 3. Model Çalıştırma Örnekleri

#### Autoencoder ile Test
```bash
python start_process.py --dataset wood_otsu_clahe --model autoencoder
```

Ek parametreler:
```bash
python start_process.py --dataset wood_otsu_clahe --model autoencoder --epochs 100 --batch-size 4
```

#### PaDiM ile Test
```bash
python start_process.py --dataset wood_otsu_clahe_gamma --model padim
```

#### PatchCore ile Test
```bash
python start_process.py --dataset wood_otsu_sobel_clahe --model patchcore
```

Ek parametreler:
```bash
python start_process.py --dataset wood_otsu_sobel_clahe --model patchcore --memory-bank-size 1000
```

## 📊 Modeller

### 1. Convolutional Autoencoder
- **Tür**: Unsupervised Learning
- **Yaklaşım**: Reconstruction error tabanlı
- **Parametreler**:
  - `--epochs`: Eğitim epoch sayısı (varsayılan: 50)
  - `--batch-size`: Batch boyutu (varsayılan: 8)

### 2. PaDiM (Patch Distribution Modeling)
- **Tür**: Feature-based Anomaly Detection
- **Yaklaşım**: Pre-trained ResNet features + Mahalanobis distance
- **Avantaj**: Hızlı eğitim, iyi performans

### 3. PatchCore
- **Tür**: Memory Bank yaklaşımı
- **Yaklaşım**: Coreset sampling + Nearest neighbor
- **Parametreler**:
  - `--memory-bank-size`: Memory bank boyutu (varsayılan: 500)

## 📈 Sonuçlar

Her model çalıştırıldığında şu çıktılar üretilir:

1. **Performans Metrikleri**:
   - AUC Score
   - F1 Score
   - Precision
   - Recall
   - Accuracy

2. **Görselleştirmeler**:
   - Confusion Matrix
   - ROC Curve
   - Eğitim geçmişi (Autoencoder için)
   - Anomaly Maps

3. **Kayıtlı Dosyalar**:
   - `results/{dataset_name}/{model_name}_results.json`

## 🔧 Gereksinimler

```bash
pip install tensorflow
pip install torch torchvision
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install numpy
pip install scipy
```

## 💡 İpuçları

1. **Dataset Seçimi**: Önişlenmiş dataset'ler genellikle daha iyi sonuç verir
   - `wood_otsu_clahe`: Otsu kırpma + CLAHE kontrast artırma
   - `wood_otsu_clahe_gamma`: + Gamma düzeltmesi
   - `wood_otsu_sobel_clahe`: + Sobel kenar tespiti

2. **Model Seçimi**:
   - Hızlı test için: **PaDiM**
   - En iyi performans için: **PatchCore**
   - Görsel reconstruction için: **Autoencoder**

3. **Parametre Ayarlama**:
   - Küçük veri seti için düşük memory bank size kullanın
   - GPU varsa batch size'ı artırabilirsiniz
   - Overfitting varsa epoch sayısını azaltın

## 🎯 Örnek Çalıştırma

Tüm adımları tek seferde:

```bash
# 1. Dataset'leri kontrol et
python start_process.py --list-datasets

# 2. Autoencoder ile test et
python start_process.py --dataset wood_otsu_clahe --model autoencoder --epochs 30 --batch-size 8

# 3. PaDiM ile test et
python start_process.py --dataset wood_otsu_clahe --model padim

# 4. PatchCore ile test et
python start_process.py --dataset wood_otsu_clahe --model patchcore --memory-bank-size 500
```

## 📝 Notlar

- İlk çalıştırmada PyTorch, pretrained modelleri indirecektir
- GPU kullanımı otomatik olarak tespit edilir
- Sonuçlar `results/` klasörüne kaydedilir
- Her çalıştırma için görselleştirmeler ekranda gösterilir
