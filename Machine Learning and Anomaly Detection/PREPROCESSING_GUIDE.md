# Veri Önişleme Rehberi

## 📋 İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Önişleme Teknikleri](#önişleme-teknikleri)
3. [Kullanım](#kullanım)
4. [Konfigürasyon Seçenekleri](#konfigürasyon-seçenekleri)
5. [Performans İpuçları](#performans-ipuçları)

## 🎯 Genel Bakış

`data_preprocessing.py` scripti, ahşap yüzey anomali tespiti için optimize edilmiş kapsamlı bir görüntü önişleme pipeline'ı sağlar. Bu script literatürde yaygın kullanılan teknikleri içerir ve model performansını artırmak için tasarlanmıştır.

### Neden Veri Önişleme?

- **Gürültü Azaltma**: Görüntülerdeki istenmeyen pikselleri temizler
- **Normalizasyon**: Model eğitimini stabilize eder
- **Kontrast İyileştirme**: Defekt bölgelerini daha belirgin hale getirir
- **Gereksiz Bölge Temizleme**: Kenar bölgelerindeki anlamsız pikselleri kaldırır
- **Veri Artırma**: Az sayıda eğitim verisini çoğaltır

## 🔧 Önişleme Teknikleri

### 1. Otomatik Kenar Kırpma (Auto Crop)
**Amaç**: Görüntünün kenarlarındaki gereksiz boş/uniform bölgeleri kaldırır.

**Nasıl Çalışır**:
- Canny edge detection ile kenarlar tespit edilir
- İçerik içeren minimum bounding box hesaplanır
- Gereksiz kenarlar otomatik olarak kırpılır

**Ne Zaman Kullanılır**: Ahşap görüntülerinde sıkça görülen kenar gürültüsü için önerilir.

### 2. Gürültü Azaltma (Denoising)

#### Bilateral Filter (Önerilen)
- **Özellik**: Kenarları koruyarak gürültü azaltır
- **Avantaj**: Detayları korur, defekt bölgelerini bulanıklaştırmaz
- **Hız**: Orta

#### Gaussian Blur
- **Özellik**: Uniform bulanıklık
- **Avantaj**: Hızlı işlem
- **Dezavantaj**: Kenarları da bulanıklaştırır

#### Median Filter
- **Özellik**: Salt-and-pepper gürültüsü için ideal
- **Avantaj**: Outlier'lara karşı dayanıklı
- **Hız**: Orta-Yavaş

#### Non-Local Means (NLM)
- **Özellik**: En gelişmiş gürültü azaltma
- **Avantaj**: En iyi sonuç
- **Dezavantaj**: Çok yavaş

### 3. Kontrast İyileştirme

#### CLAHE (Contrast Limited Adaptive Histogram Equalization) - Önerilen
- Yerel kontrastı iyileştirir
- Aşırı kontrast artışını önler
- Defekt bölgelerini vurgular

#### Global Histogram Equalization
- Tüm görüntüye uniform uygulama
- Basit ve hızlı

#### Adaptive Histogram Equalization
- Scikit-image tabanlı
- Daha yumuşak sonuçlar

### 4. Keskinleştirme (Sharpening)
- Görüntü detaylarını vurgular
- Defekt kenarlarını belirginleştirir
- Ayarlanabilir strength parametresi (0.5-1.0 önerilir)

### 5. Normalizasyon

#### Min-Max Normalization (Önerilen)
```
normalized = (pixel - min) / (max - min)
```
- [0, 1] aralığına getirir
- En yaygın kullanılan yöntem

#### Z-Score Normalization
```
normalized = (pixel - mean) / std
```
- Mean=0, Std=1 dağılımı
- İstatistiksel normalleştirme

#### Robust Normalization
```
normalized = (pixel - p2) / (p98 - p2)
```
- Outlier'lara karşı dayanıklı
- Aykırı değerleri görmezden gelir

### 6. Morfolojik İşlemler

#### Opening
- Küçük gürültüleri kaldırır
- Önce erosion, sonra dilation

#### Closing
- Küçük delikleri kapatır
- Önce dilation, sonra erosion

#### Gradient
- Kenarları vurgular
- Dilation - erosion farkı

### 7. Veri Artırma (Data Augmentation)

**Training data için kullanılır, test data için uygulanmaz!**

Desteklenen işlemler:
- ✅ Yatay çevirme (Horizontal flip)
- ✅ Dikey çevirme (Vertical flip)
- ✅ 90°, 180°, 270° döndürme
- ✅ Parlaklık ayarlama
- ✅ Gaussian gürültü ekleme
- ✅ Rastgele kırpma

## 💻 Kullanım

### Temel Kullanım

```bash
python data_preprocessing.py
```

Script interaktif olarak çalışır ve aşağıdaki adımları izler:

1. **Önizleme**: Örnek görüntü üzerinde adım adım önişleme gösterilir
2. **Karşılaştırma**: Farklı konfigürasyonlar karşılaştırılır
3. **Konfigürasyon Seçimi**: İstediğiniz profili seçersiniz
4. **Augmentation Seçimi**: Training data için augmentation isteyip istemediğinizi belirtirsiniz
5. **İşleme**: Tüm dataset işlenir ve kaydedilir

### Hızlı Demo

```bash
python demo_preprocessing.py
```

Sadece bir örnek görüntü üzerinde önişleme adımlarını görselleştirir.

## ⚙️ Konfigürasyon Seçenekleri

### 1. Minimal (Hızlı)
```python
{
    'auto_crop': True,
    'denoise': None,
    'contrast': None,
    'sharpen': False,
    'normalize': 'minmax',
    'resize': True
}
```
- **Kullanım**: İlk testler, hızlı denemeler
- **İşlem Süresi**: En hızlı
- **Kalite**: Temel

### 2. Balanced (Önerilen)
```python
{
    'auto_crop': True,
    'denoise': 'bilateral',
    'contrast': 'clahe',
    'sharpen': True,
    'sharpen_strength': 0.5,
    'normalize': 'minmax',
    'resize': True
}
```
- **Kullanım**: Genel amaçlı, production
- **İşlem Süresi**: Orta
- **Kalite**: Yüksek
- **En iyi seçim!**

### 3. Aggressive (Detaylı)
```python
{
    'auto_crop': True,
    'denoise': 'nlm',
    'contrast': 'clahe',
    'sharpen': True,
    'sharpen_strength': 1.0,
    'normalize': 'robust',
    'morphology': 'opening',
    'resize': True
}
```
- **Kullanım**: Maksimum kalite gerektiğinde
- **İşlem Süresi**: Yavaş
- **Kalite**: En yüksek

### Özel Konfigürasyon

Python kodunda kendi konfigürasyonunuzu oluşturabilirsiniz:

```python
from data_preprocessing import ImagePreprocessor, process_and_save_dataset

custom_config = {
    'auto_crop': True,
    'denoise': 'bilateral',
    'contrast': 'clahe',
    'sharpen': True,
    'sharpen_strength': 0.7,
    'normalize': 'robust',
    'morphology': 'closing',
    'resize': True
}

preprocessor = ImagePreprocessor(target_size=(512, 512))
process_and_save_dataset(preprocessor, custom_config, apply_augmentation=True)
```

## 🚀 Performans İpuçları

### İşlem Süresi Optimizasyonu

1. **Küçük Görüntü Boyutu**: 
   - 256x256 yerine 128x128 kullanın (4x hızlı)
   - Ancak model performansı düşebilir

2. **Denoising Seçimi**:
   - En Hızlı: Gaussian > Median > Bilateral > NLM (En Yavaş)
   - Balanced profile zaten optimal

3. **Augmentation**:
   - Sadece training data için kullanın
   - 3-5 augmentation yeterli (daha fazlası gereksiz)

### Bellek Kullanımı

- Büyük dataset'ler için batch işleme otomatik yapılır
- Tüm görüntüler aynı anda yüklenmez
- Tipik kullanım: ~500MB RAM

### Disk Alanı

- Orijinal: ~2-5 MB/görüntü (BMP formatı)
- Önişlenmiş: ~100-500 KB/görüntü (daha küçük boyut)
- Augmentation ile ~3-5x artış

## 📊 Sonuçlar

Önişlenmiş veriler şu klasöre kaydedilir:

```
dataset/wood_preprocessed/
├── train/
│   └── good/                  # Önişlenmiş + augmented (opsiyonel)
├── test/
│   ├── good/                  # Önişlenmiş
│   └── defect/                # Önişlenmiş
└── preprocessing_config.json  # Kullanılan konfigürasyon
```

### Model Scriptlerini Güncelleme

Önişlenmiş verileri kullanmak için model scriptlerinizde şu değişikliği yapın:

```python
# Eski
DATASET_ROOT = "dataset/wood"

# Yeni
DATASET_ROOT = "dataset/wood_preprocessed"
```

## 🔍 Görselleştirme

Script otomatik olarak şunları gösterir:

1. **Adım Adım İşleme**: Her önişleme adımının etkisi
2. **Konfigürasyon Karşılaştırma**: 3 profil yan yana
3. **İstatistikler**: İşlenen dosya sayıları

## ❓ SSS

### S: Hangi konfigürasyonu seçmeliyim?
**C**: Balanced profile çoğu durumda en iyi seçimdir. Eğer sonuçlar yeterince iyi değilse Aggressive'i deneyin.

### S: Augmentation kullanmalı mıyım?
**C**: Eğer training veriniz az ise (100'den az), kesinlikle evet. Dataset'iniz yeterince büyükse gerekmeyebilir.

### S: İşlem ne kadar sürer?
**C**: 
- Minimal: ~10-15 saniye (100 görüntü)
- Balanced: ~30-45 saniye
- Aggressive: ~2-3 dakika

### S: Orijinal verileri silebilir miyim?
**C**: Hayır! Her zaman orijinal verileri saklayın. Farklı konfigürasyonları test etmek isteyebilirsiniz.

### S: Özel bir boyut kullanabilir miyim?
**C**: Evet, kodu düzenleyerek `target_size` parametresini değiştirebilirsiniz.

## 📚 Referanslar

Kullanılan teknikler şu kaynaklara dayanır:

1. **CLAHE**: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
2. **Bilateral Filter**: Tomasi, C., & Manduchi, R. (1998). "Bilateral filtering for gray and color images"
3. **Non-Local Means**: Buades, A., et al. (2005). "A non-local algorithm for image denoising"
4. **Data Augmentation**: Shorten, C., & Khoshgoftaar, T. M. (2019). "A survey on image data augmentation"

## 🛠️ Troubleshooting

### Hata: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Hata: Dataset bulunamadı
Klasör yapısını kontrol edin:
```
dataset/wood/train/good/  → BMP dosyaları burada olmalı
```

### Hata: Görüntü yüklenemiyor
BMP formatında olduğundan emin olun. Diğer formatlar için kodu güncelleyin:
```python
# .bmp yerine .png veya .jpg
train_files = [f for f in os.listdir(TRAIN_GOOD_PATH) if f.endswith(('.bmp', '.png', '.jpg'))]
```

---

**Not**: Sorularınız için GitHub Issues kullanabilir veya doğrudan iletişime geçebilirsiniz.
