# 🎉 Veri Önişleme Scripti Oluşturuldu!

## 📦 Oluşturulan Dosyalar

### 1. **data_preprocessing.py** (Ana Script)
Kapsamlı veri önişleme pipeline'ı. Şu özellikleri içerir:

#### Önişleme Teknikleri:
- ✅ **Otomatik Kenar Kırpma**: Gereksiz sınır bölgelerini temizler
- ✅ **Gürültü Azaltma**: 4 farklı yöntem (Bilateral, Gaussian, Median, NLM)
- ✅ **Kontrast İyileştirme**: CLAHE, Histogram Equalization
- ✅ **Keskinleştirme**: Detayları vurgula
- ✅ **Normalizasyon**: MinMax, Z-Score, Robust
- ✅ **Morfolojik İşlemler**: Opening, Closing, Gradient
- ✅ **Veri Artırma**: 6 farklı augmentation tekniği

#### Konfigürasyon Profilleri:
1. **Minimal**: Hızlı, temel işlemler
2. **Balanced**: Dengeli (ÖNERILEN) ⭐
3. **Aggressive**: Kapsamlı, detaylı işlemler

### 2. **demo_preprocessing.py** (Hızlı Test)
Tek bir görüntü üzerinde önişleme adımlarını görselleştirir.

### 3. **PREPROCESSING_GUIDE.md** (Detaylı Dokümantasyon)
Tüm tekniklerin açıklaması, kullanım örnekleri ve SSS.

### 4. **requirements.txt** (Güncellendi)
Gerekli paketler eklendi: `scikit-image`, `tqdm`

## 🚀 Hızlı Başlangıç

### Adım 1: Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 2: Hızlı Demo (Opsiyonel)
```bash
python demo_preprocessing.py
```
Bu komut bir örnek görüntü üzerinde adım adım önişleme gösterir.

### Adım 3: Tam Önişleme
```bash
python data_preprocessing.py
```

Script interaktif olarak çalışır:
1. Örnek görüntü üzerinde önizleme gösterilir
2. 3 farklı konfigürasyon karşılaştırılır
3. Tercih ettiğiniz profili seçersiniz
4. Augmentation isteyip istemediğinizi belirtirsiniz
5. Tüm dataset işlenir ve kaydedilir

### Adım 4: Model Scriptlerini Güncelleyin
Önişlenmiş verileri kullanmak için model scriptlerinizde:

```python
# Eski
DATASET_ROOT = "dataset/wood"

# Yeni
DATASET_ROOT = "dataset/wood_preprocessed"
```

## 📊 Beklenen Sonuçlar

### Öncesi (Orijinal)
- Dosya boyutu: ~2-5 MB/görüntü
- Gürültülü kenarlar var
- Düşük kontrast
- Veri sayısı: 90 train + 10 test good + 36 test defect

### Sonrası (Önişlenmiş)
- Dosya boyutu: ~100-500 KB/görüntü (daha küçük!)
- Temiz kenarlar
- Geliştirilmiş kontrast
- Normalize edilmiş
- Veri sayısı: 
  - Augmentation YOK ise: Aynı
  - Augmentation VAR ise: ~270-450 train (3-5x artış)

### Kayıt Konumu
```
dataset/wood_preprocessed/
├── train/good/                    # Önişlenmiş + augmented
├── test/good/                     # Önişlenmiş
├── test/defect/                   # Önişlenmiş
└── preprocessing_config.json      # Kullanılan ayarlar
```

## 💡 Önerilen Kullanım

### Genel Kullanım (Çoğu Proje İçin):
```
Konfigürasyon: Balanced
Augmentation: Evet (eğer training data < 100)
Target Size: 256x256
```

### Hızlı Test/Prototype:
```
Konfigürasyon: Minimal
Augmentation: Hayır
Target Size: 128x128
```

### Maksimum Kalite:
```
Konfigürasyon: Aggressive
Augmentation: Evet
Target Size: 512x512
```

## 🎯 Neden Bu Önişleme Gerekli?

### 1. Model Performansını Artırır
- Gürültü azaltma → Daha temiz özellikler
- Kontrast iyileştirme → Defektler daha belirgin
- Normalizasyon → Stabil eğitim

### 2. Eğitim Süresini Azaltır
- Küçük dosya boyutu → Hızlı yükleme
- Standart boyut → Batch işleme kolaylığı

### 3. Generalizasyonu İyileştirir
- Augmentation → Daha fazla çeşitlilik
- Normalizasyon → Overfitting azalır

### 4. Gereksiz Bilgiyi Temizler
- Kenar kırpma → Sadece ilgili bölge
- Morfolojik işlemler → Küçük gürültüler kaybolur

## 📚 Ek Kaynaklar

- **Detaylı Dokümantasyon**: `PREPROCESSING_GUIDE.md`
- **Teknik Referanslar**: PREPROCESSING_GUIDE.md → Referanslar bölümü
- **Troubleshooting**: PREPROCESSING_GUIDE.md → Troubleshooting bölümü

## 🔍 Görselleştirme

Script çalıştırıldığında otomatik olarak şunları gösterir:

1. **Adım Adım İşleme**:
   - Orijinal görüntü
   - Her önişleme adımının etkisi
   - Final sonuç

2. **Konfigürasyon Karşılaştırma**:
   - Minimal vs Balanced vs Aggressive
   - Yan yana görsel karşılaştırma

3. **İstatistikler**:
   ```
   Dataset İşleme Özeti:
   ✓ Train Good: 90 görüntü
   ✓ Train Good (Augmented): 270 ek görüntü
   ✓ Toplam Train: 360 görüntü
   ✓ Test Good: 10 görüntü
   ✓ Test Defect: 36 görüntü
   ```

## ⚙️ İleri Seviye Kullanım

### Özel Konfigürasyon

Python scriptinde doğrudan kullanım:

```python
from data_preprocessing import ImagePreprocessor, process_and_save_dataset

# Özel konfigürasyon
custom_config = {
    'auto_crop': True,
    'denoise': 'bilateral',
    'contrast': 'clahe',
    'sharpen': True,
    'sharpen_strength': 0.7,
    'normalize': 'robust',
    'morphology': 'opening',
    'resize': True
}

# Özel boyut
preprocessor = ImagePreprocessor(target_size=(512, 512))

# İşle
process_and_save_dataset(preprocessor, custom_config, apply_augmentation=True)
```

### Tek Görüntü İşleme

```python
from data_preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(256, 256))
image = preprocessor.load_image("path/to/image.bmp")
processed = preprocessor.preprocess_pipeline(image)

import cv2
cv2.imwrite("output.png", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
```

## 🐛 Yaygın Hatalar ve Çözümleri

### Hata: "ModuleNotFoundError: No module named 'skimage'"
**Çözüm**:
```bash
pip install scikit-image
```

### Hata: "FileNotFoundError: dataset/wood/train/good"
**Çözüm**: Dataset klasör yapısını kontrol edin:
```
dataset/
└── wood/
    ├── train/
    │   └── good/  ← BMP dosyaları burada
    └── test/
        ├── good/
        └── defect/
```

### Hata: Görüntüler yüklenemiyor
**Çözüm**: BMP formatında olmadığı için olabilir. Script'te `.bmp` yerine:
```python
[f for f in os.listdir(path) if f.endswith(('.bmp', '.png', '.jpg'))]
```

## 📈 Performans Beklentileri

### İşlem Süresi (90 train + 46 test görüntü için):
- **Minimal**: ~10-15 saniye
- **Balanced**: ~30-45 saniye ⭐
- **Aggressive**: ~2-3 dakika

### Bellek Kullanımı:
- RAM: ~500 MB
- Disk (çıktı): ~50-200 MB

## ✅ Checklist

Önişleme yapmadan önce:

- [ ] Dataset klasörü doğru konumda
- [ ] Gerekli paketler yüklendi (`pip install -r requirements.txt`)
- [ ] Yeterli disk alanı var (~200 MB)
- [ ] Orijinal veriler yedeklendi (önerilir)

Önişleme sonrası:

- [ ] `dataset/wood_preprocessed/` klasörü oluştu
- [ ] Görüntüler başarıyla kaydedildi
- [ ] `preprocessing_config.json` oluşturuldu
- [ ] Model scriptlerinde `DATASET_ROOT` güncellendi

## 🎓 Sonuç

Artık elinizde profesyonel bir veri önişleme pipeline'ı var! Bu script:

✅ Literatürde kanıtlanmış teknikleri kullanır
✅ Esnek ve özelleştirilebilir
✅ İnteraktif ve kullanımı kolay
✅ Görselleştirme ile sonuçları gösterir
✅ Kapsamlı dokümantasyon ile desteklenir

**İyi çalışmalar!** 🚀

---

**Sorular?** 
- Detaylı bilgi için: `PREPROCESSING_GUIDE.md`
- Hata durumunda: PREPROCESSING_GUIDE.md → Troubleshooting
