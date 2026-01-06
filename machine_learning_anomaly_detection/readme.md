# Wood Anomaly Detection V3 - Technical Report

**Report Date:** 2026-01-06  
**Dataset:** baseline_512  
**Resolution:** 512×512/256×256  
**Environment:** Local(RTX 4060) / Google Colab (GPU)

---

## Executive Summary

Bu rapor, ahşap yüzey defekt tespiti için geliştirilen **cold-start anomaly detection** sisteminin teknik detaylarını içermektedir. V3 sürümü, 4 farklı derin öğrenme modeli, kapsamlı veri ön işleme pipeline'ı ve grid search tabanlı hiperparametre optimizasyonu içermektedir.

**Temel Özellikler:**

- 4 model: AutoEncoder, PatchCore, SimpleNet, EfficientAD
- Çoklu çözünürlük desteği: 256×256 ve 512×512
- Grid search ile hiperparametre optimizasyonu
- Otomatik en iyi model seçimi ve heatmap görselleştirme

---

## 1. Problem Tanımı

### 1.1 Cold-Start Anomaly Detection

**Tanım:** Modeller yalnızca "iyi" (normal) örneklerle eğitilir ve test sırasında hiç görmediği "defektli" (anormal) örnekleri tespit etmelidir.

**Zorluklar:**

- Defekt örnekleri eğitimde kullanılamaz
- Model, normal dağılımı öğrenmeli ve sapmaları tespit etmeli
- Düşük false positive oranı kritik

### 1.2 Veri Seti Yapısı

| Split | Good (Normal)             | Defect (Anormal)           |
| ----- | ------------------------- | -------------------------- |
| Train | <!-- TRAIN_GOOD_COUNT --> | 0                          |
| Test  | <!-- TEST_GOOD_COUNT -->  | <!-- TEST_DEFECT_COUNT --> |

---

## 2. Veri Ön İşleme Pipeline

### 2.1 Veri İşleme Script'i: `data_processing_script.py`

Orijinal görüntüler üzerinde uygulanan işlemler:

#### 2.1.1 Görüntü Yükleme (Türkçe Karakter Desteği)

```python
# Windows'ta Türkçe karakterli dosya yolları için
image_array = np.fromfile(image_path, dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

#### 2.1.2 Otsu Thresholding ile Kırpma

```python
def crop_with_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    margin = 10
    return image[y-margin:y+h+margin, x-margin:x+w+margin]
```

**Amaç:** Arka planı kaldırarak sadece ahşap bölgesine odaklanmak.

#### 2.1.3 Gürültü Azaltma (Bilateral Filter)

```python
def denoise(image, method='bilateral'):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
```

**Parametreler:**

- `d=9`: Filter kernel boyutu
- `sigmaColor=75`: Renk uzayında sigma
- `sigmaSpace=75`: Koordinat uzayında sigma

**Avantaj:** Kenarları koruyarak gürültüyü azaltır.

#### 2.1.4 Kontrast Artırma (CLAHE)

```python
def enhance_contrast(image, method='clahe'):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced_lab = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
```

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**

- `clipLimit=2.0`: Kontrast sınırlama eşiği
- `tileGridSize=(8,8)`: Yerel histogram eşitleme grid boyutu

#### 2.1.5 Normalizasyon

```python
def normalize_image(image, method='minmax'):
    image_float = image.astype(np.float32)
    normalized = (image_float - image_float.min()) / (image_float.max() - image_float.min() + 1e-8)
    return (normalized * 255).astype(np.uint8)
```

#### 2.1.6 Yeniden Boyutlandırma

```python
def resize_image(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
```

**INTER_LANCZOS4:** Yüksek kaliteli 8×8 Lanczos interpolasyonu.

### 2.2 Ön İşleme Varyantları

Script çalıştırıldığında oluşturulan veri setleri:

| Varyant               | İşlemler                                           | Kullanım Amacı                |
| --------------------- | -------------------------------------------------- | ----------------------------- |
| `baseline_256`        | Otsu crop + resize(256)                            | Hızlı deneyler                |
| `baseline_512`        | Otsu crop + resize(512)                            | Yüksek çözünürlük             |
| `clahe_only_256`      | Otsu + CLAHE + resize(256)                         | Kontrast iyileştirme          |
| `clahe_only_512`      | Otsu + CLAHE + resize(512)                         | Kontrast + yüksek çözünürlük  |
| `full_preprocess_256` | Otsu + bilateral + CLAHE + normalize + resize(256) | Tam pipeline                  |
| `full_preprocess_512` | Otsu + bilateral + CLAHE + normalize + resize(512) | En kapsamlı                   |
| `grayscale_256`       | Otsu + grayscale + CLAHE + normalize + resize(256) | Tek kanal                     |
| `grayscale_512`       | Otsu + grayscale + CLAHE + normalize + resize(512) | Tek kanal + yüksek çözünürlük |

---

## 3. Veri Artırma (Data Augmentation)

### 3.1 Eğitim Sırasında Uygulanan Augmentasyonlar

```python
transforms_list = [
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

| Augmentasyon    | Parametre            | Açıklama                                |
| --------------- | -------------------- | --------------------------------------- |
| Random Rotation | ±15°                 | Rastgele döndürme                       |
| Horizontal Flip | p=0.5                | %50 olasılıkla yatay çevirme            |
| Vertical Flip   | p=0.3                | %30 olasılıkla dikey çevirme            |
| Color Jitter    | B=0.1, C=0.1, S=0.05 | Parlaklık, kontrast, doygunluk değişimi |

### 3.2 ImageNet Normalizasyonu

Tüm modeller ImageNet ön eğitimli backbone kullandığı için:

- **Mean:** [0.485, 0.456, 0.406]
- **Std:** [0.229, 0.224, 0.225]

---

## 4. Model Mimarileri

### 4.1 AutoEncoder (Reconstruction-Based)

**Kavram:** Normal örnekleri yeniden yapılandırmayı öğren; anomaliler daha yüksek reconstruction error verecektir.

#### Mimari

```
Encoder:
├── Conv2d(3→32, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)   → H/2
├── Conv2d(32→64, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)  → H/4
├── Conv2d(64→128, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → H/8
├── Conv2d(128→256, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)→ H/16
└── Conv2d(256→latent_dim, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → H/32

Decoder (Mirror):
├── ConvTranspose2d(latent_dim→256) + BatchNorm + ReLU
├── ConvTranspose2d(256→128) + BatchNorm + ReLU
├── ConvTranspose2d(128→64) + BatchNorm + ReLU
├── ConvTranspose2d(64→32) + BatchNorm + ReLU
└── ConvTranspose2d(32→3) + Sigmoid
```

#### Kayıp Fonksiyonu: MSE + SSIM

```python
loss = (1 - ssim_weight) * MSE(recon, original) + ssim_weight * SSIM_Loss(recon, original)
```

**SSIM (Structural Similarity Index):**

```
SSIM(x,y) = (2μx·μy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
SSIM_Loss = 1 - SSIM
```

#### Hiperparametreler (Grid Search)

| Parametre     | Değer Aralığı   |
| ------------- | --------------- |
| latent_dim    | [128, 256, 512] |
| ssim_weight   | [0.3, 0.5, 0.7] |
| learning_rate | [1e-3, 5e-4]    |
| epochs        | 100 (max)       |
| optimizer     | Adam            |

---

### 4.2 PatchCore (Memory Bank + K-NN)

**Kavram:** Normal patch özelliklerinden bir bellek bankası oluştur; test sırasında en yakın komşu mesafesine göre anomali skoru hesapla.

#### Mimari

```
Feature Extractor (Pretrained WideResNet50-2, Frozen):
├── Conv1 → BatchNorm → ReLU → MaxPool
├── Layer1 (256 channels)
├── Layer2 (512 channels) ─────────────────┐
└── Layer3 (1024 channels) ────────────────┤
                                            ↓
                                  Concatenate + L2 Normalize
                                            ↓
                                  1536 channels (multi-layer)
                                            ↓
                                  Sparse Random Projection → 512 dim
                                            ↓
                                  Coreset Sampling → Memory Bank
```

#### Coreset Sampling Algoritması

```python
# Greedy Farthest Point Sampling
def coreset_sample(features, k):
    indices = [random_initial_point]
    min_dists = np.full(len(features), np.inf)

    for _ in range(k-1):
        dists = np.linalg.norm(features - features[indices[-1]], axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[indices] = -1  # Exclude selected
        indices.append(np.argmax(min_dists))

    return features[indices]
```

**Amaç:** Tüm normal özellikleri en iyi temsil eden alt kümeyi seçmek.

#### Anomali Skoru Hesaplama

```python
distances, _ = nn_model.kneighbors(test_features)
patch_scores = distances.mean(axis=1)  # K komşunun ortalama mesafesi
image_score = max(patch_scores)  # En yüksek patch skoru
```

#### Hiperparametreler (Grid Search)

| Parametre        | Değer Aralığı                     |
| ---------------- | --------------------------------- |
| memory_bank_size | [1000, 2000, 3000] veya -1 (tümü) |
| k_neighbors      | [3, 5, 9]                         |
| use_multi_layer  | [True, False]                     |
| target_dim       | 512                               |
| backbone         | wide_resnet50_2                   |

---

### 4.3 SimpleNet (Feature Adaptor + Discriminator)

**Kavram:** Sentetik anomali özellikleri üreterek bir discriminator eğit.

#### Mimari

```
Feature Extractor (WideResNet50-2, Frozen):
└── Layer3 output → 1024 channels

Feature Adaptor:
└── Linear(1024→adaptor_dim, bias=False) + LayerNorm

Discriminator:
├── Linear(adaptor_dim→256) + BatchNorm + LeakyReLU(0.2) + Dropout(0.1)
├── Linear(256→128) + BatchNorm + LeakyReLU(0.2)
└── Linear(128→1) → Anomaly Score (sigmoid)
```

#### Sentetik Anomali Üretimi (Multi-Noise)

```python
def generate_anomaly(features, base_noise_std=0.015):
    stds = [base_noise_std * 0.5, base_noise_std * 1.0, base_noise_std * 2.0]
    batch_size = features.shape[0] // 3

    anomaly_parts = []
    for i, std in enumerate(stds):
        start = i * batch_size
        end = (i + 1) * batch_size if i < 2 else features.shape[0]
        noise = torch.randn_like(features[start:end]) * std
        anomaly_parts.append(features[start:end] + noise)

    return torch.cat(anomaly_parts, dim=0)
```

#### Focal Loss

```python
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()
```

**Avantaj:** Class imbalance durumunda zor örneklere daha fazla ağırlık verir.

#### Hiperparametreler (Grid Search)

| Parametre       | Değer Aralığı             |
| --------------- | ------------------------- |
| noise_std       | [0.01, 0.015, 0.02]       |
| adaptor_dim     | [256, 512]                |
| use_multi_noise | [True, False]             |
| learning_rate   | 1e-4                      |
| epochs          | 50 (max)                  |
| optimizer       | AdamW (weight_decay=1e-4) |

---

### 4.4 EfficientAD (Student-Teacher + AutoEncoder)

**Kavram:** Teacher-student knowledge distillation ile birlikte lightweight autoencoder kullanarak hem global hem de lokal anomalileri tespit et.

#### Mimari

```
Teacher Network (ResNet18, Frozen):
├── Conv1 → BatchNorm → ReLU → MaxPool
├── Layer1, Layer2, Layer3
└── Output: Teacher Features (256 channels)

Student Network (ResNet18, Trainable):
├── Same architecture as Teacher
├── + Student Head:
│   └── Conv2d(256→student_dim→256, k=1)
└── Output: Student Features

Lightweight AutoEncoder:
├── Encoder:
│   ├── Conv2d(256→autoencoder_dim, k=3, s=2)
│   └── Conv2d(autoencoder_dim→autoencoder_dim/2, k=3, s=2)
└── Decoder:
    ├── ConvTranspose2d(autoencoder_dim/2→autoencoder_dim, k=4, s=2)
    └── ConvTranspose2d(autoencoder_dim→256, k=4, s=2)
```

#### Kayıp Fonksiyonları

```python
# Student-Teacher Distillation Loss
st_loss = F.mse_loss(student_features, teacher_features.detach())

# AutoEncoder Reconstruction Loss
ae_loss = F.mse_loss(autoencoder_output, teacher_features.detach())

# Combined Loss
total_loss = st_loss + ae_loss
```

#### Anomali Skoru Hesaplama

```python
# Student-Teacher difference (global anomalies)
st_diff = ((teacher_features - student_features) ** 2).mean(dim=1)

# AutoEncoder difference (local anomalies)
ae_diff = ((teacher_features - ae_output) ** 2).mean(dim=1)

# Combined anomaly map
anomaly_map = 0.5 * st_diff + 0.5 * ae_diff
```

#### Hiperparametreler (Grid Search)

| Parametre       | Değer Aralığı   |
| --------------- | --------------- |
| student_dim     | [256, 384, 512] |
| autoencoder_dim | [256, 384, 512] |
| backbone        | resnet18        |
| learning_rate   | 1e-4            |
| epochs          | 70 (max)        |
| weight_decay    | 1e-5            |

---

## 5. Eğitim Konfigürasyonu

### 5.1 Genel Ayarlar

```python
BaseConfig:
    environment: "local"  # veya "colab"
    dataset_name: "baseline_512"
    image_size: (512, 512)
    batch_size: 8
    augmentation_type: "enhanced"
    early_stopping_patience: 10
```

### 5.2 Early Stopping

```python
if avg_loss < best_loss:
    best_loss = avg_loss
    patience = 0
else:
    patience += 1

if patience >= early_stopping_patience:
    break  # Eğitimi durdur
```

### 5.3 Learning Rate Scheduler

```python
# Cosine Annealing LR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## 6. Değerlendirme Metrikleri

### 6.1 Kullanılan Metrikler

| Metrik    | Formül                | Açıklama                     |
| --------- | --------------------- | ---------------------------- |
| AUC-ROC   | Area Under ROC Curve  | Sınıflandırma performansı    |
| AUPRO     | Area Under PRO Curve  | Piksel seviyesi lokalizasyon |
| F1 Score  | 2×(P×R)/(P+R)         | Precision-Recall dengesi     |
| Accuracy  | (TP+TN)/(TP+TN+FP+FN) | Genel doğruluk               |
| Precision | TP/(TP+FP)            | Pozitif tahmin doğruluğu     |
| Recall    | TP/(TP+FN)            | Gerçek pozitifleri yakalama  |

### 6.2 Optimal Eşik Seçimi (Youden's J)

```python
def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]
```

### 6.3 AUPRO (Approximated)

```python
def calculate_aupro_approximated(anomaly_maps, labels, num_thresholds=100):
    # Ground truth mask olmadığı için yaklaşık hesaplama
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_scores = []

    for thresh in thresholds:
        defect_coverage = np.mean([np.mean(m > thresh) for m in defect_maps])
        good_fp_rate = np.mean([np.mean(m > thresh) for m in good_maps])

        # F1-like PRO score
        pro_score = 2 * defect_coverage * (1 - good_fp_rate) /
                    (defect_coverage + (1 - good_fp_rate) + 1e-8)
        pro_scores.append(pro_score)

    return np.trapz(pro_scores, thresholds)
```

---

## 7. Grid Search ve Deney Sonuçları

### 7.1 Grid Search Parametreleri

**Full Mode Kombinasyonları:**

- AutoEncoder: 3 × 3 × 2 = 18 kombinasyon
- PatchCore: 3 × 3 × 2 = 18 kombinasyon
- SimpleNet: 3 × 2 × 2 = 12 kombinasyon
- EfficientAD: 3 × 3 × 1 = 9 kombinasyon
- **Toplam:** 57 deney

### 7.2 Deney Sonuçları

<!-- RESULTS_TABLE_START -->

> ⚠️ **Not:** Bu bölüm, `run_full_experiments()` çalıştırıldıktan sonra `experiment_results.csv` dosyasından güncellenmelidir.

| Model | Experiment | AUC-ROC | AUPRO | F1 Score | Accuracy | Precision | Recall |
| ----- | ---------- | ------- | ----- | -------- | -------- | --------- | ------ |
| ...   | ...        | ...     | ...   | ...      | ...      | ...       | ...    |

<!-- RESULTS_TABLE_END -->

### 7.3 Model Bazlı En İyi Sonuçlar

<!-- BEST_RESULTS_START -->

> ⚠️ **Not:** Bu bölüm deney sonuçlarından güncellenmelidir.

**AutoEncoder:**

- Best Config: latent_dim=..., ssim_weight=..., lr=...
- AUC-ROC: ...
- F1 Score: ...

**PatchCore:**

- Best Config: memory_bank=..., k_neighbors=..., multi_layer=...
- AUC-ROC: ...
- F1 Score: ...

**SimpleNet:**

- Best Config: noise_std=..., adaptor_dim=..., multi_noise=...
- AUC-ROC: ...
- F1 Score: ...

**EfficientAD:**

- Best Config: student_dim=..., autoencoder_dim=...
- AUC-ROC: ...
- F1 Score: ...

<!-- BEST_RESULTS_END -->

---

## 8. Görselleştirmeler

### 8.1 Üretilen Çıktılar

| Dosya                      | Açıklama                               |
| -------------------------- | -------------------------------------- |
| `experiment_results.csv`   | Tüm deney sonuçları                    |
| `confusion_matrices.png`   | Model bazlı confusion matrix           |
| `roc_curves.png`           | ROC eğrileri karşılaştırması           |
| `score_distributions.png`  | Anomali skor dağılımları               |
| `metrics_comparison.png`   | Metrik karşılaştırma bar chart         |
| `best_config.json`         | En iyi model konfigürasyonu            |
| `summary_report.txt`       | Metin tabanlı özet rapor               |
| `heatmaps/{Model}/good/`   | İyi örneklerin anomali haritaları      |
| `heatmaps/{Model}/defect/` | Defektli örneklerin anomali haritaları |

### 8.2 Heatmap Görselleştirme

Her test görüntüsü için üçlü görselleştirme:

1. **Orijinal Görüntü**
2. **Anomali Haritası** (JET colormap)
3. **Overlay** (Orijinal + Heatmap)

---

## 9. Teknik Implementasyon Detayları

### 9.1 GPU Kullanımı

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model ve tensörler GPU'ya taşınır
model.to(device)
batch = batch.to(device)
```

### 9.2 Inference Pipeline

```
Input Image
    ↓
Resize to target_size (512×512)
    ↓
ImageNet Normalization
    ↓
Model Forward Pass → Patch-level Anomaly Scores
    ↓
Image Score = max(patch_scores)
    ↓
Anomaly Map → Upsample to original size (256×256)
    ↓
Score Normalization: (score - min) / (max - min)
    ↓
Threshold Comparison → Classification
```

### 9.3 Türkçe Karakter Desteği

```python
# Dosya yükleme (Windows uyumlu)
image_array = np.fromfile(image_path, dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Dosya kaydetme
_, encoded = cv2.imencode('.bmp', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
encoded.tofile(output_path)
```

---

## 10. Sonuçlar ve Öneriler

### 10.1 Temel Bulgular

<!-- CONCLUSIONS_START -->

> ⚠️ **Not:** Bu bölüm deney sonuçlarına göre güncellenmelidir.

1. **En iyi performans:** ...
2. **Veri artırma etkisi:** ...
3. **Çözünürlük etkisi (256 vs 512):** ...
4. **Model karşılaştırması:** ...

<!-- CONCLUSIONS_END -->

### 10.2 Öneriler

| Senaryo            | Önerilen Model | Gerekçe                |
| ------------------ | -------------- | ---------------------- |
| Yüksek Recall      | ...            | Tüm defektleri yakala  |
| Dengeli Performans | ...            | AUC-ROC ve F1 dengesi  |
| Hızlı Inference    | ...            | Düşük latency          |
| Lokalizasyon       | ...            | Piksel seviyesi tespit |

---

## Appendix: Dosya Referansları

### A.1 Ana Kod Dosyaları

| Dosya                          | Açıklama                            |
| ------------------------------ | ----------------------------------- |
| `compare_anomaly_models_v3.py` | Ana deney framework (1745 satır)    |
| `data_processing_script.py`    | Veri ön işleme pipeline (490 satır) |

### A.2 Sınıf ve Fonksiyon Referansları

**Configuration Classes:**

- `BaseConfig`: Temel konfigürasyon
- `AutoEncoderConfig`: AE hiperparametreleri
- `PatchCoreConfig`: PC hiperparametreleri
- `SimpleNetConfig`: SN hiperparametreleri
- `EfficientADConfig`: EAD hiperparametreleri

**Model Classes:**

- `ConvAutoEncoder`: AE mimarisi (nn.Module)
- `AutoEncoderModel`: AE training/inference wrapper
- `PatchCoreModel`: Memory bank + K-NN
- `SimpleNetModel`: Feature adaptor + discriminator
- `EfficientADModel`: Student-teacher + autoencoder

**Utility Classes:**

- `SSIMLoss`: Structural similarity loss
- `WoodDataset`: Dataset + augmentation
- `ExperimentRunner`: Grid search + visualization
- `ImagePreprocessor`: Veri ön işleme

---

**Rapor Sonu**
