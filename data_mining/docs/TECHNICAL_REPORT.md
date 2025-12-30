# UCI Heart Disease Veri Seti Üzerinde Makine Öğrenmesi ile Kalp Hastalığı Tahmini

## Teknik Rapor ve Dokümantasyon

**Tarih:** 31 Aralık 2024  
**Veri Seti:** UCI Heart Disease (Cleveland)  
**Analiz Türü:** Sınıflandırma (Binary Classification)

---

## 1. Giriş ve Problem Tanımı

### 1.1 Amaç

Bu çalışmada, UCI Heart Disease veri seti kullanılarak kalp hastalığı tahmin modelleri geliştirilmiştir. Çalışmanın temel hedefleri:

1. Farklı veri önişleme tekniklerinin model performansına etkisini incelemek
2. Hiperparametre optimizasyonu (Optuna) ile model performansını maksimize etmek
3. Farklı makine öğrenmesi algoritmalarını karşılaştırmak
4. Açıklanabilir yapay zeka (XAI) teknikleri ile model kararlarını yorumlamak

### 1.2 Veri Seti Özellikleri

- **Kaynak:** UCI Machine Learning Repository
- **Alt Veri Seti:** Cleveland (en yaygın kullanılan)
- **Örneklem Sayısı:** 304
- **Özellik Sayısı:** 13 (orijinal) + 4 (mühendislik) = 17
- **Hedef Değişken:** Binary (0: Sağlıklı, 1: Kalp Hastalığı)
- **Sınıf Dağılımı:** Sağlıklı: 165 (%54.3), Hasta: 139 (%45.7)

---

## 2. Veri Seti Özellikleri (Features)

### 2.1 Orijinal Özellikler

| Özellik    | Açıklama                              | Tip       | Değer Aralığı                                              |
| ---------- | ------------------------------------- | --------- | ---------------------------------------------------------- |
| `age`      | Yaş                                   | Sürekli   | 28-77 yıl                                                  |
| `sex`      | Cinsiyet                              | Kategorik | Male, Female                                               |
| `cp`       | Göğüs ağrısı tipi                     | Kategorik | typical angina, atypical angina, non-anginal, asymptomatic |
| `trestbps` | Dinlenme kan basıncı                  | Sürekli   | 94-200 mmHg                                                |
| `chol`     | Serum kolesterol                      | Sürekli   | 126-564 mg/dl                                              |
| `fbs`      | Açlık kan şekeri > 120 mg/dl          | Binary    | TRUE, FALSE                                                |
| `restecg`  | Dinlenme EKG sonucu                   | Kategorik | normal, st-t abnormality, lv hypertrophy                   |
| `thalch`   | Maksimum kalp hızı                    | Sürekli   | 71-202 bpm                                                 |
| `exang`    | Egzersize bağlı angina                | Binary    | TRUE, FALSE                                                |
| `oldpeak`  | Egzersizin neden olduğu ST depresyonu | Sürekli   | -2.6 - 6.2                                                 |
| `slope`    | ST segment eğimi                      | Kategorik | upsloping, flat, downsloping                               |
| `ca`       | Floroskopi ile boyanan damar sayısı   | Ordinal   | 0-3                                                        |
| `thal`     | Talasemi                              | Kategorik | normal, fixed defect, reversable defect                    |

### 2.2 Mühendislik Özellikleri (Feature Engineering)

| Yeni Özellik          | Formül                            | Gerekçe                                                        |
| --------------------- | --------------------------------- | -------------------------------------------------------------- |
| `risk_score`          | (age × chol) / 10000              | Yaş ve kolesterol etkileşimi - kardiyovasküler risk göstergesi |
| `age_group`           | Binning (0-40, 40-55, 55-70, 70+) | Yaş kategorileri ile risk gruplandırması                       |
| `hr_age_ratio`        | thalch / age                      | Yaşa göre normalize kalp hızı performansı                      |
| `bp_chol_interaction` | (trestbps × chol) / 10000         | Kan basıncı ve kolesterol etkileşimi                           |

---

## 3. Metodoloji

### 3.1 Veri Önişleme Pipeline

```
Ham Veri (920 satır)
    │
    ├── 1. Cleveland Filtresi → 304 satır
    │
    ├── 2. Kategorik Encoding (LabelEncoder)
    │     ├── sex: 2 kategori
    │     ├── cp: 4 kategori
    │     ├── restecg: 3 kategori
    │     ├── exang: 2 kategori
    │     ├── slope: 3 kategori
    │     ├── thal: 4 kategori
    │     └── fbs: 2 kategori
    │
    ├── 3. Eksik Değer Doldurma (KNN Imputer, k=5)
    │     └── 5 eksik değer → 0
    │
    ├── 4. Aykırı Değer Baskılama (Winsorizing %5-%95)
    │     ├── age: [28, 77] → [39, 68]
    │     ├── trestbps: [94, 200] → [108, 160]
    │     ├── chol: [126, 564] → [175, 327]
    │     ├── thalch: [71, 202] → [108, 182]
    │     └── oldpeak: [0, 6.2] → [0, 3.4]
    │
    ├── 5. Özellik Mühendisliği (+4 yeni özellik)
    │
    ├── 6. Ölçekleme (RobustScaler / StandardScaler)
    │
    └── 7. Sınıf Dengeleme (SMOTE)
          └── 165 Sağlıklı, 139 Hasta → 165 Sağlıklı, 165 Hasta
```

### 3.2 Kullanılan Teknikler

#### 3.2.1 KNN Imputer

- **Neden:** Ortalama/medyan yerine benzer örneklerin değerlerini kullanır
- **Parametre:** k=5 (en yakın 5 komşu)
- **Avantaj:** Verinin çok değişkenli yapısını korur

#### 3.2.2 Winsorizing (Aykırı Değer Baskılama)

- **Neden:** Küçük veri setlerinde aykırı değerleri silmek yerine baskılamak tercih edilir
- **Parametre:** %5-%95 perzantil aralığı
- **Avantaj:** Veri kaybını önler, aşırı uç değerlerin etkisini azaltır

#### 3.2.3 RobustScaler

- **Neden:** Aykırı değerlere karşı dirençli ölçekleme
- **Formül:** (x - median) / IQR
- **Avantaj:** StandardScaler'a göre outlier'lara daha az duyarlı

#### 3.2.4 SMOTE (Synthetic Minority Over-sampling Technique)

- **Neden:** Sınıf dengesizliğini gidermek için sentetik azınlık örnekleri üretir
- **Önemli:** Sadece eğitim setine uygulanır (data leakage önlemi)
- **Sonuç:** 165 vs 139 → 165 vs 165 (dengeli sınıflar)

### 3.3 Validasyon Stratejisi

```
Stratified 10-Fold Cross Validation
├── Fold 1: Test (30), Eğitim (274)
├── Fold 2: Test (30), Eğitim (274)
├── ...
└── Fold 10: Test (30), Eğitim (274)

Sonuç Formatı: Ortalama ± Standart Sapma
Örnek: F1 = 0.843 ± 0.090
```

**Neden Stratified K-Fold?**

- Her fold'da sınıf oranları korunur (Hasta/Sağlıklı)
- Küçük veri setlerinde tek train/test split yetersiz
- Daha güvenilir ve tekrarlanabilir sonuçlar

### 3.4 Hiperparametre Optimizasyonu (Optuna)

Optuna, Bayesian optimizasyon tabanlı bir hiperparametre arama kütüphanesidir.

**Avantajları:**

- Grid Search'ten daha verimli (gereksiz kombinasyonları atlar)
- Random Search'ten daha akıllı (önceki denemelerden öğrenir)
- Pruning özelliği ile zaman tasarrufu

**Optimizasyon Ayarları:**

- Trial sayısı: 30-50
- Metrik: F1-Score (maximize)
- Yön: maximize

---

## 4. Deneysel Sonuçlar

### 4.1 Senaryo Karşılaştırması

5 farklı senaryo, aynı model (Random Forest) üzerinde test edilmiştir:

| Senaryo             | Açıklama                            | F1-Score        | AUC       | Baseline'a Göre |
| ------------------- | ----------------------------------- | --------------- | --------- | --------------- |
| **1. Baseline**     | Varsayılan RF parametreleri         | 0.824±0.083     | 0.911     | -               |
| **2. Feature Eng.** | +4 mühendislik özelliği             | 0.822±0.087     | 0.909     | -0.2%           |
| **3. PCA**          | %95 varyans koruyan boyut indirgeme | 0.814±0.072     | 0.905     | -1.2%           |
| **4. Optuna**       | Hiperparametre optimizasyonu        | **0.843±0.090** | **0.912** | **+2.3%**       |
| **5. All Combined** | FE + PCA + Optuna                   | 0.824±0.050     | 0.904     | 0%              |

**Bulgular:**

1. **Optuna tek başına en etkili** iyileştirme sağlamıştır (+2.3%)
2. **Feature Engineering bu veri setinde etkisiz** - Cleveland zaten iyi tasarlanmış özellikler içeriyor
3. **PCA performansı düşürmüştür** - Küçük veri setlerinde boyut indirgeme genellikle zararlı
4. **All Combined'da varyans düşük** (±0.050) - Daha kararlı ama peak performans düşük

### 4.2 Model Karşılaştırması (Optuna Optimized)

Tüm modeller aynı preprocessing pipeline üzerinde Optuna ile optimize edilmiştir:

| Model               | Accuracy        | Recall          | F1-Score        | AUC       | Varyans   |
| ------------------- | --------------- | --------------- | --------------- | --------- | --------- |
| Logistic Regression | 0.845±0.078     | 0.806±0.082     | 0.839±0.082     | 0.907     | Orta      |
| Random Forest       | 0.848±0.092     | 0.822±0.119     | 0.841±0.101     | 0.912     | Yüksek    |
| **SVM**             | **0.848±0.049** | 0.824±0.087     | **0.843±0.052** | 0.905     | **Düşük** |
| XGBoost             | 0.845±0.072     | **0.829±0.083** | 0.843±0.077     | **0.915** | Orta      |

**Optimum Hiperparametreler:**

```
Logistic Regression:
  - C: ~1.5 (regularization strength)
  - penalty: l2

Random Forest:
  - n_estimators: 200-250
  - max_depth: 9-15
  - min_samples_split: 4-10
  - min_samples_leaf: 1-2

SVM:
  - C: ~10-50
  - kernel: rbf
  - gamma: scale

XGBoost:
  - n_estimators: 150-200
  - max_depth: 5-8
  - learning_rate: 0.05-0.15
  - subsample: 0.7-0.9
  - colsample_bytree: 0.7-0.9
```

### 4.3 Özellik Önem Analizi

Random Forest feature importance sonuçları:

| Sıra | Özellik                     | Önem Skoru | Yorumu                                       |
| ---- | --------------------------- | ---------- | -------------------------------------------- |
| 1    | `cp` (Göğüs Ağrısı Tipi)    | 0.1736     | **En kritik** - Asymptomatic tip yüksek risk |
| 2    | `thal` (Talasemi)           | 0.1356     | Reversable defect kalp hastalığı göstergesi  |
| 3    | `thalch` (Max Kalp Hızı)    | 0.1321     | Düşük max HR kötü prognoz                    |
| 4    | `ca` (Boyanan Damar Sayısı) | 0.1295     | Daha fazla damar = daha yüksek risk          |
| 5    | `oldpeak` (ST Depresyonu)   | 0.0978     | EKG anomalisi göstergesi                     |
| 6    | `exang` (Egzersiz Anginası) | 0.0902     | Pozitif = yüksek risk                        |
| 7    | `slope` (ST Eğimi)          | 0.0615     | Flat/downsloping = risk                      |
| 8    | `age` (Yaş)                 | 0.0582     | Yaşla birlikte artan risk                    |
| 9    | `trestbps` (Kan Basıncı)    | 0.0398     | Hipertansiyon riski                          |
| 10   | `sex` (Cinsiyet)            | 0.0350     | Erkeklerde daha yüksek risk                  |

### 4.4 SHAP Analizi (Açıklanabilir AI)

SHAP (SHapley Additive exPlanations) değerleri, her özelliğin model kararına katkısını gösterir:

**Pozitif SHAP Değeri:** Kalp hastalığı riskini artırır
**Negatif SHAP Değeri:** Riski azaltır

**Temel Bulgular:**

- `cp = asymptomatic` → En yüksek pozitif etki (risk artışı)
- `thal = reversable defect` → Güçlü pozitif etki
- `thalch` yüksek → Negatif etki (risk azalması)
- `ca` yüksek → Pozitif etki (risk artışı)

---

## 5. Performans Metrikleri Açıklaması

### 5.1 Metrik Tanımları

| Metrik                   | Formül                | Tıbbi Yorumu                          |
| ------------------------ | --------------------- | ------------------------------------- |
| **Accuracy**             | (TP+TN)/(TP+TN+FP+FN) | Genel doğruluk, yanıltıcı olabilir    |
| **Precision**            | TP/(TP+FP)            | "Hasta" dediğimizde ne kadar haklıyız |
| **Recall (Sensitivity)** | TP/(TP+FN)            | Gerçek hastaların kaçını yakaladık    |
| **F1-Score**             | 2×(P×R)/(P+R)         | Precision ve Recall dengesi           |
| **AUC-ROC**              | Area Under ROC Curve  | Sınıfları ayırt etme yeteneği         |

### 5.2 Tıbbi Bağlamda Metrik Önceliği

**Kritik:** Tıbbi taramada **Recall (Duyarlılık)** en önemli metriktir.

**Neden?**

- **False Negative (Tip II Hata):** Gerçek hastaya "sağlıklı" demek → Tedavi gecikmesi → Ölüm riski
- **False Positive (Tip I Hata):** Sağlıklı kişiye "hasta" demek → Gereksiz testler → Maliyet

**Sonuç:** FN >> FP maliyeti → Recall optimizasyonu kritik

### 5.3 Confusion Matrix Yorumu

```
                    Tahmin
                Sağlıklı   Hasta
Gerçek  Sağlıklı   TN       FP (Tip I)
        Hasta      FN (Tip II)  TP
```

**Örnek Sonuç (SVM):**

- TP (True Positive): ~136
- TN (True Negative): ~143
- FP (False Positive): ~22
- FN (False Negative): ~29

**Recall = 136 / (136+29) = 0.824** → Hastaların %82.4'ünü doğru tespit

---

## 6. Model Seçimi Önerileri

### 6.1 Uygulama Senaryolarına Göre

| Senaryo                   | Önerilen Model      | Gerekçe                                     |
| ------------------------- | ------------------- | ------------------------------------------- |
| **Tarama Programı**       | XGBoost             | En yüksek Recall (0.829) - Az hasta kaçırır |
| **Klinik Karar Destek**   | SVM                 | En düşük varyans (±0.052) - Güvenilir       |
| **Araştırma Projesi**     | Random Forest       | Yorumlanabilir feature importance           |
| **Gerçek Zamanlı Sistem** | Logistic Regression | Hızlı, basit, yorumlanabilir                |

### 6.2 Nihai Öneri

**XGBoost** modeli şu avantajlara sahiptir:

- En yüksek AUC (0.915) - Ayırt edicilik
- En yüksek Recall (0.829) - Hasta yakalama
- Dengeli varyans (±0.077)
- Gradient boosting ile robust performans

---

## 7. Sınırlamalar ve Gelecek Çalışmalar

### 7.1 Mevcut Sınırlamalar

1. **Küçük Veri Seti:** 304 örnek, modern standartlara göre yetersiz
2. **Tek Merkez Verisi:** Sadece Cleveland - genelleştirilebilirlik sorunu
3. **Temporal Validasyon Yok:** Zaman içinde performans değişimi test edilmedi
4. **Derin Öğrenme Uygulanmadı:** Veri yetersizliği nedeniyle

### 7.2 Gelecek Çalışma Önerileri

1. **Multi-center Validation:** Hungary, Switzerland, VA Long Beach verileri ile test
2. **Ensemble Methods:** Voting/Stacking ile model birleştirme
3. **Cost-sensitive Learning:** FN maliyetini yüksek tutarak eğitim
4. **Prospektif Validasyon:** Yeni hasta verileri ile gerçek dünya testi
5. **Neural Network:** TabNet veya 1D-CNN denemesi (daha fazla veri ile)

---

## 8. Teknik Uygulama Detayları

### 8.1 Kullanılan Kütüphaneler

```python
# Veri İşleme
pandas==2.0+
numpy==1.24+

# Makine Öğrenmesi
scikit-learn==1.3+
xgboost==2.0+
imbalanced-learn==0.11+  # SMOTE

# Hiperparametre Optimizasyonu
optuna==3.4+

# Görselleştirme
matplotlib==3.7+
seaborn==0.12+

# Açıklanabilir AI
shap==0.43+
```

### 8.2 Kod Dosyaları

| Dosya                           | Açıklama                            |
| ------------------------------- | ----------------------------------- |
| `main_2_new_technics.py`        | Temel analiz pipeline, tüm modeller |
| `main_3_scenario_comparison.py` | Senaryo karşılaştırması + Optuna    |
| `results_*/`                    | Sonuç klasörleri (tarih damgalı)    |

### 8.3 Çalıştırma Komutları

```bash
# Temel analiz
python main_2_new_technics.py

# Senaryo karşılaştırması + Model optimizasyonu
python main_3_scenario_comparison.py
```

---

## 9. Sonuç

Bu çalışmada UCI Heart Disease Cleveland veri seti üzerinde kapsamlı bir makine öğrenmesi analizi gerçekleştirilmiştir. Temel bulgular:

1. **Hiperparametre optimizasyonu (Optuna)** en etkili iyileştirme yöntemidir (+2.3% F1)
2. **Feature engineering ve PCA** bu veri setinde fayda sağlamamıştır
3. **SVM ve XGBoost** en iyi performansı göstermiştir (F1 ≈ 0.843)
4. **Göğüs ağrısı tipi (cp)** en önemli prediktör özeliktir
5. **Stratified 10-Fold CV** küçük veri setleri için zorunludur

**Tıbbi öneri:** Tarama programlarında XGBoost modeli, yüksek Recall değeri ile hastaların tespit edilme oranını maksimize eder.

---

## Kaynaklar

1. UCI Machine Learning Repository - Heart Disease Dataset
2. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
3. Lundberg, S. M., & Lee, S. I. (2017). SHAP: A Unified Approach to Interpreting Model Predictions
4. Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework
5. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique

---

**Rapor Sonu**

_Bu rapor, UCI Heart Disease veri seti üzerinde yapılan kapsamlı makine öğrenmesi analizinin teknik dokümantasyonudur._
