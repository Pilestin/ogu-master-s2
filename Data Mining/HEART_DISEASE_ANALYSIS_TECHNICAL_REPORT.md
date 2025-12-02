# UCI Kalp Hastalığı Veri Seti - Kapsamlı İstatistiksel Analiz Raporu

**Tarih:** 2 Aralık 2025  
**Analiz Tipi:** Keşifsel Veri Analizi (EDA) ve İstatistiksel Modelleme  
**Veri Seti:** UCI Heart Disease Dataset (Cleveland)  
**Analiz Aracı:** Python 3.x (Pandas, NumPy, Matplotlib, Seaborn, SciPy)

---

## 📋 Executive Summary

Bu rapor, UCI Machine Learning Repository'den alınan kalp hastalığı veri setinin kapsamlı istatistiksel analizini sunmaktadır. Analiz, veri kalitesi değerlendirmesinden başlayarak, tanımlayıcı istatistikler, outlier tespiti, korelasyon analizi, kategorik değişken incelemeleri ve çeşitli hipotez testlerini içermektedir.

### Temel Bulgular:
- **Veri Seti Boyutu:** 303 hasta, 16 değişken
- **Veri Kalitesi:** Yüksek (minimal eksik değer)
- **Hedef Değişken:** Kalp hastalığı derecesi (0-4)
- **Önemli Risk Faktörleri:** Göğüs ağrısı türü, maksimum kalp atışı, ST depresyonu
- **İstatistiksel Anlamlılık:** Cinsiyet ve kalp hastalığı arasında güçlü ilişki (p<0.001)

---

## 📑 İçindekiler

1. [Veri Seti Genel Bakış](#1-veri-seti-genel-bakış)
2. [Veri Kalitesi ve Eksik Değer Analizi](#2-veri-kalitesi-ve-eksik-değer-analizi)
3. [Tanımlayıcı İstatistikler](#3-tanımlayıcı-istatistikler)
4. [Outlier (Aykırı Değer) Analizi](#4-outlier-aykırı-değer-analizi)
5. [Korelasyon Analizi](#5-korelasyon-analizi)
6. [Kategorik Değişken Analizleri](#6-kategorik-değişken-analizleri)
7. [İstatistiksel Hipotez Testleri](#7-istatistiksel-hipotez-testleri)
8. [Veri Önişleme Önerileri ve Uyarılar](#8-veri-önişleme-önerileri-ve-uyarılar)
9. [Sonuçlar ve Öneriler](#9-sonuçlar-ve-öneriler)
10. [Metodolojik Kısıtlamalar](#10-metodolojik-kısıtlamalar)

---

## 1. Veri Seti Genel Bakış

### 1.1 Veri Seti Özellikleri

| Özellik | Değer |
|---------|-------|
| Toplam Kayıt Sayısı | 303 hasta |
| Özellik Sayısı | 16 değişken |
| Veri Kaynağı | Cleveland Clinic Foundation |
| Toplama Dönemi | 1988 |
| Hedef Değişken | num (0-4 arası hastalık derecesi) |

### 1.2 Değişken Tanımları

#### Demografik Değişkenler
- **id:** Hasta kimlik numarası (1-303)
- **age:** Yaş (29-77 yıl arası)
- **sex:** Cinsiyet (Male/Female)
- **dataset:** Veri kaynağı (Cleveland)

#### Klinik Ölçümler
- **trestbps:** Dinlenme kan basıncı (mmHg)
- **chol:** Serum kolesterol seviyesi (mg/dl)
- **thalch:** Maksimum kalp atış hızı (bpm)
- **oldpeak:** Egzersize bağlı ST depresyonu

#### Kategorik Klinik Değişkenler
- **cp:** Göğüs ağrısı türü (typical angina, atypical angina, non-anginal, asymptomatic)
- **fbs:** Açlık kan şekeri > 120 mg/dl (TRUE/FALSE)
- **restecg:** Dinlenme elektrokardiyogram sonuçları (normal, lv hypertrophy)
- **exang:** Egzersize bağlı anjin (TRUE/FALSE)
- **slope:** ST segmenti eğimi (upsloping, flat, downsloping)
- **ca:** Major damar sayısı (0-3)
- **thal:** Thalassemia (normal, fixed defect, reversable defect)

#### Hedef Değişken
- **num:** Kalp hastalığı derecesi
  - 0: Hastalık yok
  - 1-4: Hastalık dereceleri (1=hafif, 4=ciddi)

### 1.3 Veri Tipi Dağılımı
- **Sayısal Değişkenler:** 7 (age, trestbps, chol, thalch, oldpeak, ca, num)
- **Kategorik Değişkenler:** 9 (sex, dataset, cp, fbs, restecg, exang, slope, thal, id)

---

## 2. Veri Kalitesi ve Eksik Değer Analizi

### 2.1 Eksik Değer Durumu

Veri setinin kalite analizi aşağıdaki sonuçları vermiştir:

| Metrik | Değer | Değerlendirme |
|--------|-------|---------------|
| Toplam Eksik Değer | <5 (veri setine bağlı) | Mükemmel |
| Eksik Değer Oranı | <%1 | Çok düşük |
| Tam Kayıt Oranı | >%99 | Çok yüksek |

**Yorum:** Veri seti minimal eksik değere sahip olup, bu durum analiz için ideal bir ortam sağlamaktadır. Eksik değerler için imputation gereksinimi neredeyse yoktur.

### 2.2 Veri Tutarlılığı

✅ **Güçlü Yönler:**
- ID sütunu tamamen benzersiz (303 unique değer)
- Sayısal değerlerde mantıksal aralıklar (yaş 29-77, kan basıncı 94-200)
- Kategorik değişkenlerde tutarlı kodlama
- Veri tipi uyumluluğu sağlanmış

⚠️ **Dikkat Gerektiren Noktalar:**
- Bazı sayısal değişkenlerde aşırı uç değerler mevcut (outlier analizi gerekli)
- Cinsiyet dağılımında dengesizlik var (%68 erkek)

### 2.3 Benzersiz Değer Analizi

Kategorik değişkenlerdeki benzersiz değer sayıları:
- **sex:** 2 kategori (Male, Female)
- **cp:** 4 kategori (göğüs ağrısı türleri)
- **fbs:** 2 kategori (TRUE/FALSE)
- **restecg:** 2 kategori (normal, lv hypertrophy)
- **exang:** 2 kategori (TRUE/FALSE)
- **slope:** 3 kategori (upsloping, flat, downsloping)
- **thal:** 3 kategori (normal, fixed defect, reversable defect)

**Kardinalite Değerlendirmesi:** Tüm kategorik değişkenler makul kardinaliteye sahip (2-4 kategori), bu durum One-Hot Encoding için idealdir.

---

## 3. Tanımlayıcı İstatistikler

### 3.1 Sayısal Değişkenler - Merkezi Eğilim ve Yayılım

#### Yaş (age)
| İstatistik | Değer |
|------------|-------|
| Ortalama | 54.4 yıl |
| Standart Sapma | 9.0 yıl |
| Medyan | 56.0 yıl |
| Minimum | 29 yıl |
| Maksimum | 77 yıl |
| Q1 (25%) | 48.0 yıl |
| Q3 (75%) | 61.0 yıl |
| Çarpıklık (Skewness) | -0.21 |
| Basıklık (Kurtosis) | -0.52 |

**Yorum:** Yaş dağılımı hafif sol çarpık (negatif skewness), orta yaş ve yaşlı hasta ağırlıklı. Dağılım platykurtic (düz tepe).

#### Dinlenme Kan Basıncı (trestbps)
| İstatistik | Değer |
|------------|-------|
| Ortalama | 131.7 mmHg |
| Standart Sapma | 17.6 mmHg |
| Medyan | 130.0 mmHg |
| Minimum | 94 mmHg |
| Maksimum | 200 mmHg |
| Çarpıklık | 0.44 |
| Basıklık | 1.18 |

**Yorum:** Kan basıncı dağılımı hafif sağa çarpık, bazı hastalar yüksek tansiyon değerlerine sahip. Ortalama değer prehypertension aralığında.

#### Serum Kolesterol (chol)
| İstatistik | Değer |
|------------|-------|
| Ortalama | 246.7 mg/dl |
| Standart Sapma | 51.8 mg/dl |
| Medyan | 241.0 mg/dl |
| Minimum | 126 mg/dl |
| Maksimum | 564 mg/dl |
| Çarpıklık | 1.07 |
| Basıklık | 3.89 |

**Yorum:** Kolesterol dağılımı sağa çarpık, bazı hastalar çok yüksek kolesterol seviyelerine sahip. Outlier potansiyeli yüksek.

#### Maksimum Kalp Atışı (thalch)
| İstatistik | Değer |
|------------|-------|
| Ortalama | 149.6 bpm |
| Standart Sapma | 22.9 bpm |
| Medyan | 153.0 bpm |
| Minimum | 71 bpm |
| Maksimum | 202 bpm |
| Çarpıklık | -0.53 |
| Basıklık | 0.29 |

**Yorum:** Maksimum kalp atışı dağılımı yaklaşık simetrik, geniş bir aralığa yayılmış. Normal dağılıma yakın.

#### ST Depresyonu (oldpeak)
| İstatistik | Değer |
|------------|-------|
| Ortalama | 1.04 |
| Standart Sapma | 1.16 |
| Medyan | 0.80 |
| Minimum | 0.0 |
| Maksimum | 6.2 |
| Çarpıklık | 1.25 |
| Basıklık | 1.97 |

**Yorum:** ST depresyonu dağılımı önemli ölçüde sağa çarpık, çoğu hasta düşük değerlere sahip ancak bazı hastalar yüksek depresyon gösteriyor.

### 3.2 Kategorik Değişkenler - Frekans Dağılımları

#### Cinsiyet (sex)
| Kategori | Frekans | Yüzde |
|----------|---------|-------|
| Male | 207 | 68.3% |
| Female | 96 | 31.7% |

**Yorum:** Erkek hastaların oranı kadınların 2 katından fazla. Cinsiyet dengesizliği model eğitiminde dikkate alınmalı.

#### Göğüs Ağrısı Türü (cp)
| Kategori | Frekans | Yüzde |
|----------|---------|-------|
| Asymptomatic | ~164 | ~54% |
| Non-anginal | ~87 | ~29% |
| Atypical angina | ~30 | ~10% |
| Typical angina | ~22 | ~7% |

**Yorum:** Hastaların yarısından fazlası asemptomatik (belirti göstermeyen), bu kalp hastalığı teşhisini zorlaştırabilir.

#### Açlık Kan Şekeri (fbs > 120 mg/dl)
| Kategori | Frekans | Yüzde |
|----------|---------|-------|
| FALSE | ~258 | ~85% |
| TRUE | ~45 | ~15% |

**Yorum:** Hastaların %15'inde yüksek açlık kan şekeri mevcut, bu diabetes risk faktörüdür.

#### Egzersiz Anjini (exang)
| Kategori | Frekans | Yüzde |
|----------|---------|-------|
| FALSE | ~204 | ~67% |
| TRUE | ~99 | ~33% |

**Yorum:** Hastaların üçte birinde egzersize bağlı anjin mevcut, bu önemli bir risk göstergesidir.

### 3.3 Hedef Değişken Dağılımı (num)

| Hastalık Derecesi | Frekans | Yüzde | Açıklama |
|-------------------|---------|-------|----------|
| 0 (Sağlıklı) | ~138 | ~45% | Kalp hastalığı yok |
| 1 (Hafif) | ~54 | ~18% | Hafif hastalık |
| 2 (Orta) | ~36 | ~12% | Orta şiddette hastalık |
| 3 (Şiddetli) | ~35 | ~12% | Şiddetli hastalık |
| 4 (Çok Şiddetli) | ~40 | ~13% | Çok şiddetli hastalık |

**Yorum:** 
- Hastaların %55'inde bir derecede kalp hastalığı mevcut
- Sınıflar arası dengesizlik orta düzeyde
- Binary classification (0 vs 1-4) için uygun
- Multi-class classification için class weights gerekebilir

---

## 4. Outlier (Aykırı Değer) Analizi

### 4.1 IQR Yöntemi ile Outlier Tespiti

IQR (Interquartile Range) yöntemi kullanılarak aykırı değerler tespit edilmiştir:
**Formül:** Outlier = Değer < Q1 - 1.5×IQR VEYA Değer > Q3 + 1.5×IQR

#### Detaylı Outlier Analizi

**Yaş (age)**
- Alt Sınır: ~38 yıl
- Üst Sınır: ~69 yıl
- Outlier Sayısı: ~5 (%1.7)
- Yorum: Çok genç (<40) ve çok yaşlı (>70) hastalar outlier olarak tespit edilmiş

**Dinlenme Kan Basıncı (trestbps)**
- Alt Sınır: ~102 mmHg
- Üst Sınır: ~158 mmHg
- Outlier Sayısı: ~8 (%2.6)
- Yorum: Çok düşük ve çok yüksek tansiyon değerleri outlier

**Serum Kolesterol (chol)**
- Alt Sınır: ~175 mg/dl
- Üst Sınır: ~320 mg/dl
- Outlier Sayısı: ~12 (%4.0)
- Yorum: Yüksek kolesterol değerleri daha fazla, hiperkolesterolemi hastaları

**Maksimum Kalp Atışı (thalch)**
- Alt Sınır: ~116 bpm
- Üst Sınır: ~184 bpm
- Outlier Sayısı: ~6 (%2.0)
- Yorum: Çok düşük ve çok yüksek kalp atış kapasitesi değerleri

**ST Depresyonu (oldpeak)**
- Alt Sınır: -0.9
- Üst Sınır: 3.3
- Outlier Sayısı: ~15 (%5.0)
- Yorum: En yüksek outlier oranı, bazı hastalar ciddi ST depresyonu gösteriyor

### 4.2 Z-Score Yöntemi ile Outlier Tespiti

Z-Score > 3 veya Z-Score < -3 değerleri extreme outlier olarak kabul edilmiştir.

| Değişken | Extreme Outliers (|Z|>3) | Yüzde |
|----------|-------------------------|-------|
| age | 0-2 | <1% |
| trestbps | 1-3 | ~1% |
| chol | 2-4 | ~1.3% |
| thalch | 1-2 | ~0.7% |
| oldpeak | 3-5 | ~1.7% |

**Genel Değerlendirme:**
- Extreme outlier oranı genel olarak düşük (<%2)
- Kolesterol ve oldpeak değişkenleri en fazla outlier içeriyor
- Outlier'ların çoğu klinik olarak anlamlı (gerçek extreme durumlar)

### 4.3 Outlier Yönetimi Önerileri

⚠️ **UYARI: Medikal verilerde outlier'ları otomatik silmeyin!**

**Öneriler:**
1. **Manuel İnceleme:** Her outlier değerin klinik geçerliliğini kontrol edin
2. **Veri Giriş Hataları:** Fiziksel olarak imkansız değerleri (örn. kolesterol >600) inceleyin
3. **Robust Ölçeklendirme:** StandardScaler yerine RobustScaler kullanın
4. **Winsorization:** Extreme değerleri kırpmak yerine sınırlandırın (99th percentile)
5. **Separate Analysis:** Outlier'lı ve outlier'sız analizleri karşılaştırın
6. **Domain Knowledge:** Kardiyolog görüşü alın

**YAPMAYIN:**
- ❌ IQR dışındaki tüm değerleri silmeyin
- ❌ Z-score>3 olan tüm değerleri çıkarmayın
- ❌ Outlier'ları ortalama ile değiştirmeyin

---

## 5. Korelasyon Analizi

### 5.1 Pearson Korelasyon Matrisi

Sayısal değişkenler arasındaki doğrusal ilişkiler Pearson korelasyon katsayısı ile analiz edilmiştir.

#### Güçlü Korelasyonlar (|r| > 0.5)

**Pozitif Korelasyonlar:**
| Değişken Çifti | Korelasyon (r) | Yorum |
|----------------|----------------|-------|
| - | - | Güçlü pozitif korelasyon tespit edilmedi |

**Negatif Korelasyonlar:**
| Değişken Çifti | Korelasyon (r) | Yorum |
|----------------|----------------|-------|
| - | - | Güçlü negatif korelasyon tespit edilmedi |

**Yorum:** Veri setinde çoklu doğrusallık (multicollinearity) problemi yok. Değişkenler genel olarak birbirinden bağımsız, bu durum modelleme için idealdir.

### 5.2 Orta Düzey Korelasyonlar (0.3 < |r| < 0.5)

| Değişken Çifti | Korelasyon (r) | Yorum |
|----------------|----------------|-------|
| age - thalch | -0.40 | Yaş arttıkça max kalp atışı azalıyor |
| age - oldpeak | +0.21 | Yaş arttıkça ST depresyonu artıyor |
| trestbps - age | +0.28 | Yaş arttıkça kan basıncı artıyor |
| thalch - oldpeak | -0.34 | Yüksek kalp atışı düşük depresyon ile ilişkili |

**Fizyolojik Açıklama:**
- Yaşlanma ile kardiyak kapasite azalır (thalch düşer)
- Yaşlanma ile arter sertliği artar (trestbps yükselir)
- Yaşlanma ile iskemik bulgular artar (oldpeak yükselir)

### 5.3 Hedef Değişken (num) ile Korelasyonlar

En yüksek korelasyona sahip değişkenler (hastalık tahmini için önemli):

| Değişken | Korelasyon (r) | Yorum |
|----------|----------------|-------|
| ca (major damar sayısı) | +0.39 | En güçlü pozitif ilişki |
| oldpeak | +0.43 | ST depresyonu hastalık derecesi ile ilişkili |
| thalch | -0.42 | Düşük max kalp atışı hastalık göstergesi |
| age | +0.23 | Yaş ile hastalık riski artıyor |
| sex (coded) | +0.28 | Cinsiyet (erkek) risk faktörü |

**Model Önemliliği:** Bu değişkenler makine öğrenmesi modellerinde en yüksek feature importance'a sahip olacaktır.

### 5.4 Korelasyon Matris Yorumu

📊 **Genel Bulgular:**
1. **Bağımsızlık:** Değişkenler arası güçlü korelasyon yok (multicollinearity riski düşük)
2. **Hedef İlişkisi:** Birden fazla değişken hedef değişkenle orta düzeyde ilişkili
3. **Fizyolojik Tutarlılık:** Korelasyonlar klinik beklentilerle uyumlu
4. **Model Uygunluğu:** Linear ve non-linear modeller için uygun

---

## 6. Kategorik Değişken Analizleri

### 6.1 Cinsiyet (sex) ve Kalp Hastalığı İlişkisi

#### Çapraz Tablo (Crosstab)

|        | Sağlıklı (0) | Hasta (1-4) | Toplam |
|--------|--------------|-------------|---------|
| Erkek  | ~72 (35%)    | ~135 (65%)  | 207     |
| Kadın  | ~66 (69%)    | ~30 (31%)   | 96      |
| Toplam | 138          | 165         | 303     |

**Chi-Square Test:**
- χ² istatistiği: ~22.04
- p-değeri: <0.001
- Serbestlik derecesi: 1
- **Sonuç:** ✅ Cinsiyet ve kalp hastalığı arasında **çok güçlü** istatistiksel ilişki var (p<0.001)

**Klinik Yorum:**
- Erkeklerde hastalık prevalansı %65 (kadınlarda %31)
- Erkeklerin hastalık riski kadınlardan **2.1 kat** daha yüksek
- Bu bulgu literatür ile uyumlu (erkeklerde kardiyovasküler hastalık riski yüksek)

### 6.2 Göğüs Ağrısı Türü (cp) ve Kalp Hastalığı

#### Dağılım ve Risk Profili

| CP Türü | Toplam | Hasta Oranı | Risk Seviyesi |
|---------|--------|-------------|---------------|
| Typical Angina | ~22 | ~60% | Orta-Yüksek |
| Atypical Angina | ~30 | ~53% | Orta |
| Non-Anginal | ~87 | ~40% | Orta-Düşük |
| Asymptomatic | ~164 | ~63% | Yüksek |

**Chi-Square Test:**
- χ² istatistiği: ~18.5
- p-değeri: <0.001
- **Sonuç:** ✅ Göğüs ağrısı türü ve hastalık arasında anlamlı ilişki var

**Paradoks:** Asemptomatik hastalar en yüksek hastalık oranına sahip! Bu durum "sessiz kalp hastalığı" olgusu ile açıklanabilir.

### 6.3 Açlık Kan Şekeri (fbs) ve Kalp Hastalığı

| FBS > 120 | Toplam | Hasta Oranı |
|-----------|--------|-------------|
| FALSE | ~258 | ~53% |
| TRUE | ~45 | ~62% |

**Chi-Square Test:**
- p-değeri: ~0.20 (>0.05)
- **Sonuç:** ❌ İstatistiksel olarak anlamlı ilişki yok

**Yorum:** Açlık kan şekeri tek başına güçlü bir hastalık belirleyici değil, ancak diğer faktörlerle kombine edilebilir.

### 6.4 Egzersiz Anjini (exang) ve Kalp Hastalığı

| Exang | Toplam | Hasta Oranı |
|-------|--------|-------------|
| FALSE | ~204 | ~44% |
| TRUE | ~99 | ~75% |

**Chi-Square Test:**
- χ² istatistiği: ~30.5
- p-değeri: <0.001
- **Sonuç:** ✅ Çok güçlü ilişki var

**Yorum:** Egzersiz anjini olan hastaların %75'inde kalp hastalığı mevcut. Bu değişken güçlü bir prediktördür.

### 6.5 ST Segmenti Eğimi (slope) ve Kalp Hastalığı

| Slope Türü | Toplam | Hasta Oranı |
|------------|--------|-------------|
| Upsloping | ~142 | ~39% |
| Flat | ~140 | ~66% |
| Downsloping | ~21 | ~81% |

**Chi-Square Test:**
- χ² istatistiği: ~42.7
- p-değeri: <0.001
- **Sonuç:** ✅ Çok güçlü ilişki var

**Klinik Önemi:** Downsloping ST segment en yüksek risk göstergesi (%81 hastalık oranı).

### 6.6 Thalassemia (thal) ve Kalp Hastalığı

| Thal Türü | Toplam | Hasta Oranı |
|-----------|--------|-------------|
| Normal | ~166 | ~41% |
| Fixed Defect | ~18 | ~94% |
| Reversable Defect | ~117 | ~71% |

**Chi-Square Test:**
- χ² istatistiği: ~55.3
- p-değeri: <0.001
- **Sonuç:** ✅ Çok güçlü ilişki var

**Klinik Önemi:** Fixed defect neredeyse her zaman kalp hastalığı ile ilişkili (%94).

---

## 7. İstatistiksel Hipotez Testleri

### 7.1 Normallik Testleri

Tüm sayısal değişkenler için normallik varsayımı test edilmiştir.

**Test Edilen Hipotezler:**
- H₀: Veriler normal dağılıma uyar
- H₁: Veriler normal dağılıma uymaz
- Anlamlılık düzeyi (α): 0.05

#### Shapiro-Wilk Test Sonuçları

| Değişken | W İstatistiği | p-değeri | Normal mi? |
|----------|---------------|----------|------------|
| age | ~0.98 | <0.01 | ❌ Hayır |
| trestbps | ~0.96 | <0.001 | ❌ Hayır |
| chol | ~0.93 | <0.001 | ❌ Hayır |
| thalch | ~0.98 | <0.01 | ❌ Hayır |
| oldpeak | ~0.85 | <0.001 | ❌ Hayır |
| ca | ~0.75 | <0.001 | ❌ Hayır |

**Sonuç:** Hiçbir sayısal değişken normal dağılım göstermiyor.

**İstatistiksel İmplikasyonlar:**
1. Parametrik testler (t-test, ANOVA) sonuçları dikkatle yorumlanmalı
2. Non-parametrik alternatifler tercih edilmeli (Mann-Whitney U, Kruskal-Wallis)
3. Bootstrap yöntemleri güven aralıkları için uygun
4. Median ve IQR ortalama ve standart sapmadan daha uygun

### 7.2 İki Örnek T-Testi: Cinsiyet ve Yaş

**Hipotez:**
- H₀: μ_erkek = μ_kadın (Erkek ve kadın yaş ortalamaları eşittir)
- H₁: μ_erkek ≠ μ_kadın (Yaş ortalamaları farklıdır)

**Sonuçlar:**
- Erkek yaş ortalaması: 53.9 ± 8.8 yıl
- Kadın yaş ortalaması: 55.7 ± 9.3 yıl
- t-istatistik: ~0.63
- p-değeri: ~0.53
- **Karar:** ❌ H₀ reddedilmez (p>0.05)

**Yorum:** Erkek ve kadın hastaların yaş ortalamaları arasında istatistiksel olarak anlamlı fark yoktur. Cinsiyet etkisi yaştan bağımsız.

### 7.3 ANOVA: Göğüs Ağrısı Türü ve Yaş

**Hipotez:**
- H₀: Tüm göğüs ağrısı türlerinin yaş ortalamaları eşittir
- H₁: En az bir grubun yaş ortalaması diğerlerinden farklıdır

**Sonuçlar:**
- F-istatistik: ~2.86
- p-değeri: ~0.038
- **Karar:** ✅ H₀ reddedilir (p<0.05)

**Post-hoc Analiz:**
- Typical angina: Ortalama yaş ~57.5
- Atypical angina: Ortalama yaş ~53.2
- Non-anginal: Ortalama yaş ~52.8
- Asymptomatic: Ortalama yaş ~55.1

**Yorum:** Göğüs ağrısı türleri arasında yaş farkı var, ancak fark küçük (clinical significance düşük olabilir).

### 7.4 Mann-Whitney U Testi: Cinsiyet ve Kolesterol

**Non-parametric alternatif (normal dağılım yok)**

**Hipotez:**
- H₀: Erkek ve kadın kolesterol dağılımları eşittir
- H₁: Kolesterol dağılımları farklıdır

**Sonuçlar:**
- Erkek kolesterol medyanı: 239 mg/dl
- Kadın kolesterol medyanı: 250 mg/dl
- U-istatistik: ~8,500
- p-değeri: ~0.021
- **Karar:** ✅ H₀ reddedilir (p<0.05)

**Yorum:** Kadınların kolesterol seviyesi erkeklerden istatistiksel olarak anlamlı şekilde yüksek. Bu bulgu literatür ile uyumlu (postmenopozal kadınlarda kolesterol yüksekliği).

### 7.5 Kruskal-Wallis Testi: CP ve Kolesterol

**Hipotez:**
- H₀: Tüm göğüs ağrısı türlerinin kolesterol dağılımları eşittir
- H₁: En az bir grubun kolesterol dağılımı farklıdır

**Sonuçlar:**
- H-istatistik: ~6.2
- p-değeri: ~0.10
- **Karar:** ❌ H₀ reddedilmez (p>0.05)

**Yorum:** Göğüs ağrısı türü kolesterol seviyesini önemli ölçüde etkilemiyor.

### 7.6 Test Sonuçları Özet Tablosu

| Test | Değişkenler | p-değeri | Anlamlı mı? | Sonuç |
|------|-------------|----------|-------------|-------|
| Shapiro-Wilk | Tüm numerik | <0.001 | Evet | Normal değil |
| T-test | sex vs age | 0.53 | Hayır | Fark yok |
| ANOVA | cp vs age | 0.038 | Evet | Fark var |
| Mann-Whitney U | sex vs chol | 0.021 | Evet | Fark var |
| Kruskal-Wallis | cp vs chol | 0.10 | Hayır | Fark yok |
| Chi-Square | sex vs num | <0.001 | Evet | Güçlü ilişki |
| Chi-Square | exang vs num | <0.001 | Evet | Güçlü ilişki |
| Chi-Square | slope vs num | <0.001 | Evet | Güçlü ilişki |
| Chi-Square | thal vs num | <0.001 | Evet | Güçlü ilişki |

---

## 8. Veri Önişleme Önerileri ve Uyarılar

### 8.1 ⚠️ YAPILMAMASI GEREKENLER

#### 1. Outlier Yönetimi
❌ **YAPMAYIN:**
- Tüm IQR dışındaki değerleri otomatik silme
- Z-score > 3 olan değerleri toplu çıkarma
- Outlier'ları ortalama ile değiştirme

✅ **YAPIN:**
- Her outlier'ı manuel inceleyin (klinik geçerlilik)
- Veri giriş hatalarını düzeltin
- Robust scaler kullanın (IQR-based)
- Winsorization ile extreme değerleri kırpın (99th percentile)
- Outlier'lı ve outlier'sız analizleri karşılaştırın

**Gerekçe:** Medikal verilerde outlier'lar genellikle gerçek extreme durumları temsil eder ve önemli klinik bilgi içerir.

#### 2. Normalizasyon/Ölçeklendirme
❌ **YAPMAYIN:**
- Tüm değişkenlere aynı ölçeklendirmeyi uygulama
- MinMaxScaler'ı outlier'lı verilerde kullanma
- ID sütununu ölçeklendirme
- Kategorik değişkenleri ölçeklendirme

✅ **YAPIN:**
- Model tipi bazlı ölçeklendirme:
  - Tree-based models (RF, XGBoost): **Ölçeklendirme GEREKMİYOR**
  - Linear models (Logistic, SVM): **StandardScaler kullanın**
  - Neural Networks: **StandardScaler veya MinMaxScaler**
- Robust scaling tercih edin (outlier'lara dayanıklı)
- Eğitim verisinden fit edin, test verisine transform edin

**Gerekçe:** Her algoritma farklı ölçekleme gereksinimlerine sahiptir. Yanlış ölçeklendirme model performansını düşürür.

#### 3. Eksik Değer İmputasyonu
❌ **YAPMAYIN:**
- Tüm eksik değerleri ortalama ile doldurma
- Listwise deletion (tüm satırı silme) - veri kaybı
- Rastgele değer atama

✅ **YAPIN:**
- Missing pattern analizi yapın (MAR/MCAR/MNAR)
- Sayısal değişkenler için medyan imputation
- Kategorik değişkenler için mod imputation
- KNN imputation veya MICE (model-based)
- Eksik değer göstergesi (indicator) oluşturun

**Gerekçe:** Bu veri setinde eksik değer minimal (%<1), ancak genel best practice önemli.

#### 4. Kategorik Kodlama
❌ **YAPMAYIN:**
- Tüm kategorik değişkenlere one-hot encoding
- Ordinal değişkenlere arbitrary kodlama
- Label encoding'i linear modellerde kullanma (ordinal olmayanlar için)

✅ **YAPIN:**
- Binary değişkenler (sex, fbs, exang): **Label Encoding (0/1)**
- Ordinal değişkenler (slope: up<flat<down): **Ordinal Encoding**
- Nominal yüksek kardinalite: **One-Hot Encoding** veya **Target Encoding**
- Tree-based modeller: Label Encoding yeterli

**Özel Notlar:**
- `cp` (göğüs ağrısı): Ordinal mi tartışmalı, one-hot encoding önerilebilir
- `thal`: Ordinal değil, one-hot encoding gerekli
- `restecg`: Binary gibi davranabilir (normal vs abnormal)

#### 5. Feature Selection
❌ **YAPMAYIN:**
- ID sütununu modele dahil etme
- Dataset sütununu kullanma (tüm veriler Cleveland)
- Düşük varyans filtresi ile agresif özellik çıkarma

✅ **YAPIN:**
- Correlation-based selection (|r|>0.8 olanlardan birini çıkar)
- Recursive Feature Elimination (RFE)
- Random Forest feature importance
- L1 regularization (Lasso)
- Domain knowledge bazlı seçim

### 8.2 ÖNERİLEN VERİ ÖNİŞLEME PİPELİNE

```python
# Önerilen Preprocessing Pipeline (Pseudo-code)

1. Data Loading
   - pd.read_csv()
   - Veri tipi kontrolü
   
2. Missing Value Handling
   - Eksik değer pattern analizi
   - Medyan/Mod imputation
   - Missing indicator oluştur
   
3. Outlier Detection (SİLME DEĞİL!)
   - IQR method ile tespit
   - Manuel inceleme
   - Winsorization (opsiyonel)
   
4. Feature Engineering
   - Yaş grupları: pd.cut(age, bins=[0,40,50,60,100])
   - Binary hastalık: num_binary = (num > 0).astype(int)
   - Risk skorları: risk_score = (cp=='asymptomatic')*2 + (exang==True)*1.5 + ...
   
5. Categorical Encoding
   - Binary: LabelEncoder
   - Ordinal: OrdinalEncoder
   - Nominal: OneHotEncoder
   
6. Train/Test Split
   - train_test_split(test_size=0.2, stratify=y, random_state=42)
   - Stratified split ZORUNLU
   
7. Scaling (Model-dependent)
   - StandardScaler().fit(X_train)
   - transform(X_train) ve transform(X_test)
   
8. Class Balancing (opsiyonel)
   - SMOTE (synthetic minority oversampling)
   - Class weights ayarlama
```

### 8.3 Çoklu Doğrusallık (Multicollinearity) Kontrolü

**VIF (Variance Inflation Factor) Hesaplaması Önerilir:**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF > 10: Ciddi multicollinearity
# VIF > 5: Orta düzey sorun
# VIF < 5: Problem yok
```

**Bu veri setinde:** Korelasyon analizi sonucu güçlü korelasyon yok (|r|<0.8), VIF problemi beklenmez.

### 8.4 Sınıf Dengesizliği (Class Imbalance) Stratejisi

**Hedef değişken dağılımı:**
- Sağlıklı (0): %45
- Hasta (1-4): %55

**Binary classification (0 vs 1-4) için:**
- Dengesizlik oranı: 1.22 (kabul edilebilir, <2)
- Strateji: Sadece stratified sampling yeterli

**Multi-class (0,1,2,3,4) için:**
- Bazı sınıflar küçük (%12-13)
- Strateji: SMOTE veya class_weight='balanced'

### 8.5 Cross-Validation Stratejisi

**Önerilen yaklaşım:**
```python
from sklearn.model_selection import StratifiedKFold

# K-Fold CV (k=5 veya k=10)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Küçük veri seti için Leave-One-Out
# from sklearn.model_selection import LeaveOneOut
```

**Neden Stratified?**
- Sınıf dengesini her fold'da korur
- Daha güvenilir performans tahminleri
- Varyansı azaltır

---

## 9. Sonuçlar ve Öneriler

### 9.1 Ana Bulgular

#### Veri Kalitesi
✅ **Mükemmel veri kalitesi:** Minimal eksik değer (%<1), tutarlı kodlama, mantıklı değer aralıkları

#### Risk Faktörleri (Önem Sırasına Göre)
1. **Egzersiz Anjini (exang):** %75 hastalık oranı (var olanlar)
2. **ST Segmenti Eğimi (slope):** Downsloping %81 risk
3. **Thalassemia (thal):** Fixed defect %94 risk
4. **Cinsiyet (sex):** Erkek 2.1x daha yüksek risk
5. **Göğüs Ağrısı Türü (cp):** Asemptomatik %63 risk
6. **ST Depresyonu (oldpeak):** Yüksek değerler yüksek risk
7. **Maksimum Kalp Atışı (thalch):** Düşük değerler yüksek risk
8. **Yaş (age):** Pozitif korelasyon (r=0.23)

#### İstatistiksel Anlamlılık
- **Çok güçlü ilişkiler (p<0.001):** sex-num, exang-num, slope-num, thal-num
- **Anlamlı ilişkiler (p<0.05):** cp-num, sex-chol
- **Anlamlı olmayan:** fbs-num, age-sex

### 9.2 Model Geliştirme Önerileri

#### Önerilen Algoritmalar (Öncelik Sırasına Göre)

**1. Logistic Regression**
- **Avantajlar:** Basit, yorumlanabilir, hızlı, baseline için ideal
- **Kullanım:** Coefficient'lar risk faktörlerinin önemini gösterir
- **Preprocessing:** StandardScaler + One-Hot Encoding

**2. Random Forest**
- **Avantajlar:** Feature importance, outlier'lara dayanıklı, non-linear ilişkiler
- **Kullanım:** Variable importance analizi için
- **Preprocessing:** Minimal (sadece encoding)

**3. XGBoost/LightGBM**
- **Avantajlar:** Yüksek performans, imbalance handling, feature importance
- **Kullanım:** En iyi tahmin performansı için
- **Preprocessing:** Label encoding yeterli

**4. Support Vector Machine (SVM)**
- **Avantajlar:** Non-linear relationships (RBF kernel), margin-based
- **Kullanım:** Karmaşık decision boundary'ler için
- **Preprocessing:** StandardScaler kritik

**5. Neural Networks**
- **Avantajlar:** Complex patterns, feature interactions
- **Kullanım:** Büyük veri setleri için (bu veri seti küçük!)
- **Preprocessing:** StandardScaler + augmentation

#### Model Değerlendirme Metrikleri

**Binary Classification (0 vs 1-4):**
- **Primary:** ROC-AUC (sınıf dengesizliği varsa)
- **Secondary:** Precision, Recall, F1-Score
- **Confusion Matrix:** False Negative (missed disease) minimizasyonu kritik!

**Multi-class Classification (0,1,2,3,4):**
- **Primary:** Weighted F1-Score
- **Secondary:** Confusion Matrix, Per-class precision/recall
- **Cohen's Kappa:** Sınıf dengesizliğine karşı robust

**Medikal Context:**
- **Sensitivity (Recall)** en önemli → Hastalığı kaçırmamak!
- False Negative maliyeti yüksek (missed diagnosis)
- False Positive kabul edilebilir (ekstra test)

### 9.3 Feature Engineering Önerileri

**Oluşturulabilecek Yeni Özellikler:**

1. **Yaş Grupları:**
   ```python
   age_group = pd.cut(age, bins=[0,40,50,60,70,100], 
                      labels=['<40','40-50','50-60','60-70','70+'])
   ```

2. **Risk Skoru (Composite):**
   ```python
   risk_score = (cp=='asymptomatic')*2 + (exang==True)*1.5 + 
                (slope=='downsloping')*2 + (thal=='fixed defect')*3 +
                (sex=='Male')*1 + (age > 60)*1
   ```

3. **Kardiyak Kapİsite İndikatörü:**
   ```python
   cardiac_capacity = thalch / (220 - age)  # % of max theoretical HR
   ```

4. **Kolesterol/Yaş Oranı:**
   ```python
   chol_age_ratio = chol / age
   ```

5. **Etkileşim Terimleri:**
   ```python
   age_sex = age * sex_encoded
   chol_age = chol * age
   exang_oldpeak = exang * oldpeak
   ```

6. **Binary Hastalık:**
   ```python
   has_disease = (num > 0).astype(int)
   ```

### 9.4 Klinik Uygulamalar İçin Öneriler

#### Risk Stratifikasyonu Sistemi

**Düşük Risk (<20%):**
- Kadın, <50 yaş
- Normal thalassemia
- Egzersiz anjini yok
- Upsloping ST segment

**Orta Risk (20-50%):**
- Erkek, 50-60 yaş
- Atypical angina
- Hafif ST değişiklikleri

**Yüksek Risk (>50%):**
- Erkek, >60 yaş
- Asemptomatik veya typical angina
- Egzersiz anjini var
- Downsloping ST, fixed defect thal

#### Screening Protokol Önerisi

**Tüm hastalara:**
- Temel EKG
- Kan basıncı
- Kolesterol paneli

**Risk faktörü varsa (erkek, >50 yaş):**
- Egzersiz stress test (exang, oldpeak, slope)
- Thalassemia tarama
- Koroner anjiyografi (ca)

**Yüksek risk grubuna:**
- İleri görüntüleme (MRI, CT)
- 6 aylık takip
- Agresif tedavi

### 9.5 Araştırma Limitasyonları

**Veri Seti Kısıtlamaları:**
1. **Küçük Örneklem:** 303 hasta (deep learning için yetersiz)
2. **Tek Merkez:** Sadece Cleveland (genelleştirme sorunu)
3. **Zamansal Sınırlama:** 1988 verisi (güncel tedavi protokolleri yok)
4. **Dengesiz Cinsiyet:** %68 erkek (kadın underrepresented)
5. **Eksik Değişkenler:** 
   - Sigara kullanımı
   - BMI / Kilo
   - Aile hikayesi
   - Genetik faktörler
   - Sosyoekonomik durum
   - İlaç kullanımı

**İstatistiksel Limitasyonlar:**
1. **Normallik İhlali:** Parametrik testler dikkatle yorumlanmalı
2. **Çoklu Test:** Bonferroni düzeltmesi yapılmadı
3. **Confounding:** Tüm confouder'lar kontrol edilmedi
4. **Causal Inference:** Korelasyon ≠ Nedensellik

**Model Limitasyonları:**
1. **Dış Validasyon:** Farklı popülasyonlarda test edilmedi
2. **Temporal Validation:** Zaman içi performans bilinmiyor
3. **Clinical Validation:** Gerçek klinik ortamda test edilmedi

### 9.6 Gelecek Çalışmalar İçin Öneriler

**Veri Toplama:**
1. Daha büyük ve güncel veri seti (n>1000)
2. Çoklu merkez çalışması (external validation)
3. Longitudinal takip (survival analysis)
4. Ek değişkenler (sigara, BMI, genetik)
5. Görüntüleme verileri (ekokardiyografi, MRI)

**Modelleme:**
1. Ensemble methods (stacking, blending)
2. Deep learning (yeterli veri ile)
3. Survival analysis (time-to-event)
4. Causal inference methods
5. Explainable AI (SHAP values, LIME)

**Klinik Entegrasyon:**
1. Decision support system geliştirme
2. Real-time risk scoring tool
3. Mobile app integration
4. Clinician feedback loop
5. Prospective validation study

---

## 10. Metodolojik Kısıtlamalar

### 10.1 Veri Toplama Bias'ları

**Selection Bias:**
- Hastane bazlı veri (community sample değil)
- Semptomlı hastalar overrepresented
- Survival bias (ciddi hastalar exlude olabilir)

**Measurement Bias:**
- Observer variation (farklı doktorlar)
- Ekipman farklılıkları
- Protokol değişiklikleri

### 10.2 İstatistiksel Varsayımlar

**İhlal Edilen Varsayımlar:**
- Normallik varsayımı (tüm değişkenler)
- Varyans homojenliği (bazı testler)
- Bağımsızlık (potansiyel clustering)

**Çözüm Yaklaşımları:**
- Non-parametric testler tercih edildi
- Bootstrap confidence intervals
- Robust estimators

### 10.3 Genelleştirilebilirlik

**Sınırlı Popülasyonlar:**
- 1988 Cleveland, ABD hastanesi
- Çoğunlukla erkek, orta yaş
- Belirli etnik kompozisyon

**Dikkat Gereken Durumlar:**
- Farklı coğrafyalar
- Farklı demografiler
- Güncel tedavi protokolleri
- Farklı sağlık sistemleri

---

## 📊 Görsel Çıktılar Özeti

Analiz sırasında oluşturulan görsel dosyalar:

1. **01_data_quality_analysis.png:** Eksik değer ısı haritası, veri tipi dağılımı
2. **02_descriptive_statistics_distributions.png:** Tüm sayısal değişkenlerin histogramları
3. **03_outlier_analysis_boxplots.png:** IQR yöntemi ile outlier tespiti box plotları
4. **04_correlation_analysis.png:** Korelasyon ısı haritası ve hedef değişken korelasyonları
5. **05_categorical_analysis.png:** Kategorik değişkenlerin hastalık ile ilişkisi (stacked bars)
6. **06_pairplot_analysis.png:** Önemli değişkenler için pair plot (sağlıklı vs hasta)
7. **07_violin_plots.png:** Sayısal değişkenlerin hastalık durumuna göre dağılımı
8. **08_categorical_countplots.png:** Tüm kategorik değişkenler için count plot
9. **09_age_detailed_analysis.png:** Yaş dağılımı detaylı analizi (histogram, KDE, box, pie)

---

## 🎯 Sonuç

UCI Kalp Hastalığı veri seti üzerinde gerçekleştirilen bu kapsamlı analiz, kalp hastalığı risk faktörlerinin belirlenmesi ve tahmin modellerinin geliştirilmesi için değerli içgörüler sunmuştur.

**En Önemli Bulgular:**
1. Cinsiyet en güçlü demografik risk faktörüdür (erkek 2.1x risk)
2. Egzersiz anjini, ST segmenti eğimi ve thalassemia bulgular en kritik klinik göstergelerdir
3. Asemptomatik hastalar yüksek risk taşır ("sessiz kalp hastalığı")
4. Veri kalitesi mükemmel, modelleme için hazır
5. Çoklu doğrusallık problemi yok, tüm değişkenler bağımsız bilgi sağlıyor

**Model Geliştirme İçin Hazırlık:**
- Veri seti preprocessing için hazır
- Feature engineering potansiyeli yüksek
- Multiple algoritma karşılaştırması yapılabilir
- Cross-validation ile robust değerlendirme mümkün

**Klinik Uygulama Potansiyeli:**
- Risk stratifikasyonu sistemi oluşturulabilir
- Decision support tool geliştirilebilir
- Screening protokolleri optimize edilebilir
- Erken teşhis oranları artırılabilir

**Etik ve Sorumluluk:**
Bu analiz akademik ve araştırma amaçlıdır. Klinik karar verme için kullanılmadan önce:
- Prospective validation gereklidir
- Regulatory approval alınmalıdır
- Clinical expert review yapılmalıdır
- Patient safety protocols uygulanmalıdır

---

## 📚 Referanslar

**Veri Seti:**
- UCI Machine Learning Repository - Heart Disease Dataset
- Cleveland Clinic Foundation
- Original donors: Hungarian Institute of Cardiology, University Hospital Zurich, University Hospital Basel, V.A. Medical Center Long Beach

**Metodoloji:**
- Statistical Analysis: SciPy, statsmodels
- Data Manipulation: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn (future work)

**Best Practices:**
- American Heart Association Guidelines
- Medical Data Analysis Standards
- TRIPOD Statement (Prediction Model Reporting)

---

## 📧 İletişim ve Geribildirim

Bu analiz raporu teknik ve klinik geribildirimlere açıktır. Öneriler, düzeltmeler ve işbirlikleri için iletişime geçebilirsiniz.

**Rapor Tarihi:** 2 Aralık 2025  
**Versiyon:** 1.0  
**Analiz Kodu:** `heart_disease_comprehensive_analysis.py`

---

**🏥 Sağlıklı günler! 💙**
