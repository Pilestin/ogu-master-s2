# 🎓 OGU Yüksek Lisans - 2. Dönem Çalışmaları

<div align="center">

![GitHub last commit](https://img.shields.io/github/last-commit/Pilestin/ogu-master-s2)
![GitHub repo size](https://img.shields.io/github/repo-size/Pilestin/ogu-master-s2)
![GitHub](https://img.shields.io/github/license/Pilestin/ogu-master-s2)

**Bilgisayar Mühendisliği Yüksek Lisans Programı**  
**Osmangazi Üniversitesi - 2. Dönem**

</div>

---

## 📖 Hakkında

Bu repo, Osmangazi Üniversitesi Bilgisayar Mühendisliği Yüksek Lisans Programı 2. dönem derslerinin kod geliştirmelerini, projeleri ve çalışmalarını içermektedir. Her ders için ayrı klasörler bulunmakta ve ilgili projelerin detaylı dokümantasyonu, kaynak kodları ve sonuçları bu repo üzerinde yönetilmektedir.

## 🎯 Amaç ve Kapsam

Bu repository'nin temel amaçları:
- 📚 Yüksek lisans derslerinde geliştirilen tüm kod projelerini merkezi bir yerde toplamak
- 🔬 Makine öğrenimi, veri madenciliği ve istatistiksel analiz konularında pratik deneyim kazanmak
- 📊 Gerçek dünya veri setleri üzerinde çalışarak akademik bilgiyi uygulamaya aktarmak
- 🚀 Modern veri bilimi ve makine öğrenimi araçlarını kullanarak proje geliştirmek
- 📝 Detaylı dokümantasyon ile projelerin anlaşılabilir ve tekrar üretilebilir olmasını sağlamak

## 📗 Dersler ve Projeler

### 🤖 Machine Learning and Anomaly Detection
**Öğretim Elemanı:** Doç. Dr. Eyüp ÇİNAR

**Proje:** Wood Anomaly Detection - Ahşap Yüzey Defekt Tespiti

Derin öğrenme tabanlı anomali tespit sistemi geliştirme. Proje, cold-start anomaly detection yaklaşımı kullanarak ahşap yüzeylerdeki defektleri tespit etmeyi amaçlamaktadır.

**Kullanılan Modeller:**
- AutoEncoder (Reconstruction-Based)
- PatchCore (Memory Bank + K-NN)
- SimpleNet (Feature Adaptor + Discriminator)
- EfficientAD (Student-Teacher + AutoEncoder)

**Teknik Özellikler:**
- ✅ Çoklu çözünürlük desteği (256×256, 512×512)
- ✅ Grid search ile hiperparametre optimizasyonu
- ✅ Kapsamlı veri ön işleme pipeline
- ✅ GPU hızlandırma desteği
- ✅ Otomatik model değerlendirme ve görselleştirme

[📂 Proje Detayları](/machine_learning_anomaly_detection) | [📄 Teknik Rapor](/machine_learning_anomaly_detection/readme.md)

---

### 💎 Data Mining
**Öğretim Elemanı:** Doç. Dr. Efnan ŞORA GÜNAL

**Proje:** UCI Heart Disease Dataset - Kalp Hastalığı Tahmini

UCI Machine Learning Repository'deki kalp hastalığı veri setini kullanarak kapsamlı makine öğrenmesi modelleme ve karşılaştırma çalışması.

**Çalışma Senaryoları:**
1. **Senaryo 1:** Temel makine öğrenmesi modellerinin baseline performans testi
2. **Senaryo 2:** SMOTE, Feature Selection ve Optuna ile optimize edilmiş modeller

**Kullanılan Modeller:**
- Logistic Regression
- Decision Tree & Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Gradient Boosting & AdaBoost
- XGBoost

**Teknik Özellikler:**
- ✅ İki farklı senaryo ile karşılaştırmalı analiz
- ✅ SMOTE ile veri dengeleme
- ✅ Optuna ile hiperparametre optimizasyonu
- ✅ Feature importance analizi
- ✅ Kapsamlı istatistiksel görselleştirme

[📂 Proje Detayları](/data_mining) | [📄 Proje Dokümantasyonu](/data_mining/README.md)

---

### 📊 Data Analysis and Statistics
**Öğretim Elemanı:** Dr. Öğr. Üyesi Sinem BOZKURT KESER

**Proje:** Online Shoppers Intention Dataset - İstatistiksel Analiz ve Veri Görselleştirme

Veri analizi ve istatistik teknikleri kullanılarak online alışveriş davranışlarının incelenmesi ve yorumlanması.

**Çalışma Konuları:**
- 📈 Tanımlayıcı istatistikler
- 📊 Veri görselleştirme teknikleri
- 🔍 Keşifsel veri analizi (EDA)
- 📉 İstatistiksel testler ve çıkarımlar

[📂 Proje Detayları](/data_analysis) | [📄 Proje Dokümantasyonu](/data_analysis/README.md)

---

## 🛠️ Genel Kurulum ve Gereksinimler

### Önkoşullar
- Python 3.8+
- pip veya conda package manager
- (Opsiyonel) CUDA destekli GPU (Machine Learning projesi için)

### Kurulum Adımları

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/Pilestin/ogu-master-s2.git
cd ogu-master-s2
```

2. **İlgili proje klasörüne gidin:**
```bash
cd machine_learning_anomaly_detection  # veya data_mining, data_analysis
```

3. **Gerekli bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Projeyi çalıştırın:**
Her projenin kendi README dosyasında detaylı çalıştırma talimatları bulunmaktadır.

> 💡 **Not:** Her proje klasörünün kendi `requirements.txt` dosyası bulunmaktadır. İlgili proje dizinindeki kurulum talimatlarını takip edin.

## 📁 Repository Yapısı

```
ogu-master-s2/
│
├── README.md                              # Ana dokümantasyon (bu dosya)
├── .gitignore                             # Git yapılandırması
│
├── machine_learning_anomaly_detection/    # ML & Anomaly Detection dersi
│   ├── readme.md                          # Proje teknik raporu
│   ├── compare_anomaly_models.py          # Ana deney framework
│   ├── data_processing_script.py          # Veri ön işleme
│   ├── gpu_test_code.py                   # GPU test kodu
│   ├── notebooks/                         # Jupyter notebook'lar
│   └── docs/                              # Ek dokümantasyon
│
├── data_mining/                           # Data Mining dersi
│   ├── README.md                          # Proje dokümantasyonu
│   ├── main.py                            # Ana analiz scripti
│   ├── main_2_new_technics.py             # Gelişmiş teknikler
│   ├── main_3_scenario_comparison.py      # Senaryo karşılaştırması
│   ├── data_analysis.ipynb                # Analiz notebook'u
│   ├── requirements.txt                   # Python bağımlılıkları
│   ├── data/                              # Veri setleri
│   ├── img/                               # Görselleştirmeler
│   └── docs/                              # Ek dokümantasyon
│
└── data_analysis/                         # Data Analysis dersi
    ├── README.md                          # Proje dokümantasyonu
    ├── main.py                            # Ana analiz scripti
    ├── preprocessing.py                   # Veri ön işleme
    ├── main_notebook.ipynb                # Ana notebook
    ├── requirements.txt                   # Python bağımlılıkları
    ├── data/                              # Veri setleri
    └── results/                           # Analiz sonuçları
```

## 🔧 Kullanılan Teknolojiler

### Programlama Dilleri
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

### Makine Öğrenimi & Deep Learning
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
- ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)

### Veri İşleme & Analiz
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
- ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)

### Görselleştirme
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
- ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat)

### Diğer Araçlar
- OpenCV, imbalanced-learn, Optuna

## 📊 Projelerden Örnekler

### Machine Learning - Anomaly Detection Heatmap Örnekleri
Ahşap yüzey defekt tespiti için oluşturulan anomali haritaları modellerin defektleri başarıyla lokalize edebildiğini göstermektedir.

### Data Mining - Model Karşılaştırma Sonuçları
İki farklı senaryo (baseline vs. optimized) ile çalışan modellerin performans karşılaştırmaları, SMOTE ve hiperparametre optimizasyonunun etkisini net bir şekilde ortaya koymaktadır.

### Data Analysis - İstatistiksel Görselleştirmeler
Online alışveriş veri seti üzerinde yapılan detaylı istatistiksel analizler ve görselleştirmeler.

## 🤝 Katkıda Bulunma

Bu bir akademik proje repository'sidir. Önerileriniz ve geri bildirimleriniz için:
1. Issue açabilirsiniz
2. Pull request gönderebilirsiniz
3. Doğrudan iletişime geçebilirsiniz

## 📄 Lisans

Bu proje eğitim amaçlıdır ve akademik kullanım içindir.

## 👤 İletişim

**Repository Sahibi:** Pilestin  
**Kurum:** Osmangazi Üniversitesi  
**Program:** Bilgisayar Mühendisliği Yüksek Lisans

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

Made with ❤️ for academic excellence

</div>
