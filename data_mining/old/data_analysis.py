# -*- coding: utf-8 -*-
"""
UCI Kalp Hastalığı Veri Seti - Kapsamlı Veri Analizi Raporu
===========================================================

Yazan: Veri Bilimci
Tarih: 26 Ekim 2025

Bu rapor UCI kalp hastalığı veri setinin kapsamlı analizini içermektedir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("UCI KALP HASTALIĞI VERİ SETİ - KAPSAMLI VERİ ANALİZİ")
print("="*80)

def load_and_explore_data(file_path):
    """
    Veri setini yükler ve temel keşif analizini yapar.
    
    Parameters:
    -----------
    file_path : str
        CSV dosyasının yolu
        
    Returns:
    --------
    pd.DataFrame
        Yüklenen veri seti
    """
    print("\n1. VERİ YÜKLEME VE İLK İNCELEME")
    print("-" * 40)
    
    # Veri setini yükle
    df = pd.read_csv(file_path)
    
    print(f"Veri seti boyutu: {df.shape}")
    print(f"Toplam kayıt sayısı: {df.shape[0]}")
    print(f"Özellik sayısı: {df.shape[1]}")
    
    print("\nVeri setinin ilk 5 satırı:")
    print(df.head())
    
    print("\nVeri setinin temel bilgileri:")
    print(df.info())
    
    print("\nVeri türleri:")
    print(df.dtypes)
    
    return df

def check_missing_values(df):
    """
    Eksik değerleri kontrol eder ve rapor eder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n2. EKSİK DEĞER ANALİZİ")
    print("-" * 40)
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Sütun': missing_values.index,
        'Eksik Değer Sayısı': missing_values.values,
        'Eksik Değer Yüzdesi (%)': missing_percentage.values
    }).sort_values('Eksik Değer Sayısı', ascending=False)
    
    print("Eksik değer raporu:")
    print(missing_df[missing_df['Eksik Değer Sayısı'] > 0])
    
    if missing_values.sum() == 0:
        print("✓ Veri setinde eksik değer bulunmuyor.")
    else:
        print(f"⚠ Toplam eksik değer sayısı: {missing_values.sum()}")
        
        # Eksik değerlerin görselleştirilmesi
        if missing_values.sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_cols = missing_values[missing_values > 0]
            plt.bar(missing_cols.index, missing_cols.values, color='red', alpha=0.7)
            plt.title('Sütunlara Göre Eksik Değer Sayıları')
            plt.xlabel('Sütunlar')
            plt.ylabel('Eksik Değer Sayısı')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def analyze_dataset_sources(df):
    """
    Veri setindeki farklı kaynaklarını (Cleveland, Hungary, etc.) analiz eder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n3. VERİ SETİ KAYNAKLARI ANALİZİ")
    print("-" * 40)
    
    if 'dataset' in df.columns:
        print("Veri seti kaynaklarının dağılımı:")
        dataset_counts = df['dataset'].value_counts()
        print(dataset_counts)
        
        # Veri kaynakları görselleştirmesi
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        dataset_counts.plot(kind='bar', color='lightblue', alpha=0.8)
        plt.title('Veri Seti Kaynaklarına Göre Dağılım')
        plt.xlabel('Veri Seti Kaynağı')
        plt.ylabel('Kayıt Sayısı')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
        plt.title('Veri Seti Kaynaklarının Yüzde Dağılımı')
        
        # Dataset'e göre kalp hastalığı dağılımı
        if 'num' in df.columns:
            df['heart_disease'] = (df['num'] > 0).astype(int)
            
            plt.subplot(2, 2, 3)
            cross_tab = pd.crosstab(df['dataset'], df['heart_disease'], normalize='index') * 100
            cross_tab.plot(kind='bar', stacked=False, color=['lightgreen', 'lightcoral'])
            plt.title('Veri Kaynağına Göre Kalp Hastalığı Oranları (%)')
            plt.xlabel('Veri Kaynağı')
            plt.ylabel('Oran (%)')
            plt.legend(['Sağlıklı', 'Hasta'])
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            # Dataset'e göre yaş ortalaması
            age_by_dataset = df.groupby('dataset')['age'].mean()
            age_by_dataset.plot(kind='bar', color='orange', alpha=0.8)
            plt.title('Veri Kaynağına Göre Ortalama Yaş')
            plt.xlabel('Veri Kaynağı')
            plt.ylabel('Ortalama Yaş')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("\nVeri kaynağına göre istatistikler:")
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            print(f"\n{dataset}:")
            print(f"  - Kayıt sayısı: {len(subset)}")
            print(f"  - Ortalama yaş: {subset['age'].mean():.1f}")
            if 'heart_disease' in df.columns:
                print(f"  - Kalp hastalığı oranı: %{(subset['heart_disease'].mean() * 100):.1f}")

def check_fields_to_remove(df):
    """
    Atılması gereken sütunları kontrol eder ve önerir.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n4. ALINMASI GEREKEN SÜTUN KONTROLÜ")
    print("-" * 40)
    
    recommendations = []
    
    # ID sütunu kontrolü
    if 'id' in df.columns:
        if df['id'].nunique() == len(df):
            recommendations.append(('id', 'Benzersiz ID sütunu - analiz için gereksiz'))
    
    # Tek değerli sütunlar
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
            recommendations.append((col, f'Tek değer içeriyor: {df[col].iloc[0]}'))
    
    # Yüksek eksik değer oranına sahip sütunlar
    high_missing_cols = []
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio > 0.5:  # %50'den fazla eksik
            high_missing_cols.append(col)
            recommendations.append((col, f'%{missing_ratio*100:.1f} eksik değer'))
    
    # Yüksek kardinalite (çok fazla benzersiz değer)
    high_cardinality_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        cardinality_ratio = df[col].nunique() / len(df)
        if cardinality_ratio > 0.8:  # %80'den fazla benzersiz değer
            high_cardinality_cols.append(col)
            recommendations.append((col, f'Yüksek kardinalite: {df[col].nunique()} benzersiz değer'))
    
    if recommendations:
        print("Atılması önerilen sütunlar:")
        for col, reason in recommendations:
            print(f"  - {col}: {reason}")
    else:
        print("✓ Atılması gereken sütun tespit edilmedi.")
    
    return [rec[0] for rec in recommendations]

def detect_outliers(df):
    """
    Aykırı değerleri tespit eder ve görselleştirir.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n5. AYKIRI DEĞER ANALİZİ")
    print("-" * 40)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'num', 'heart_disease']]
    
    # IQR yöntemi ile aykırı değer tespiti
    outlier_summary = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols[:6]):  # İlk 6 sayısal sütun
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # Box plot
        if i < len(axes):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col}\n{len(outliers)} aykırı değer (%{(len(outliers)/len(df)*100):.1f})')
            axes[i].set_ylabel(col)
    
    # Kullanılmayan subplot'ları gizle
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Aykırı değer raporu
    print("Aykırı değer raporu (IQR yöntemi):")
    print("-" * 50)
    for col, info in outlier_summary.items():
        print(f"{col}:")
        print(f"  - Aykırı değer sayısı: {info['count']}")
        print(f"  - Yüzdesi: %{info['percentage']:.1f}")
        print(f"  - Alt sınır: {info['lower_bound']:.2f}")
        print(f"  - Üst sınır: {info['upper_bound']:.2f}")
        print()

def analyze_gender_heart_disease_distribution(df):
    """
    Cinsiyete göre kalp hastalığı dağılımını analiz eder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n6. CİNSİYETE GÖRE KALP HASTALIĞI DAĞILIMI")
    print("-" * 40)
    
    if 'heart_disease' not in df.columns and 'num' in df.columns:
        df['heart_disease'] = (df['num'] > 0).astype(int)
    
    # Cinsiyet ve kalp hastalığı çapraz tablosu
    cross_tab = pd.crosstab(df['sex'], df['heart_disease'], margins=True)
    print("Cinsiyet ve kalp hastalığı çapraz tablosu:")
    print(cross_tab)
    
    # Yüzdelik dağılım
    cross_tab_pct = pd.crosstab(df['sex'], df['heart_disease'], normalize='index') * 100
    print("\nYüzdelik dağılım (satır bazında):")
    print(cross_tab_pct)
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mutlak sayılar
    cross_tab_counts = pd.crosstab(df['sex'], df['heart_disease'])
    cross_tab_counts.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
    axes[0].set_title('Cinsiyete Göre Kalp Hastalığı Sayıları')
    axes[0].set_xlabel('Cinsiyet')
    axes[0].set_ylabel('Kişi Sayısı')
    axes[0].legend(['Sağlıklı', 'Hasta'])
    axes[0].tick_params(axis='x', rotation=0)
    
    # Yüzdelik oranlar
    cross_tab_pct.plot(kind='bar', ax=axes[1], color=['lightblue', 'lightcoral'])
    axes[1].set_title('Cinsiyete Göre Kalp Hastalığı Oranları (%)')
    axes[1].set_xlabel('Cinsiyet')
    axes[1].set_ylabel('Oran (%)')
    axes[1].legend(['Sağlıklı', 'Hasta'])
    axes[1].tick_params(axis='x', rotation=0)
    
    # Stacked bar chart
    cross_tab_counts.plot(kind='bar', stacked=True, ax=axes[2], color=['lightblue', 'lightcoral'])
    axes[2].set_title('Cinsiyete Göre Kalp Hastalığı Dağılımı (Yığılmış)')
    axes[2].set_xlabel('Cinsiyet')
    axes[2].set_ylabel('Kişi Sayısı')
    axes[2].legend(['Sağlıklı', 'Hasta'])
    axes[2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()

def analyze_age_gender_distributions(df):
    """
    Yaş ve cinsiyet dağılımlarını analiz eder ve normal dağılıma uygunluğunu test eder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n7. YAŞ VE CİNSİYET DAĞILIMLARI - NORMAL DAĞILIM TESTİ")
    print("-" * 50)
    
    # Genel yaş dağılımı analizi
    print("Yaş dağılımı istatistikleri:")
    print(f"  - Ortalama: {df['age'].mean():.2f}")
    print(f"  - Medyan: {df['age'].median():.2f}")
    print(f"  - Standart sapma: {df['age'].std():.2f}")
    print(f"  - Minimum: {df['age'].min()}")
    print(f"  - Maksimum: {df['age'].max()}")
    print(f"  - Çeyreklikler: Q1={df['age'].quantile(0.25):.1f}, Q3={df['age'].quantile(0.75):.1f}")
    
    # Cinsiyet dağılımı
    print(f"\nCinsiyet dağılımı:")
    gender_counts = df['sex'].value_counts()
    gender_pct = df['sex'].value_counts(normalize=True) * 100
    for gender in gender_counts.index:
        print(f"  - {gender}: {gender_counts[gender]} kişi (%{gender_pct[gender]:.1f})")
    
    # Normal dağılım testleri
    print(f"\nNormal dağılım testleri:")
    print("H₀: Veriler normal dağılıma uyar")
    print("H₁: Veriler normal dağılıma uymaz")
    print("α = 0.05")
    
    # Genel yaş dağılımı
    shapiro_stat, shapiro_p = stats.shapiro(df['age'])
    jarque_stat, jarque_p = stats.jarque_bera(df['age'])
    
    print(f"\nGenel yaş dağılımı:")
    print(f"  - Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")
    print(f"  - Jarque-Bera: JB={jarque_stat:.4f}, p={jarque_p:.6f}")
    
    # Cinsiyete göre yaş dağılımı
    for gender in df['sex'].unique():
        age_subset = df[df['sex'] == gender]['age']
        shapiro_stat, shapiro_p = stats.shapiro(age_subset)
        print(f"\n{gender} yaş dağılımı:")
        print(f"  - Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")
        print(f"  - Ortalama: {age_subset.mean():.2f}")
        print(f"  - Standart sapma: {age_subset.std():.2f}")
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Genel yaş histogramı
    axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].set_title('Yaş Dağılımı (Genel)')
    axes[0, 0].set_xlabel('Yaş')
    axes[0, 0].set_ylabel('Yoğunluk')
    
    # Normal dağılım eğrisi ekleme
    mu, sigma = df['age'].mean(), df['age'].std()
    x = np.linspace(df['age'].min(), df['age'].max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal Dağılım')
    axes[0, 0].legend()
    
    # Q-Q plot
    stats.probplot(df['age'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Yaş)')
    
    # Cinsiyete göre yaş dağılımı
    for i, gender in enumerate(df['sex'].unique()):
        age_subset = df[df['sex'] == gender]['age']
        axes[0, 2].hist(age_subset, alpha=0.7, label=gender, bins=15, density=True)
    axes[0, 2].set_title('Cinsiyete Göre Yaş Dağılımı')
    axes[0, 2].set_xlabel('Yaş')
    axes[0, 2].set_ylabel('Yoğunluk')
    axes[0, 2].legend()
    
    # Box plot - cinsiyete göre yaş
    df.boxplot(column='age', by='sex', ax=axes[1, 0])
    axes[1, 0].set_title('Cinsiyete Göre Yaş Box Plot')
    axes[1, 0].set_xlabel('Cinsiyet')
    axes[1, 0].set_ylabel('Yaş')
    
    # Violin plot
    if 'heart_disease' in df.columns:
        # Kalp hastalığına göre yaş dağılımı
        axes[1, 1].violinplot([df[df['heart_disease']==0]['age'], 
                              df[df['heart_disease']==1]['age']], 
                             positions=[0, 1])
        axes[1, 1].set_title('Kalp Hastalığına Göre Yaş Dağılımı')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['Sağlıklı', 'Hasta'])
        axes[1, 1].set_ylabel('Yaş')
    
    # Yaş gruplarına göre dağılım
    age_groups = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '60+'])
    age_group_counts = age_groups.value_counts()
    axes[1, 2].pie(age_group_counts.values, labels=age_group_counts.index, autopct='%1.1f%%')
    axes[1, 2].set_title('Yaş Gruplarına Göre Dağılım')
    
    plt.tight_layout()
    plt.show()

# Ana analiz fonksiyonları
def descriptive_statistics(df):
    """
    Tanımlayıcı istatistikleri hesaplar ve görüntüler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n8. TANIMLAYICI İSTATİSTİKLER")
    print("-" * 40)

    print("Sayısal değişkenler için temel istatistikler:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())

    print("\nKategorik değişkenler için frekans analizi:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col} değişkeni dağılımı:")
        print(df[col].value_counts())

    # Target değişken analizi
    target_col = 'num'  # Kalp hastalığı hedef değişkeni
    if target_col in df.columns:
        print(f"\n\nHedef değişken ({target_col}) dağılımı:")
        target_counts = df[target_col].value_counts()
        print(target_counts)
        
        # Binary sınıflandırma için hedef değişkeni dönüştür
        df['heart_disease'] = (df[target_col] > 0).astype(int)
        print(f"\nBinary kalp hastalığı dağılımı:")
        print("0: Hastalık yok, 1: Hastalık var")
        print(df['heart_disease'].value_counts())

def correlation_analysis(df):
    """
    Özellikler arasındaki korelasyon analizini yapar ve ısı haritası oluşturur.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    """
    print("\n9. KORELASYON ANALİZİ")
    print("-" * 40)

    # Kategorik değişkenleri sayısal hale dönüştür (geçici olarak)
    df_temp = df.copy()
    label_encoders = {}
    
    for col in df_temp.select_dtypes(include=['object']).columns:
        if col != 'dataset':  # dataset sütununu koruyalım
            le = LabelEncoder()
            df_temp[col] = le.fit_transform(df_temp[col].astype(str))
            label_encoders[col] = le

    # Sayısal değişkenler için korelasyon matrisi
    numeric_df = df_temp.select_dtypes(include=[np.number])
    
    # ID sütununu çıkar
    if 'id' in numeric_df.columns:
        numeric_df = numeric_df.drop('id', axis=1)
    
    correlation_matrix = numeric_df.corr()

    # Korelasyon ısı haritası
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Korelasyon matrisi görselleştirmesi
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Kalp hastalığı ile en yüksek korelasyonlu değişkenler
    if 'heart_disease' in correlation_matrix.columns:
        heart_correlations = correlation_matrix['heart_disease'].abs().sort_values(ascending=False)
        print("Kalp hastalığı ile en yüksek korelasyonlu değişkenler:")
        print(heart_correlations.drop('heart_disease'))  # Kendisi hariç
        
        # En yüksek korelasyonlu ilk 10 özellik
        top_correlations = heart_correlations.drop('heart_disease').head(10)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'blue' for x in correlation_matrix['heart_disease'][top_correlations.index]]
        bars = plt.bar(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
        plt.title('Kalp Hastalığı ile En Yüksek Korelasyonlu Özellikler')
        plt.xlabel('Özellikler')
        plt.ylabel('Mutlak Korelasyon')
        plt.xticks(range(len(top_correlations)), top_correlations.index, rotation=45, ha='right')
        
        # Değerleri çubukların üzerine yaz
        for bar, value in zip(bars, top_correlations.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    # Güçlü korelasyonları (>0.5 veya <-0.5) raporla
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                strong_correlations.append({
                    'Değişken 1': correlation_matrix.columns[i],
                    'Değişken 2': correlation_matrix.columns[j],
                    'Korelasyon': corr_value
                })
    
    if strong_correlations:
        print(f"\nGüçlü korelasyonlar (|r| > 0.5):")
        strong_corr_df = pd.DataFrame(strong_correlations).sort_values('Korelasyon', key=abs, ascending=False)
        print(strong_corr_df)
    
    return correlation_matrix

def generate_final_report(df, test_results, model_results, clustering_results, pca_results, feature_importance):
    """
    Final analiz raporunu oluşturur ve yazdırır.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Orijinal veri seti
    test_results : dict
        İstatistiksel test sonuçları
    model_results : list
        Model performans sonuçları
    clustering_results : dict
        Kümeleme analizi sonuçları
    pca_results : dict
        PCA analizi sonuçları
    feature_importance : pd.DataFrame
        Özellik önemliliği sonuçları
    """
    print("\n" + "="*80)
    print("UCI KALP HASTALIĞI VERİ SETİ - FİNAL ANALİZ RAPORU")
    print("="*80)
    
    print("\n📊 VERİ SETİ ÖZETİ:")
    print(f"• Toplam kayıt sayısı: {df.shape[0]}")
    print(f"• Özellik sayısı: {df.shape[1]}")
    
    if 'heart_disease' in df.columns:
        print(f"• Kalp hastalığı olan kişi sayısı: {df['heart_disease'].sum()}")
        print(f"• Kalp hastalığı oranı: {df['heart_disease'].mean():.1%}")
    
    if 'dataset' in df.columns:
        print(f"• Veri kaynakları: {', '.join(df['dataset'].unique())}")
        
    print(f"• Yaş aralığı: {df['age'].min()}-{df['age'].max()} yıl")
    print(f"• Ortalama yaş: {df['age'].mean():.1f} yıl")
    
    # Cinsiyet dağılımı
    gender_dist = df['sex'].value_counts(normalize=True) * 100
    print(f"• Cinsiyet dağılımı: {gender_dist.to_dict()}")

    print("\n📈 İSTATİSTİKSEL BULGULAR:")
    
    # Normallik test sonuçları
    if 'normality' in test_results:
        normal_count = sum(1 for result in test_results['normality'].values() if result['result'] == 'Normal dağılım')
        total_tests = len(test_results['normality'])
        print(f"• Normallik testleri: {total_tests} testten {normal_count} tanesi normal dağılım gösterdi")
    
    # T-test sonuçları
    if 't_tests' in test_results:
        for test_name, result in test_results['t_tests'].items():
            print(f"• {test_name}: p={result['p_value']:.6f} - {result['result']}")
    
    # Chi-square sonuçları
    if 'chi_square' in test_results:
        for test_name, result in test_results['chi_square'].items():
            print(f"• {test_name} Chi-square: p={result['p_value']:.6f} - {result['result']}")
    
    # ANOVA sonuçları
    if 'anova' in test_results:
        for test_name, result in test_results['anova'].items():
            print(f"• {test_name} ANOVA: p={result['p_value']:.6f} - {result['result']}")

    print("\n🤖 MAKİNE ÖĞRENMESİ SONUÇLARI:")
    model_df = pd.DataFrame(model_results, columns=['Model', 'Doğruluk Oranı'])
    model_df = model_df.sort_values('Doğruluk Oranı', ascending=False)
    
    for _, row in model_df.iterrows():
        print(f"• {row['Model']}: {row['Doğruluk Oranı']:.3f}")
    
    print(f"\n🏆 En iyi performans: {model_df.iloc[0]['Model']} ({model_df.iloc[0]['Doğruluk Oranı']:.3f})")

    print("\n🎯 ÖNEMLİ ÖZELLİKLER:")
    print("En önemli özellikler (Random Forest modeline göre):")
    for i in range(min(8, len(feature_importance))):
        feature = feature_importance.iloc[i]
        print(f"• {feature['feature']}: {feature['importance']:.3f}")

    print("\n🔍 KÜMELEME ANALİZİ:")
    if clustering_results:
        print(f"• Optimal küme sayısı: {clustering_results['optimal_k']}")
        print(f"• Silhouette skoru: {clustering_results['silhouette_score']:.3f}")
        print(f"• Kümeleme kalitesi: {'İyi' if clustering_results['silhouette_score'] > 0.5 else 'Orta' if clustering_results['silhouette_score'] > 0.3 else 'Düşük'}")

    print(f"\n📉 PCA ANALİZİ:")
    if pca_results:
        print(f"• İlk 2 bileşenin açıkladığı varyans: %{sum(pca_results['explained_variance_ratio'][:2])*100:.1f}")
        print(f"• %95 varyans için gerekli bileşen: {pca_results['components_95']}")
        print(f"• Boyut indirgeme potansiyeli: {'Yüksek' if pca_results['components_95'] < len(pca_results['explained_variance_ratio'])/2 else 'Orta' if pca_results['components_95'] < len(pca_results['explained_variance_ratio'])*0.8 else 'Düşük'}")

    print("\n💡 ANA BULGULAR VE ÖNERİLER:")
    
    # Cinsiyet etkisi analizi
    if 'chi_square' in test_results and 'gender_heart_disease' in test_results['chi_square']:
        if test_results['chi_square']['gender_heart_disease']['p_value'] < 0.05:
            print("• Cinsiyet kalp hastalığı riski üzerinde anlamlı etkiye sahiptir")
            
            # Hangi cinsiyette risk daha yüksek?
            if 'heart_disease' in df.columns:
                male_risk = df[df['sex'] == 'Male']['heart_disease'].mean()
                female_risk = df[df['sex'] == 'Female']['heart_disease'].mean()
                
                if male_risk > female_risk:
                    print(f"  - Erkeklerde risk daha yüksek: %{male_risk*100:.1f} vs %{female_risk*100:.1f}")
                else:
                    print(f"  - Kadınlarda risk daha yüksek: %{female_risk*100:.1f} vs %{male_risk*100:.1f}")
    
    # Model önerileri
    best_model = model_df.iloc[0]['Model']
    if model_df.iloc[0]['Doğruluk Oranı'] > 0.85:
        print(f"• {best_model} modeli yüksek performans gösterdi (%{model_df.iloc[0]['Doğruluk Oranı']*100:.1f})")
        print("  - Klinik uygulamalar için umut verici")
    elif model_df.iloc[0]['Doğruluk Oranı'] > 0.75:
        print(f"• {best_model} modeli makul performans gösterdi")
        print("  - Daha fazla veri ve özellik mühendisliği ile iyileştirilebilir")
    
    # Veri kalitesi değerlendirmesi
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio < 0.01:
        print("• Veri kalitesi mükemmel - minimal eksik değer")
    elif missing_ratio < 0.05:
        print("• Veri kalitesi iyi - az eksik değer")
    else:
        print("• Veri kalitesi iyileştirilebilir - eksik değerler mevcut")

    print("\n📋 SONUÇ:")
    print("UCI kalp hastalığı veri seti üzerinde gerçekleştirilen kapsamlı analiz,")
    print("kalp hastalığı risk faktörlerinin belirlenmesi ve tahmin modellerinin")
    print("geliştirilmesi açısından değerli bulgular sunmuştur.")
    
    if model_df.iloc[0]['Doğruluk Oranı'] > 0.80:
        print(f"\n{best_model} modeli ile %{model_df.iloc[0]['Doğruluk Oranı']*100:.1f} doğruluk oranında")
        print("kalp hastalığı tahmini yapılabilmektedir.")
    
    print(f"\nEn kritik risk faktörleri arasında {feature_importance.iloc[0]['feature']},")
    print(f"{feature_importance.iloc[1]['feature']} ve {feature_importance.iloc[2]['feature']} bulunmaktadır.")
    
    print("\n" + "="*80)
    print("ANALİZ TAMAMLANDI - DETAYLI RAPOR İÇİN 'UCI_Heart_Disease_Analysis_Report.md' DOSYASINA BAKINIZ")
    print("="*80)

# Ana analiz akışı
def main_analysis():
    """
    Tüm analiz adımlarını sırasıyla çalıştırır.
    """
    print("Analiz başlatılıyor...")
    
    # 1. Veri yükleme ve keşif
    df = load_and_explore_data('heart_disease_uci.csv')
    
    # 2. Eksik değer kontrolü
    check_missing_values(df)
    
    # 3. Veri seti kaynakları analizi
    analyze_dataset_sources(df)
    
    # 4. Atılması gereken sütun kontrolü
    columns_to_remove = check_fields_to_remove(df)
    
    # 5. Aykırı değer analizi
    detect_outliers(df)
    
    # 6. Cinsiyet-kalp hastalığı analizi
    analyze_gender_heart_disease_distribution(df)
    
    # 7. Yaş ve cinsiyet dağılımları
    analyze_age_gender_distributions(df)
    
    # 8. Tanımlayıcı istatistikler
    descriptive_statistics(df)
    
    # 9. Korelasyon analizi
    correlation_matrix = correlation_analysis(df)
    
    # 10. Kapsamlı görselleştirme
    comprehensive_data_visualization(df)
    
    # 11. İstatistiksel testler
    test_results = statistical_tests(df)
    
    # 12. Gelişmiş görselleştirme
    advanced_visualization(df)
    
    # 13. Veri ön işleme
    df_processed, label_encoders = data_preprocessing(df)
    
    # 14. Makine öğrenmesi modelleri
    model_results, model_details, feature_importance = machine_learning_models(df_processed)
    
    # Veri setini ML için hazırla
    feature_cols = [col for col in df_processed.columns if col not in ['id', 'num', 'heart_disease', 'age_group']]
    X = df_processed[feature_cols]
    y = df_processed['heart_disease']
    
    # Eğitim verisi için split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 15. Kümeleme analizi
    clustering_results = clustering_analysis(X_train_scaled, y_train)
    
    # 16. PCA analizi
    pca_results = pca_analysis(X_train_scaled, y_train, clustering_results['cluster_labels'])
    
    # 17. Final rapor
    generate_final_report(df, test_results, model_results, clustering_results, pca_results, feature_importance)
    
    return {
        'original_data': df,
        'processed_data': df_processed,
        'correlation_matrix': correlation_matrix,
        'test_results': test_results,
        'model_results': model_results,
        'model_details': model_details,
        'feature_importance': feature_importance,
        'clustering_results': clustering_results,
        'pca_results': pca_results,
        'label_encoders': label_encoders
    }

# Analizi çalıştır
if __name__ == "__main__":
    print("🚀 UCI Kalp Hastalığı Veri Seti Analizi Başlatılıyor...")
    results = main_analysis()
    print("✅ Analiz başarıyla tamamlandı!")

def comprehensive_data_visualization(df):
    """
    Kapsamlı veri görselleştirmesi yapar.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Görselleştirilecek veri seti
    """
    print("\n10. KAPSAMLI VERİ GÖRSELLEŞTİRME")
    print("-" * 40)

    # Ana dağılım grafikleri
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Hedef değişken dağılımı
    if 'heart_disease' in df.columns:
        df['heart_disease'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Kalp Hastalığı Dağılımı')
        axes[0, 0].set_xlabel('0: Sağlıklı, 1: Hasta')
        axes[0, 0].set_ylabel('Kişi Sayısı')
        axes[0, 0].tick_params(axis='x', rotation=0)

    # Yaş dağılımı
    axes[0, 1].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Yaş Dağılımı')
    axes[0, 1].set_xlabel('Yaş')
    axes[0, 1].set_ylabel('Frekans')

    # Cinsiyet dağılımı
    df['sex'].value_counts().plot(kind='pie', ax=axes[0, 2], autopct='%1.1f%%', colors=['lightpink', 'lightblue'])
    axes[0, 2].set_title('Cinsiyet Dağılımı')

    # Göğüs ağrısı türü dağılımı
    df['cp'].value_counts().plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Göğüs Ağrısı Türü Dağılımı')
    axes[1, 0].set_xlabel('Göğüs Ağrısı Türü')
    axes[1, 0].set_ylabel('Frekans')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Kolesterol seviyesi dağılımı
    axes[1, 1].hist(df['chol'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('Kolesterol Seviyesi Dağılımı')
    axes[1, 1].set_xlabel('Kolesterol (mg/dl)')
    axes[1, 1].set_ylabel('Frekans')

    # Maksimum kalp atışı dağılımı
    axes[1, 2].hist(df['thalch'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_title('Maksimum Kalp Atışı Dağılımı')
    axes[1, 2].set_xlabel('Kalp Atışı (bpm)')
    axes[1, 2].set_ylabel('Frekans')

    plt.tight_layout()
    plt.show()

    # İkinci set görselleştirmeler - Klinik parametreler
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Kan basıncı dağılımı
    axes[0, 0].hist(df['trestbps'], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_title('Dinlenme Kan Basıncı Dağılımı')
    axes[0, 0].set_xlabel('Kan Basıncı (mmHg)')
    axes[0, 0].set_ylabel('Frekans')
    
    # ST depresyonu dağılımı
    axes[0, 1].hist(df['oldpeak'], bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[0, 1].set_title('ST Depresyonu Dağılımı')
    axes[0, 1].set_xlabel('ST Depresyonu')
    axes[0, 1].set_ylabel('Frekans')
    
    # Major damar sayısı
    if 'ca' in df.columns:
        df['ca'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='teal')
        axes[1, 0].set_title('Major Damar Sayısı Dağılımı')
        axes[1, 0].set_xlabel('Major Damar Sayısı')
        axes[1, 0].set_ylabel('Frekans')
        axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Thalassemia dağılımı
    if 'thal' in df.columns:
        df['thal'].value_counts().plot(kind='bar', ax=axes[1, 1], color='gold')
        axes[1, 1].set_title('Thalassemia Türü Dağılımı')
        axes[1, 1].set_xlabel('Thalassemia Türü')
        axes[1, 1].set_ylabel('Frekans')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

    print("✓ Kapsamlı görselleştirmeler oluşturuldu.")

def statistical_tests(df):
    """
    Kapsamlı istatistiksel testleri gerçekleştirir.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Test edilecek veri seti
        
    Returns:
    --------
    dict
        Test sonuçlarını içeren sözlük
    """
    print("\n11. İSTATİSTİKSEL TESTLER")
    print("-" * 40)
    
    test_results = {}

    # Normallik testleri (Shapiro-Wilk)
    print("Normallik Testleri (Shapiro-Wilk):")
    print("H₀: Veriler normal dağılıma uyar")
    print("H₁: Veriler normal dağılıma uymaz")
    print("α = 0.05")
    
    normality_results = {}
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    
    for col in numeric_cols:
        if col in df.columns:
            # Shapiro-Wilk testi
            statistic, p_value = stats.shapiro(df[col].dropna())
            result = "Normal dağılım" if p_value > 0.05 else "Normal dağılım değil"
            normality_results[col] = {'statistic': statistic, 'p_value': p_value, 'result': result}
            print(f"{col}: W={statistic:.4f}, p-value = {p_value:.6f} → {result}")
    
    test_results['normality'] = normality_results

    # T-testleri
    print("\n\nBağımsız T-Testleri:")
    t_test_results = {}
    
    # 1. Cinsiyete göre yaş farkı
    print("\n1. Cinsiyete göre yaş farkı:")
    male_ages = df[df['sex'] == 'Male']['age']
    female_ages = df[df['sex'] == 'Female']['age']
    t_stat, t_p_value = stats.ttest_ind(male_ages, female_ages)
    print(f"   T-istatistiği: {t_stat:.4f}")
    print(f"   P-değeri: {t_p_value:.6f}")
    print(f"   Erkek yaş ort: {male_ages.mean():.1f}, Kadın yaş ort: {female_ages.mean():.1f}")
    result = "Anlamlı fark var" if t_p_value < 0.05 else "Anlamlı fark yok"
    print(f"   Sonuç (α=0.05): {result}")
    t_test_results['gender_age'] = {'t_stat': t_stat, 'p_value': t_p_value, 'result': result}
    
    # 2. Kalp hastalığına göre yaş farkı
    if 'heart_disease' in df.columns:
        print("\n2. Kalp hastalığına göre yaş farkı:")
        healthy_ages = df[df['heart_disease'] == 0]['age']
        diseased_ages = df[df['heart_disease'] == 1]['age']
        t_stat2, t_p_value2 = stats.ttest_ind(healthy_ages, diseased_ages)
        print(f"   T-istatistiği: {t_stat2:.4f}")
        print(f"   P-değeri: {t_p_value2:.6f}")
        print(f"   Sağlıklı yaş ort: {healthy_ages.mean():.1f}, Hasta yaş ort: {diseased_ages.mean():.1f}")
        result2 = "Anlamlı fark var" if t_p_value2 < 0.05 else "Anlamlı fark yok"
        print(f"   Sonuç (α=0.05): {result2}")
        t_test_results['heart_disease_age'] = {'t_stat': t_stat2, 'p_value': t_p_value2, 'result': result2}
    
    test_results['t_tests'] = t_test_results

    # Chi-square testleri
    print("\n\nChi-square Bağımsızlık Testleri:")
    chi_square_results = {}
    
    # 1. Cinsiyet ve kalp hastalığı ilişkisi
    if 'heart_disease' in df.columns:
        print("\n1. Cinsiyet ve kalp hastalığı ilişkisi:")
        contingency_table = pd.crosstab(df['sex'], df['heart_disease'])
        chi2, chi_p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print("   Kontinjensi Tablosu:")
        print(contingency_table)
        print(f"   Chi-square: {chi2:.4f}")
        print(f"   P-değeri: {chi_p_value:.6f}")
        print(f"   Serbestlik derecesi: {dof}")
        result = "Bağımlı (ilişki var)" if chi_p_value < 0.05 else "Bağımsız (ilişki yok)"
        print(f"   Sonuç (α=0.05): {result}")
        chi_square_results['gender_heart_disease'] = {
            'chi2': chi2, 'p_value': chi_p_value, 'dof': dof, 'result': result,
            'contingency_table': contingency_table
        }
    
    # 2. Göğüs ağrısı türü ve kalp hastalığı ilişkisi
    if 'heart_disease' in df.columns:
        print("\n2. Göğüs ağrısı türü ve kalp hastalığı ilişkisi:")
        contingency_table2 = pd.crosstab(df['cp'], df['heart_disease'])
        chi2_2, chi_p_value2, dof2, expected2 = stats.chi2_contingency(contingency_table2)
        print("   Kontinjensi Tablosu:")
        print(contingency_table2)
        print(f"   Chi-square: {chi2_2:.4f}")
        print(f"   P-değeri: {chi_p_value2:.6f}")
        result2 = "Bağımlı (ilişki var)" if chi_p_value2 < 0.05 else "Bağımsız (ilişki yok)"
        print(f"   Sonuç (α=0.05): {result2}")
        chi_square_results['cp_heart_disease'] = {
            'chi2': chi2_2, 'p_value': chi_p_value2, 'dof': dof2, 'result': result2
        }
    
    test_results['chi_square'] = chi_square_results

    # ANOVA testleri
    print("\n\nTek Yönlü ANOVA Testleri:")
    anova_results = {}
    
    # 1. Göğüs ağrısı türüne göre yaş farkları
    print("\n1. Göğüs ağrısı türüne göre yaş farkları:")
    cp_groups = [df[df['cp'] == cp_type]['age'].dropna() for cp_type in df['cp'].unique()]
    f_stat, anova_p_value = stats.f_oneway(*cp_groups)
    print(f"   F-istatistiği: {f_stat:.4f}")
    print(f"   P-değeri: {anova_p_value:.6f}")
    result = "Gruplar arası anlamlı fark var" if anova_p_value < 0.05 else "Gruplar arası anlamlı fark yok"
    print(f"   Sonuç (α=0.05): {result}")
    anova_results['cp_age'] = {'f_stat': f_stat, 'p_value': anova_p_value, 'result': result}
    
    # 2. Veri seti kaynağına göre yaş farkları
    if 'dataset' in df.columns:
        print("\n2. Veri seti kaynağına göre yaş farkları:")
        dataset_groups = [df[df['dataset'] == dataset_type]['age'].dropna() for dataset_type in df['dataset'].unique()]
        f_stat2, anova_p_value2 = stats.f_oneway(*dataset_groups)
        print(f"   F-istatistiği: {f_stat2:.4f}")
        print(f"   P-değeri: {anova_p_value2:.6f}")
        result2 = "Gruplar arası anlamlı fark var" if anova_p_value2 < 0.05 else "Gruplar arası anlamlı fark yok"
        print(f"   Sonuç (α=0.05): {result2}")
        anova_results['dataset_age'] = {'f_stat': f_stat2, 'p_value': anova_p_value2, 'result': result2}
    
    test_results['anova'] = anova_results
    
    # Mann-Whitney U testleri (parametrik olmayan)
    print("\n\nMann-Whitney U Testleri (Parametrik Olmayan):")
    mann_whitney_results = {}
    
    if 'heart_disease' in df.columns:
        print("\n1. Kalp hastalığına göre kolesterol seviyesi farkı:")
        healthy_chol = df[df['heart_disease'] == 0]['chol'].dropna()
        diseased_chol = df[df['heart_disease'] == 1]['chol'].dropna()
        u_stat, u_p_value = stats.mannwhitneyu(healthy_chol, diseased_chol, alternative='two-sided')
        print(f"   U-istatistiği: {u_stat:.4f}")
        print(f"   P-değeri: {u_p_value:.6f}")
        print(f"   Sağlıklı kolesterol med: {healthy_chol.median():.1f}, Hasta kolesterol med: {diseased_chol.median():.1f}")
        result = "Anlamlı fark var" if u_p_value < 0.05 else "Anlamlı fark yok"
        print(f"   Sonuç (α=0.05): {result}")
        mann_whitney_results['heart_disease_cholesterol'] = {
            'u_stat': u_stat, 'p_value': u_p_value, 'result': result
        }
    
    test_results['mann_whitney'] = mann_whitney_results
    
    return test_results

def advanced_visualization(df):
    """
    Gelişmiş görselleştirmeler ve karşılaştırmalı analizler yapar.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Görselleştirilecek veri seti
    """
    print("\n12. GELIŞMIŞ GÖRSELLEŞTİRME VE KARŞILAŞTIRMALI ANALİZ")
    print("-" * 50)

    if 'heart_disease' not in df.columns and 'num' in df.columns:
        df['heart_disease'] = (df['num'] > 0).astype(int)

    # Çiftli değişken ilişkileri - Box plotlar
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Yaş vs Kalp hastalığı (Box plot)
    axes[0, 0].boxplot([df[df['heart_disease']==0]['age'], 
                        df[df['heart_disease']==1]['age']], 
                       labels=['Sağlıklı', 'Hasta'])
    axes[0, 0].set_title('Kalp Hastalığı Durumuna Göre Yaş Dağılımı')
    axes[0, 0].set_ylabel('Yaş')

    # Kolesterol vs Kalp hastalığı
    axes[0, 1].boxplot([df[df['heart_disease']==0]['chol'], 
                        df[df['heart_disease']==1]['chol']], 
                       labels=['Sağlıklı', 'Hasta'])
    axes[0, 1].set_title('Kalp Hastalığı Durumuna Göre Kolesterol Seviyesi')
    axes[0, 1].set_ylabel('Kolesterol (mg/dl)')

    # Maksimum kalp atışı vs Kalp hastalığı
    axes[1, 0].boxplot([df[df['heart_disease']==0]['thalch'], 
                        df[df['heart_disease']==1]['thalch']], 
                       labels=['Sağlıklı', 'Hasta'])
    axes[1, 0].set_title('Kalp Hastalığı Durumuna Göre Max Kalp Atışı')
    axes[1, 0].set_ylabel('Kalp Atışı (bpm)')

    # Dinlenme kan basıncı vs Kalp hastalığı
    axes[1, 1].boxplot([df[df['heart_disease']==0]['trestbps'], 
                        df[df['heart_disease']==1]['trestbps']], 
                       labels=['Sağlıklı', 'Hasta'])
    axes[1, 1].set_title('Kalp Hastalığı Durumuna Göre Dinlenme Kan Basıncı')
    axes[1, 1].set_ylabel('Kan Basıncı (mmHg)')

    plt.tight_layout()
    plt.show()

    # Kategorik değişkenler analizi
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Cinsiyet ve kalp hastalığı
    cross_tab = pd.crosstab(df['sex'], df['heart_disease'], normalize='index') * 100
    cross_tab.plot(kind='bar', ax=axes[0, 0], color=['lightblue', 'lightcoral'])
    axes[0, 0].set_title('Cinsiyete Göre Kalp Hastalığı Oranları (%)')
    axes[0, 0].set_xlabel('Cinsiyet')
    axes[0, 0].set_ylabel('Oran (%)')
    axes[0, 0].legend(['Sağlıklı', 'Hasta'])
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Göğüs ağrısı türü ve kalp hastalığı
    cp_cross_tab = pd.crosstab(df['cp'], df['heart_disease'], normalize='index') * 100
    cp_cross_tab.plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightcoral'])
    axes[0, 1].set_title('Göğüs Ağrısı Türüne Göre Kalp Hastalığı Oranları (%)')
    axes[0, 1].set_xlabel('Göğüs Ağrısı Türü')
    axes[0, 1].set_ylabel('Oran (%)')
    axes[0, 1].legend(['Sağlıklı', 'Hasta'])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Açlık kan şekeri ve kalp hastalığı
    fbs_cross_tab = pd.crosstab(df['fbs'], df['heart_disease'], normalize='index') * 100
    fbs_cross_tab.plot(kind='bar', ax=axes[0, 2], color=['lightblue', 'lightcoral'])
    axes[0, 2].set_title('Açlık Kan Şekerine Göre Kalp Hastalığı Oranları (%)')
    axes[0, 2].set_xlabel('Açlık Kan Şekeri > 120 mg/dl')
    axes[0, 2].set_ylabel('Oran (%)')
    axes[0, 2].legend(['Sağlıklı', 'Hasta'])
    axes[0, 2].tick_params(axis='x', rotation=0)
    
    # Egzersiz anjini ve kalp hastalığı
    exang_cross_tab = pd.crosstab(df['exang'], df['heart_disease'], normalize='index') * 100
    exang_cross_tab.plot(kind='bar', ax=axes[1, 0], color=['lightblue', 'lightcoral'])
    axes[1, 0].set_title('Egzersiz Anjinine Göre Kalp Hastalığı Oranları (%)')
    axes[1, 0].set_xlabel('Egzersiz Anjini')
    axes[1, 0].set_ylabel('Oran (%)')
    axes[1, 0].legend(['Sağlıklı', 'Hasta'])
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Yaş grupları ve kalp hastalığı
    df['age_group'] = pd.cut(df['age'], bins=[0, 45, 55, 65, 100], 
                            labels=['≤45', '46-55', '56-65', '>65'])
    age_group_cross_tab = pd.crosstab(df['age_group'], df['heart_disease'], normalize='index') * 100
    age_group_cross_tab.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'lightcoral'])
    axes[1, 1].set_title('Yaş Gruplarına Göre Kalp Hastalığı Oranları (%)')
    axes[1, 1].set_xlabel('Yaş Grubu')
    axes[1, 1].set_ylabel('Oran (%)')
    axes[1, 1].legend(['Sağlıklı', 'Hasta'])
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    # Veri seti kaynağına göre kalp hastalığı (eğer varsa)
    if 'dataset' in df.columns:
        dataset_cross_tab = pd.crosstab(df['dataset'], df['heart_disease'], normalize='index') * 100
        dataset_cross_tab.plot(kind='bar', ax=axes[1, 2], color=['lightblue', 'lightcoral'])
        axes[1, 2].set_title('Veri Kaynağına Göre Kalp Hastalığı Oranları (%)')
        axes[1, 2].set_xlabel('Veri Kaynağı')
        axes[1, 2].set_ylabel('Oran (%)')
        axes[1, 2].legend(['Sağlıklı', 'Hasta'])
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    print("✓ Gelişmiş görselleştirmeler ve karşılaştırmalı analizler oluşturuldu.")

def data_preprocessing(df):
    """
    Makine öğrenmesi için veri ön işleme yapar.
    
    Parameters:
    -----------
    df : pd.DataFrame
        İşlenecek veri seti
        
    Returns:
    --------
    tuple
        İşlenmiş veri seti ve encoder'lar
    """
    print("\n13. VERİ ÖN İŞLEME")
    print("-" * 40)

    # Kategorik değişkenleri sayısala dönüştürme
    df_processed = df.copy()

    # Hedef değişkeni oluştur
    if 'heart_disease' not in df_processed.columns and 'num' in df_processed.columns:
        df_processed['heart_disease'] = (df_processed['num'] > 0).astype(int)

    # Label encoding için kategorik sütunları belirle
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    label_encoders = {}
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            # Eksik değerleri önce doldur
            df_processed[col] = df_processed[col].fillna('missing')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"✓ {col} sütunu sayısala dönüştürüldü")

    # Eksik değerleri medyan/mod ile doldur
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if df_processed[col].dtype in ['int64', 'float64']:
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
                print(f"✓ {col} sütunundaki eksik değerler medyan ({median_value}) ile dolduruldu")
            else:
                mode_value = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_value, inplace=True)
                print(f"✓ {col} sütunundaki eksik değerler mod ({mode_value}) ile dolduruldu")
    
    print(f"\n✓ Veri ön işleme tamamlandı. Final boyut: {df_processed.shape}")
    
    return df_processed, label_encoders

def machine_learning_models(df_processed):
    """
    Çeşitli makine öğrenmesi modellerini eğitir ve karşılaştırır.
    
    Parameters:
    -----------
    df_processed : pd.DataFrame
        Ön işlemesi yapılmış veri seti
        
    Returns:
    --------
    tuple
        Model sonuçları, en iyi model ve özellik önemliliği
    """
    print("\n14. MAKİNE ÖĞRENMESİ MODELLERİ")
    print("-" * 40)

    # Özellik ve hedef değişkeni ayırma
    feature_cols = [col for col in df_processed.columns if col not in ['id', 'num', 'heart_disease', 'age_group']]
    X = df_processed[feature_cols]
    y = df_processed['heart_disease']

    print(f"Özellik sayısı: {X.shape[1]}")
    print(f"Kullanılan özellikler: {list(X.columns)}")

    # Veri setini eğitim ve test olarak ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    print(f"Sınıf dağılımı - Eğitim: {y_train.value_counts().to_dict()}")
    print(f"Sınıf dağılımı - Test: {y_test.value_counts().to_dict()}")

    # Özellik ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model performanslarını saklamak için liste
    model_results = []
    model_details = {}

    # 1. Logistic Regression
    print("\n14.1. LOJİSTİK REGRESYON")
    print("-" * 25)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_accuracy = accuracy_score(y_test, lr_pred)

    print(f"Doğruluk Oranı: {lr_accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    lr_report = classification_report(y_test, lr_pred, output_dict=True)
    print(classification_report(y_test, lr_pred))

    model_results.append(('Logistic Regression', lr_accuracy))
    model_details['Logistic Regression'] = {
        'model': lr_model, 'predictions': lr_pred, 'probabilities': lr_prob,
        'accuracy': lr_accuracy, 'report': lr_report
    }

    # 2. Random Forest
    print("\n14.2. RASTGELE ORMAN (RANDOM FOREST)")
    print("-" * 35)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print(f"Doğruluk Oranı: {rf_accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    print(classification_report(y_test, rf_pred))

    # Özellik önemliliği
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nÖzellik Önemliliği (Random Forest):")
    print(feature_importance.head(10))

    model_results.append(('Random Forest', rf_accuracy))
    model_details['Random Forest'] = {
        'model': rf_model, 'predictions': rf_pred, 'probabilities': rf_prob,
        'accuracy': rf_accuracy, 'report': rf_report, 'feature_importance': feature_importance
    }

    # 3. Support Vector Machine
    print("\n14.3. DESTEK VEKTÖR MAKİNESİ (SVM)")
    print("-" * 35)
    svm_model = SVC(kernel='rbf', random_state=42, probability=True, C=1.0, gamma='scale')
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    svm_accuracy = accuracy_score(y_test, svm_pred)

    print(f"Doğruluk Oranı: {svm_accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    svm_report = classification_report(y_test, svm_pred, output_dict=True)
    print(classification_report(y_test, svm_pred))

    model_results.append(('SVM', svm_accuracy))
    model_details['SVM'] = {
        'model': svm_model, 'predictions': svm_pred, 'probabilities': svm_prob,
        'accuracy': svm_accuracy, 'report': svm_report
    }

    # 4. Gradient Boosting (ek model)
    from sklearn.ensemble import GradientBoostingClassifier
    print("\n14.4. GRADIENT BOOSTING")
    print("-" * 25)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_prob = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_accuracy = accuracy_score(y_test, gb_pred)

    print(f"Doğruluk Oranı: {gb_accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    gb_report = classification_report(y_test, gb_pred, output_dict=True)
    print(classification_report(y_test, gb_pred))

    model_results.append(('Gradient Boosting', gb_accuracy))
    model_details['Gradient Boosting'] = {
        'model': gb_model, 'predictions': gb_pred, 'probabilities': gb_prob,
        'accuracy': gb_accuracy, 'report': gb_report
    }

    # Model performanslarının karşılaştırılması
    print("\n14.5. MODEL PERFORMANS KARŞILAŞTIRMASI")
    print("-" * 35)
    model_df = pd.DataFrame(model_results, columns=['Model', 'Doğruluk Oranı'])
    model_df = model_df.sort_values('Doğruluk Oranı', ascending=False)
    print(model_df)

    # Detaylı performans metrikleri
    print("\n14.6. DETAYLI PERFORMANS METRİKLERİ")
    print("-" * 35)
    
    detailed_metrics = []
    for model_name, details in model_details.items():
        report = details['report']
        detailed_metrics.append({
            'Model': model_name,
            'Accuracy': details['accuracy'],
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score'],
            'Macro Avg F1': report['macro avg']['f1-score']
        })
    
    detailed_df = pd.DataFrame(detailed_metrics).round(4)
    print(detailed_df)

    # Görselleştirmeler
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Model performans karşılaştırması
    model_df.plot(x='Model', y='Doğruluk Oranı', kind='bar', ax=axes[0, 0], 
                  color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], legend=False)
    axes[0, 0].set_title('Model Doğruluk Oranları Karşılaştırması')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Doğruluk Oranı')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # En iyi modelin karışıklık matrisi
    best_model_name = model_df.iloc[0]['Model']
    best_pred = model_details[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Sağlıklı', 'Hasta'],
                yticklabels=['Sağlıklı', 'Hasta'])
    axes[0, 1].set_title(f'Karışıklık Matrisi - {best_model_name}')
    axes[0, 1].set_xlabel('Tahmin Edilen')
    axes[0, 1].set_ylabel('Gerçek')

    # Özellik önemliliği (Random Forest)
    if 'Random Forest' in model_details:
        top_features = model_details['Random Forest']['feature_importance'].head(10)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='lightgreen')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Önem Skoru')
        axes[1, 0].set_title('En Önemli 10 Özellik (Random Forest)')
        axes[1, 0].invert_yaxis()

    # ROC Eğrileri karşılaştırması
    from sklearn.metrics import roc_curve, auc
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (model_name, details) in enumerate(model_details.items()):
        fpr, tpr, _ = roc_curve(y_test, details['probabilities'])
        roc_auc = auc(fpr, tpr)
        axes[1, 1].plot(fpr, tpr, color=colors[i], label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Eğrileri Karşılaştırması')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✓ Makine öğrenmesi modelleri tamamlandı.")
    
    return model_results, model_details, feature_importance

def clustering_analysis(X_train_scaled, y_train):
    """
    Kümeleme analizi yapar ve sonuçları görselleştirir.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Ölçeklendirilmiş eğitim verisi
    y_train : array-like
        Eğitim hedef değişkeni
        
    Returns:
    --------
    dict
        Kümeleme sonuçları
    """
    print("\n15. KÜMELEME ANALİZİ (CLUSTERING)")
    print("-" * 35)

    # K-Means kümeleme
    print("15.1. K-MEANS KÜMELEME")
    print("-" * 20)

    # Optimal küme sayısını bulma (Elbow Method)
    sse = []
    silhouette_scores = []
    k_range = range(2, 11)  # Silhouette için en az 2 küme gerekli
    
    from sklearn.metrics import silhouette_score

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_train_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_train_scaled, cluster_labels))

    # Elbow ve Silhouette grafikleri
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Elbow grafiği
    ax1.plot(k_range, sse, 'bo-')
    ax1.set_title('Elbow Method - Optimal Küme Sayısı')
    ax1.set_xlabel('Küme Sayısı (k)')
    ax1.set_ylabel('SSE (Sum of Squared Errors)')
    ax1.grid(True, alpha=0.3)

    # Silhouette skorları
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_title('Silhouette Skorları - Optimal Küme Sayısı')
    ax2.set_xlabel('Küme Sayısı (k)')
    ax2.set_ylabel('Silhouette Skoru')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Optimal küme sayısını belirle (en yüksek silhouette skoru)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal küme sayısı (Silhouette): {optimal_k}")
    print(f"En yüksek Silhouette skoru: {max(silhouette_scores):.3f}")

    # Final kümeleme
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_train_scaled)

    print(f"\nKüme dağılımı (k={optimal_k}):")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for i, count in enumerate(counts):
        print(f"Küme {i}: {count} kişi (%{count/len(cluster_labels)*100:.1f})")

    # Kümeleme kalitesi değerlendirmesi
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    if len(np.unique(y_train)) > 1:  # Eğer hedef değişkende birden fazla sınıf varsa
        ari_score = adjusted_rand_score(y_train, cluster_labels)
        nmi_score = normalized_mutual_info_score(y_train, cluster_labels)
        print(f"\nKümeleme kalitesi:")
        print(f"  - Adjusted Rand Index: {ari_score:.3f}")
        print(f"  - Normalized Mutual Information: {nmi_score:.3f}")

    # Küme merkezlerini analiz et
    cluster_centers = kmeans_final.cluster_centers_
    print(f"\nKüme merkezleri analizi:")
    for i, center in enumerate(cluster_centers):
        print(f"Küme {i} merkezi - Ortalama değerler:")
        print(f"  Boyut: {len(center)} özellik")

    clustering_results = {
        'optimal_k': optimal_k,
        'cluster_labels': cluster_labels,
        'silhouette_score': silhouette_scores[optimal_k-2],  # k_range 2'den başladığı için
        'cluster_centers': cluster_centers,
        'sse_scores': sse,
        'silhouette_scores': silhouette_scores
    }

    return clustering_results

def pca_analysis(X_train_scaled, y_train, cluster_labels=None):
    """
    Temel Bileşen Analizi (PCA) yapar ve görselleştirir.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Ölçeklendirilmiş eğitim verisi
    y_train : array-like
        Eğitim hedef değişkeni
    cluster_labels : array-like, optional
        Küme etiketleri
        
    Returns:
    --------
    dict
        PCA sonuçları
    """
    print("\n16. PCA ANALİZİ (TEMEL BİLEŞEN ANALİZİ)")
    print("-" * 40)

    # PCA uygulama
    pca = PCA()
    X_pca = pca.fit_transform(X_train_scaled)

    # Açıklanan varyans oranları
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # PCA istatistikleri
    print(f"Toplam bileşen sayısı: {len(explained_variance_ratio)}")
    print(f"İlk bileşenin açıkladığı varyans: %{explained_variance_ratio[0]*100:.1f}")
    print(f"İlk iki bileşenin açıkladığı varyans: %{sum(explained_variance_ratio[:2])*100:.1f}")
    
    # %95 varyansı açıklamak için gereken bileşen sayısı
    components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"%95 varyansı açıklamak için gereken bileşen sayısı: {components_95}")
    print(f"%90 varyansı açıklamak için gereken bileşen sayısı: {np.argmax(cumulative_variance_ratio >= 0.90) + 1}")

    # PCA görselleştirmesi
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Açıklanan varyans oranları
    axes[0, 0].bar(range(1, min(len(explained_variance_ratio) + 1, 11)), 
                   explained_variance_ratio[:10], alpha=0.7, color='skyblue')
    axes[0, 0].set_title('İlk 10 Bileşenin Açıkladığı Varyans Oranı')
    axes[0, 0].set_xlabel('Bileşen')
    axes[0, 0].set_ylabel('Açıklanan Varyans Oranı')
    axes[0, 0].grid(True, alpha=0.3)

    # Kümülatif açıklanan varyans
    axes[0, 1].plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='%95 Varyans')
    axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='%90 Varyans')
    axes[0, 1].set_title('Kümülatif Açıklanan Varyans Oranı')
    axes[0, 1].set_xlabel('Bileşen Sayısı')
    axes[0, 1].set_ylabel('Kümülatif Varyans Oranı')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # İlk 2 bileşenle görselleştirme
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_train_scaled)

    # Gerçek sınıflar ile PCA
    scatter1 = axes[1, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_train, 
                                  cmap='coolwarm', alpha=0.7, s=50)
    axes[1, 0].set_title('PCA - Gerçek Kalp Hastalığı Sınıfları')
    axes[1, 0].set_xlabel(f'1. Bileşen (%{pca_2d.explained_variance_ratio_[0]*100:.1f} varyans)')
    axes[1, 0].set_ylabel(f'2. Bileşen (%{pca_2d.explained_variance_ratio_[1]*100:.1f} varyans)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Colorbar ekle
    cbar1 = plt.colorbar(scatter1, ax=axes[1, 0])
    cbar1.set_ticks([0, 1])
    cbar1.set_ticklabels(['Sağlıklı', 'Hasta'])

    # Kümeleme sonuçları ile PCA (eğer varsa)
    if cluster_labels is not None:
        scatter2 = axes[1, 1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_labels, 
                                      cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_title('PCA - K-Means Kümeleme Sonuçları')
        axes[1, 1].set_xlabel(f'1. Bileşen (%{pca_2d.explained_variance_ratio_[0]*100:.1f} varyans)')
        axes[1, 1].set_ylabel(f'2. Bileşen (%{pca_2d.explained_variance_ratio_[1]*100:.1f} varyans)')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1, 1])
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Bileşen yükleri analizi (Component loadings)
    print(f"\nİlk 2 bileşenin özellik yükleri:")
    if hasattr(pca_2d, 'components_'):
        components_df = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=[f'Feature_{i}' for i in range(pca_2d.components_.shape[1])]
        )
        
        # En yüksek yüklere sahip özellikler
        pc1_top = components_df['PC1'].abs().nlargest(5)
        pc2_top = components_df['PC2'].abs().nlargest(5)
        
        print("PC1 için en önemli özellikler:")
        for feature, loading in pc1_top.items():
            sign = '+' if components_df.loc[feature, 'PC1'] > 0 else '-'
            print(f"  {feature}: {sign}{loading:.3f}")
            
        print("\nPC2 için en önemli özellikler:")
        for feature, loading in pc2_top.items():
            sign = '+' if components_df.loc[feature, 'PC2'] > 0 else '-'
            print(f"  {feature}: {sign}{loading:.3f}")

    pca_results = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'components_95': components_95,
        'pca_2d_data': X_pca_2d,
        'pca_model': pca,
        'pca_2d_model': pca_2d
    }

    return pca_results

