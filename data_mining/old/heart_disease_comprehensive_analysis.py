"""
UCI Kalp Hastalığı Veri Seti - Kapsamlı İstatistiksel Analiz ve Görselleştirme
================================================================================

Bu script, UCI Heart Disease veri setinin detaylı analizini gerçekleştirir:
- Veri yapısı ve kalite analizi
- Tanımlayıcı istatistikler
- Outlier (aykırı değer) tespiti
- Korelasyon analizi
- Kategorik değişken analizleri
- İstatistiksel testler
- Kapsamlı görselleştirmeler

Tarih: Aralık 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, normaltest, kstest
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("UCI KALP HASTALIĞI VERİ SETİ - KAPSAMLI İSTATİSTİKSEL ANALİZ")
print("="*80)

# ============================================================================
# 1. VERİ YÜKLEME VE GENEL BAKIŞ
# ============================================================================

def load_and_explore_data(file_path):
    """
    Veri setini yükler ve temel bilgileri gösterir.
    """
    print("\n" + "="*80)
    print("1. VERİ YÜKLEME VE GENEL BAKIŞ")
    print("="*80)
    
    df = pd.read_csv(file_path)
    
    # Sadece Cleveland verisini filtrele
    if 'dataset' in df.columns:
        print(f"\n🔍 Veri seti filtreleme: Sadece Cleveland verileri kullanılıyor...")
        original_count = len(df)
        df = df[df['dataset'] == 'Cleveland'].copy()
        print(f"   • Orijinal kayıt sayısı: {original_count}")
        print(f"   • Cleveland kayıt sayısı: {len(df)}")
        print(f"   • Filtrelenen kayıt: {original_count - len(df)}")
    
    print(f"\n📊 Veri Seti Boyutları:")
    print(f"   • Toplam kayıt sayısı: {df.shape[0]}")
    print(f"   • Özellik sayısı: {df.shape[1]}")
    
    print(f"\n📋 İlk 10 Satır:")
    print(df.head(10))
    
    print(f"\n🔍 Veri Tipleri ve Genel Bilgi:")
    print(df.info())
    
    print(f"\n📈 Sütun Adları:")
    print(df.columns.tolist())
    
    return df

# ============================================================================
# 2. EKSİK DEĞER VE VERİ KALİTESİ ANALİZİ
# ============================================================================

def analyze_data_quality(df):
    """
    Eksik değerleri ve veri kalitesini analiz eder.
    """
    print("\n" + "="*80)
    print("2. VERİ KALİTESİ VE EKSİK DEĞER ANALİZİ")
    print("="*80)
    
    # Eksik değer analizi
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Sütun': missing_count.index,
        'Eksik Sayı': missing_count.values,
        'Eksik %': missing_percent.values
    }).sort_values('Eksik Sayı', ascending=False)
    
    print("\n🔍 Eksik Değer Raporu:")
    print(missing_df)
    
    total_missing = missing_count.sum()
    total_cells = df.shape[0] * df.shape[1]
    
    print(f"\n📊 Özet:")
    print(f"   • Toplam eksik değer: {total_missing}")
    print(f"   • Toplam hücre sayısı: {total_cells}")
    print(f"   • Eksik değer oranı: {(total_missing/total_cells)*100:.2f}%")
    
    # Veri tipi analizi
    print("\n📋 Veri Tipi Dağılımı:")
    print(f"   • Sayısal (numeric): {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   • Kategorik (object): {len(df.select_dtypes(include=['object']).columns)}")
    
    # Benzersiz değer sayıları
    print("\n🔢 Her Sütundaki Benzersiz Değer Sayıları:")
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_percent = (unique_count / len(df)) * 100
        print(f"   • {col:15s}: {unique_count:4d} benzersiz değer ({unique_percent:5.1f}%)")
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Eksik değer ısı haritası
    if missing_count.sum() > 0:
        missing_data = df.isnull()
        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Eksik Değer Isı Haritası', fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'Eksik değer yok!', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
        axes[0].set_title('Eksik Değer Analizi', fontsize=14, fontweight='bold')
    
    # Veri tipi dağılımı
    dtype_counts = df.dtypes.value_counts()
    axes[1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Veri Tipi Dağılımı', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('img/01_data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return missing_df

# ============================================================================
# 3. TANIMLAYICI İSTATİSTİKLER
# ============================================================================

def descriptive_statistics(df):
    """
    Detaylı tanımlayıcı istatistikler hesaplar.
    """
    print("\n" + "="*80)
    print("3. TANIMLAYICI İSTATİSTİKLER")
    print("="*80)
    
    # Sayısal değişkenler
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    print("\n📊 Sayısal Değişkenler - Temel İstatistikler:")
    stats_df = df[numeric_cols].describe()
    print(stats_df)
    
    # Ek istatistikler
    print("\n📈 Ek İstatistiksel Ölçümler:")
    additional_stats = pd.DataFrame({
        'Çarpıklık (Skewness)': df[numeric_cols].skew(),
        'Basıklık (Kurtosis)': df[numeric_cols].kurtosis(),
        'Medyan': df[numeric_cols].median(),
        'Mod': df[numeric_cols].mode().iloc[0] if len(df[numeric_cols].mode()) > 0 else np.nan,
        'Varyans': df[numeric_cols].var()
    })
    print(additional_stats)
    
    # Kategorik değişkenler
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    print("\n📋 Kategorik Değişkenler - Frekans Analizi:")
    for col in categorical_cols:
        print(f"\n{col}:")
        freq_table = df[col].value_counts()
        freq_percent = df[col].value_counts(normalize=True) * 100
        freq_df = pd.DataFrame({
            'Frekans': freq_table,
            'Yüzde (%)': freq_percent
        })
        print(freq_df)
    
    # Hedef değişken (num) analizi
    if 'num' in df.columns:
        print("\n🎯 Hedef Değişken (num - Kalp Hastalığı Derecesi) Dağılımı:")
        target_dist = df['num'].value_counts().sort_index()
        target_percent = df['num'].value_counts(normalize=True).sort_index() * 100
        target_df = pd.DataFrame({
            'Frekans': target_dist,
            'Yüzde (%)': target_percent
        })
        print(target_df)
    
    # Görselleştirme
    n_numeric = len(numeric_cols)
    n_rows = (n_numeric + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
    axes = axes.ravel() if n_numeric > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_title(f'{col} Dağılımı', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frekans')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Ortalama: {df[col].mean():.2f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Medyan: {df[col].median():.2f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Kullanılmayan subplot'ları gizle
    for idx in range(n_numeric, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('img/02_descriptive_statistics_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_df, additional_stats

# ============================================================================
# 4. OUTLIER (AYKIRI DEĞER) ANALİZİ
# ============================================================================

def detect_outliers(df):
    """
    IQR ve Z-score yöntemleri ile outlier tespiti yapar.
    """
    print("\n" + "="*80)
    print("4. OUTLIER (AYKIRI DEĞER) ANALİZİ")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    outlier_summary = {}
    
    print("\n🔍 IQR Yöntemi ile Outlier Tespiti:")
    print("-" * 80)
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(df)) * 100
        
        outlier_summary[col] = {
            'count': outlier_count,
            'percent': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
        
        print(f"\n{col}:")
        print(f"   • Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"   • Alt sınır: {lower_bound:.2f}, Üst sınır: {upper_bound:.2f}")
        print(f"   • Outlier sayısı: {outlier_count} ({outlier_percent:.1f}%)")
        if outlier_count > 0:
            print(f"   • Outlier değerleri: {outliers[col].values}")
    
    # Z-score yöntemi
    print("\n" + "-" * 80)
    print("📊 Z-Score Yöntemi ile Outlier Tespiti (|Z| > 3):")
    print("-" * 80)
    
    z_outlier_summary = {}
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outlier_mask = z_scores > 3
        outlier_count = outlier_mask.sum()
        outlier_percent = (outlier_count / len(df)) * 100
        
        z_outlier_summary[col] = {
            'count': outlier_count,
            'percent': outlier_percent
        }
        
        print(f"{col:15s}: {outlier_count:3d} outlier ({outlier_percent:5.2f}%)")
    
    # Görselleştirme - Box plots
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
    axes = axes.ravel() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        # Box plot
        box_parts = axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                       labels=[col],
                                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                                       medianprops=dict(color='red', linewidth=2),
                                       whiskerprops=dict(color='blue', linewidth=1.5),
                                       capprops=dict(color='blue', linewidth=1.5),
                                       flierprops=dict(marker='o', markerfacecolor='red', 
                                                      markersize=8, alpha=0.5))
        
        axes[idx].set_title(f'{col} - Box Plot\n(Outliers: {outlier_summary[col]["count"]})', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Değer')
        axes[idx].grid(True, alpha=0.3)
        
        # Outlier bilgilerini ekle
        textstr = f'Q1: {outlier_summary[col]["Q1"]:.2f}\n'
        textstr += f'Q3: {outlier_summary[col]["Q3"]:.2f}\n'
        textstr += f'IQR: {outlier_summary[col]["IQR"]:.2f}'
        axes[idx].text(0.98, 0.98, textstr, transform=axes[idx].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      fontsize=9)
    
    # Kullanılmayan subplot'ları gizle
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('img/03_outlier_analysis_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Outlier özet tablosu
    outlier_summary_df = pd.DataFrame(outlier_summary).T
    print("\n📋 Outlier Özet Tablosu:")
    print(outlier_summary_df[['count', 'percent', 'lower_bound', 'upper_bound']])
    
    return outlier_summary, z_outlier_summary

# ============================================================================
# 5. KORELASYON ANALİZİ
# ============================================================================

def correlation_analysis(df):
    """
    Değişkenler arası korelasyon analizi yapar.
    """
    print("\n" + "="*80)
    print("5. KORELASYON ANALİZİ")
    print("="*80)
    
    # Sayısal değişkenler için korelasyon
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    correlation_matrix = df[numeric_cols].corr()
    
    print("\n📊 Korelasyon Matrisi:")
    print(correlation_matrix)
    
    # Güçlü korelasyonları bul (|r| > 0.5)
    print("\n🔍 Güçlü Korelasyonlar (|r| > 0.5):")
    strong_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.5:
                strong_corr.append({
                    'Değişken 1': correlation_matrix.columns[i],
                    'Değişken 2': correlation_matrix.columns[j],
                    'Korelasyon': correlation_matrix.iloc[i, j]
                })
    
    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr).sort_values('Korelasyon', 
                                                                key=abs, 
                                                                ascending=False)
        print(strong_corr_df)
    else:
        print("   • Güçlü korelasyon bulunamadı.")
    
    # Hedef değişken ile korelasyonlar
    if 'num' in numeric_cols:
        print("\n🎯 Hedef Değişken (num) ile En Yüksek Korelasyonlar:")
        target_corr = correlation_matrix['num'].sort_values(ascending=False)
        print(target_corr)
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Tam korelasyon ısı haritası
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Korelasyon Matrisi (Üst Üçgen)', fontsize=14, fontweight='bold')
    
    # Hedef değişken korelasyonları (bar plot)
    if 'num' in numeric_cols:
        target_corr_plot = correlation_matrix['num'].drop('num').sort_values()
        colors = ['red' if x < 0 else 'green' for x in target_corr_plot]
        target_corr_plot.plot(kind='barh', ax=axes[1], color=colors, alpha=0.7)
        axes[1].set_title('Hedef Değişken (num) ile Korelasyonlar', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Korelasyon Katsayısı')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('img/04_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix, strong_corr_df if strong_corr else None

# ============================================================================
# 6. KATEGORİK DEĞİŞKEN ANALİZLERİ
# ============================================================================

def categorical_analysis(df):
    """
    Kategorik değişkenlerin detaylı analizi.
    """
    print("\n" + "="*80)
    print("6. KATEGORİK DEĞİŞKEN ANALİZLERİ")
    print("="*80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    if 'dataset' in categorical_cols:
        categorical_cols.remove('dataset')  # Tüm veriler Cleveland'dan
    
    # Her kategorik değişken için analiz
    for col in categorical_cols:
        print(f"\n{'='*80}")
        print(f"📊 {col.upper()} Analizi")
        print(f"{'='*80}")
        
        # Frekans tablosu
        freq_table = df[col].value_counts()
        freq_percent = df[col].value_counts(normalize=True) * 100
        
        freq_df = pd.DataFrame({
            'Frekans': freq_table,
            'Yüzde (%)': freq_percent
        })
        print(freq_df)
        
        # Hedef değişken ile çapraz tablo
        if 'num' in df.columns:
            print(f"\n{col} vs Kalp Hastalığı (num) - Çapraz Tablo:")
            crosstab = pd.crosstab(df[col], df['num'], margins=True)
            print(crosstab)
            
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df[col], df['num']))
            print(f"\n📈 Chi-Square Test Sonuçları:")
            print(f"   • Chi-square: {chi2:.4f}")
            print(f"   • p-değeri: {p_value:.6f}")
            print(f"   • Serbestlik derecesi: {dof}")
            
            if p_value < 0.05:
                print(f"   ✓ {col} ve kalp hastalığı arasında istatistiksel olarak anlamlı ilişki var (p < 0.05)")
            else:
                print(f"   ✗ {col} ve kalp hastalığı arasında istatistiksel olarak anlamlı ilişki yok (p ≥ 0.05)")
    
    # Görselleştirme
    n_cols_cat = len(categorical_cols)
    n_rows = (n_cols_cat + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, n_rows * 6))
    axes = axes.ravel() if n_cols_cat > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        if 'num' in df.columns:
            # Stacked bar chart
            crosstab_norm = pd.crosstab(df[col], df['num'], normalize='index') * 100
            crosstab_norm.plot(kind='bar', stacked=True, ax=axes[idx], 
                              colormap='viridis', alpha=0.8)
            axes[idx].set_title(f'{col} - Kalp Hastalığı Dağılımı', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Yüzde (%)')
            axes[idx].legend(title='Hastalık Derecesi', bbox_to_anchor=(1.05, 1))
            axes[idx].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Basit bar chart
            df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue', alpha=0.8)
            axes[idx].set_title(f'{col} Dağılımı', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frekans')
            axes[idx].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Kullanılmayan subplot'ları gizle
    for idx in range(n_cols_cat, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('img/05_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 7. İSTATİSTİKSEL TESTLER
# ============================================================================

def statistical_tests(df):
    """
    Çeşitli istatistiksel testler uygular.
    """
    print("\n" + "="*80)
    print("7. İSTATİSTİKSEL TESTLER")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    test_results = {}
    
    # Normallik testleri
    print("\n📊 NORMALLİK TESTLERİ")
    print("-" * 80)
    print("H₀: Veriler normal dağılıma uyar")
    print("H₁: Veriler normal dağılıma uymaz")
    print("α = 0.05")
    print("-" * 80)
    
    normality_results = []
    for col in numeric_cols:
        # Shapiro-Wilk testi
        shapiro_stat, shapiro_p = shapiro(df[col].dropna())
        
        # Kolmogorov-Smirnov testi
        ks_stat, ks_p = kstest(df[col].dropna(), 'norm')
        
        # Anderson-Darling testi (normaltest)
        anderson_stat, anderson_p = normaltest(df[col].dropna())
        
        normality_results.append({
            'Değişken': col,
            'Shapiro-Wilk p': shapiro_p,
            'K-S p': ks_p,
            'Normaltest p': anderson_p,
            'Normal?': 'Evet' if shapiro_p > 0.05 else 'Hayır'
        })
        
        print(f"\n{col}:")
        print(f"   • Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")
        print(f"   • K-S: D={ks_stat:.4f}, p={ks_p:.6f}")
        print(f"   • Normaltest: stat={anderson_stat:.4f}, p={anderson_p:.6f}")
        print(f"   • Sonuç: {'Normal dağılım' if shapiro_p > 0.05 else 'Normal değil'}")
    
    normality_df = pd.DataFrame(normality_results)
    test_results['normality'] = normality_df
    
    # Cinsiyet ve yaş arasındaki ilişki (t-test)
    if 'sex' in df.columns and 'age' in numeric_cols:
        print("\n📊 CİNSİYET VE YAŞ İLİŞKİSİ (T-TEST)")
        print("-" * 80)
        
        male_ages = df[df['sex'] == 'Male']['age'].dropna()
        female_ages = df[df['sex'] == 'Female']['age'].dropna()
        
        t_stat, t_p = stats.ttest_ind(male_ages, female_ages)
        
        print(f"H₀: Erkek ve kadın yaş ortalamaları arasında fark yoktur")
        print(f"H₁: Erkek ve kadın yaş ortalamaları arasında fark vardır")
        print(f"\nErkek yaş ortalaması: {male_ages.mean():.2f} ± {male_ages.std():.2f}")
        print(f"Kadın yaş ortalaması: {female_ages.mean():.2f} ± {female_ages.std():.2f}")
        print(f"\nt-istatistik: {t_stat:.4f}")
        print(f"p-değeri: {t_p:.6f}")
        
        if t_p < 0.05:
            print(f"✓ Anlamlı fark var (p < 0.05)")
        else:
            print(f"✗ Anlamlı fark yok (p ≥ 0.05)")
        
        test_results['t_test_sex_age'] = {'t_stat': t_stat, 'p_value': t_p}
    
    # ANOVA - Göğüs ağrısı türü ve yaş
    if 'cp' in df.columns and 'age' in numeric_cols:
        print("\n📊 GÖĞÜS AĞRISI TÜRÜ VE YAŞ İLİŞKİSİ (ANOVA)")
        print("-" * 80)
        
        groups = [group['age'].dropna() for name, group in df.groupby('cp')]
        f_stat, f_p = stats.f_oneway(*groups)
        
        print(f"H₀: Tüm göğüs ağrısı türlerinin yaş ortalamaları eşittir")
        print(f"H₁: En az bir göğüs ağrısı türünün yaş ortalaması farklıdır")
        print(f"\nF-istatistik: {f_stat:.4f}")
        print(f"p-değeri: {f_p:.6f}")
        
        if f_p < 0.05:
            print(f"✓ Gruplar arası anlamlı fark var (p < 0.05)")
        else:
            print(f"✗ Gruplar arası anlamlı fark yok (p ≥ 0.05)")
        
        test_results['anova_cp_age'] = {'f_stat': f_stat, 'p_value': f_p}
    
    # Mann-Whitney U test (non-parametric)
    if 'sex' in df.columns and 'chol' in numeric_cols:
        print("\n📊 CİNSİYET VE KOLESTEROL İLİŞKİSİ (MANN-WHITNEY U)")
        print("-" * 80)
        
        male_chol = df[df['sex'] == 'Male']['chol'].dropna()
        female_chol = df[df['sex'] == 'Female']['chol'].dropna()
        
        u_stat, u_p = stats.mannwhitneyu(male_chol, female_chol)
        
        print(f"H₀: Erkek ve kadın kolesterol dağılımları aynıdır")
        print(f"H₁: Erkek ve kadın kolesterol dağılımları farklıdır")
        print(f"\nErkek kolesterol: {male_chol.median():.2f} (medyan)")
        print(f"Kadın kolesterol: {female_chol.median():.2f} (medyan)")
        print(f"\nU-istatistik: {u_stat:.4f}")
        print(f"p-değeri: {u_p:.6f}")
        
        if u_p < 0.05:
            print(f"✓ Anlamlı fark var (p < 0.05)")
        else:
            print(f"✗ Anlamlı fark yok (p ≥ 0.05)")
        
        test_results['mannwhitney_sex_chol'] = {'u_stat': u_stat, 'p_value': u_p}
    
    return test_results

# ============================================================================
# 8. GELİŞMİŞ GÖRSELLEŞTİRMELER
# ============================================================================

def advanced_visualizations(df):
    """
    Kapsamlı görselleştirmeler oluşturur.
    """
    print("\n" + "="*80)
    print("8. GELİŞMİŞ GÖRSELLEŞTİRMELER")
    print("="*80)
    
    # 1. Pair plot (sayısal değişkenler)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    # İlk 5 en önemli sayısal değişken için pair plot
    important_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    important_cols = [col for col in important_cols if col in numeric_cols]
    
    if 'num' in df.columns:
        print("\n📊 Pair Plot oluşturuluyor...")
        pairplot_df = df[important_cols + ['num']].copy()
        pairplot_df['num_binary'] = (pairplot_df['num'] > 0).astype(int)
        
        g = sns.pairplot(pairplot_df.drop('num', axis=1), hue='num_binary', 
                        palette={0: 'blue', 1: 'red'}, 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle('Pair Plot - Önemli Değişkenler (Mavi: Sağlıklı, Kırmızı: Hasta)', 
                      y=1.02, fontsize=16, fontweight='bold')
        plt.savefig('img/06_pairplot_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Violin plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    plot_cols = important_cols[:6]
    for idx, col in enumerate(plot_cols):
        if 'num' in df.columns:
            df_plot = df.copy()
            df_plot['num_binary'] = (df_plot['num'] > 0).astype(str)
            df_plot['num_binary'] = df_plot['num_binary'].map({'0': 'Sağlıklı', '1': 'Hasta'})
            
            sns.violinplot(data=df_plot, x='num_binary', y=col, ax=axes[idx], 
                          palette={'Sağlıklı': 'lightblue', 'Hasta': 'lightcoral'})
            axes[idx].set_title(f'{col} Dağılımı (Kalp Hastalığına Göre)', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Durum')
            axes[idx].set_ylabel(col)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('img/07_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Kategorik değişkenler - count plots
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    n_cat = len(categorical_cols)
    n_rows = (n_cat + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, n_rows * 5))
    axes = axes.ravel() if n_cat > 1 else [axes]
    
    for idx, col in enumerate(categorical_cols):
        if 'num' in df.columns:
            df_plot = df.copy()
            df_plot['num_binary'] = (df_plot['num'] > 0).astype(str)
            df_plot['num_binary'] = df_plot['num_binary'].map({'0': 'Sağlıklı', '1': 'Hasta'})
            
            sns.countplot(data=df_plot, x=col, hue='num_binary', ax=axes[idx],
                         palette={'Sağlıklı': 'steelblue', 'Hasta': 'orangered'})
            axes[idx].set_title(f'{col} - Kalp Hastalığı Durumu', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Sayı')
            axes[idx].legend(title='Durum')
            axes[idx].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Kullanılmayan subplot'ları gizle
    for idx in range(n_cat, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('img/08_categorical_countplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Yaş dağılımı detaylı analiz
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram
    axes[0, 0].hist(df['age'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Yaş Dağılımı', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Yaş')
    axes[0, 0].set_ylabel('Frekans')
    axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f"Ortalama: {df['age'].mean():.1f}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # KDE plot
    df['age'].plot(kind='kde', ax=axes[0, 1], color='darkblue', linewidth=2)
    axes[0, 1].set_title('Yaş Yoğunluk Dağılımı (KDE)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Yaş')
    axes[0, 1].set_ylabel('Yoğunluk')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cinsiyet ve yaşa göre
    if 'sex' in df.columns:
        for sex in df['sex'].unique():
            df[df['sex'] == sex]['age'].plot(kind='kde', ax=axes[1, 0], 
                                              label=sex, linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Yaş Dağılımı (Cinsiyete Göre)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Yaş')
        axes[1, 0].set_ylabel('Yoğunluk')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Yaş grupları
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], 
                             labels=['<40', '40-50', '50-60', '60+'])
    age_group_counts = df['age_group'].value_counts().sort_index()
    age_group_counts.plot(kind='bar', ax=axes[1, 1], color='coral', alpha=0.8)
    axes[1, 1].set_title('Yaş Grupları Dağılımı', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Yaş Grubu')
    axes[1, 1].set_ylabel('Frekans')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('img/09_age_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 9. VERİ ÖNİŞLEME ÖNERİLERİ
# ============================================================================

def preprocessing_recommendations(df, outlier_summary):
    """
    Veri önişleme için öneriler sunar.
    """
    print("\n" + "="*80)
    print("9. VERİ ÖNİŞLEME ÖNERİLERİ VE UYARILAR")
    print("="*80)
    
    print("\n⚠️  ÖNEMLİ UYARILAR VE ÖNERİLER:")
    print("-" * 80)
    
    # 1. Eksik değer kontrolü
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n1. EKSİK DEĞERLER:")
        print(f"   • Toplam {missing_count} eksik değer tespit edildi.")
        print(f"   • ÖNERİ: Eksik değerleri impute etmeden önce:")
        print(f"     - Missing at Random (MAR) olup olmadığını kontrol edin")
        print(f"     - Medyan/mod yerine model-based imputation düşünün")
        print(f"     - Eksik değerlerin pattern'ini inceleyin")
    else:
        print(f"\n1. EKSİK DEĞERLER:")
        print(f"   ✓ Veri setinde eksik değer yok - harika!")
    
    # 2. Outlier analizi
    print(f"\n2. OUTLIER (AYKIRI DEĞER) YÖNETİMİ:")
    print(f"   ⚠️  UYARI: Outlier'ları otomatik olarak silmeyin!")
    print(f"   • Medikal verilerde outlier'lar önemli bilgiler içerebilir")
    print(f"   • ÖNERİLER:")
    print(f"     - Outlier'ların klinik olarak anlamlı olup olmadığını kontrol edin")
    print(f"     - Veri giriş hataları için manuel kontrol yapın")
    print(f"     - Robust scaler (IQR-based) kullanmayı düşünün")
    print(f"     - Winsorization (kırpma) ile extreme değerleri sınırlayın")
    
    high_outlier_cols = [col for col, info in outlier_summary.items() 
                         if info['percent'] > 5]
    if high_outlier_cols:
        print(f"\n   Yüksek outlier oranına sahip değişkenler (>%5):")
        for col in high_outlier_cols:
            print(f"     - {col}: %{outlier_summary[col]['percent']:.1f}")
    
    # 3. Normalizasyon uyarıları
    print(f"\n3. NORMALİZASYON/ÖLÇEKLENDİRME:")
    print(f"   ⚠️  UYARI: Tüm değişkenlere aynı ölçeklendirmeyi uygulamayın!")
    print(f"   • ÖNERİLER:")
    print(f"     - Tree-based modeller (Random Forest, XGBoost) için ölçeklendirme GEREKMİYOR")
    print(f"     - Logistic Regression, SVM, Neural Networks için StandardScaler kullanın")
    print(f"     - MinMaxScaler outlier'lara duyarlıdır, dikkatli kullanın")
    print(f"     - ID sütununu ölçeklendirmeyin ve model eğitiminde kullanmayın")
    
    # 4. Kategorik kodlama
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    print(f"\n4. KATEGORİK DEĞİŞKEN KODLAMA:")
    print(f"   • {len(categorical_cols)} kategorik değişken var")
    print(f"   • ÖNERİLER:")
    print(f"     - Binary değişkenler (sex, fbs, exang): Label Encoding (0/1)")
    print(f"     - Ordinal değişkenler (cp, slope): Ordinal Encoding")
    print(f"     - Nominal değişkenler: One-Hot Encoding")
    print(f"     - Yüksek kardinalite için Target Encoding düşünün")
    print(f"     - Tree-based modeller için Label Encoding yeterli")
    
    # 5. Hedef değişken dengesizliği
    if 'num' in df.columns:
        target_dist = df['num'].value_counts()
        target_percent = df['num'].value_counts(normalize=True) * 100
        
        print(f"\n5. HEDEF DEĞİŞKEN DENGESİ:")
        print(f"   • Sınıf dağılımı:")
        for cls, count in target_dist.items():
            print(f"     - Sınıf {cls}: {count} ({target_percent[cls]:.1f}%)")
        
        # Dengesizlik kontrolü
        max_class_ratio = target_percent.max() / target_percent.min()
        if max_class_ratio > 2:
            print(f"\n   ⚠️  UYARI: Sınıf dengesizliği tespit edildi (oran: {max_class_ratio:.1f})")
            print(f"   • ÖNERİLER:")
            print(f"     - SMOTE (Synthetic Minority Over-sampling) kullanın")
            print(f"     - Class weights ayarlayın")
            print(f"     - Stratified sampling kullanın (train/test split)")
            print(f"     - Under-sampling yerine over-sampling tercih edin")
        else:
            print(f"\n   ✓ Sınıf dağılımı makul seviyede dengeli")
    
    # 6. Feature engineering önerileri
    print(f"\n6. ÖZELLİK MÜHENDİSLİĞİ ÖNERİLERİ:")
    print(f"   • Yaş grupları oluşturabilirsiniz (30-40, 40-50, etc.)")
    print(f"   • BMI hesaplayabilirsiniz (eğer kilo/boy varsa)")
    print(f"   • Risk skorları oluşturabilirsiniz (çoklu risk faktörü kombinasyonu)")
    print(f"   • Etkileşim terimleri ekleyin (age*sex, chol*age, etc.)")
    print(f"   • Polinomial özellikler (age², chol², etc.) deneyebilirsiniz")
    
    # 7. Veri bölme stratejisi
    print(f"\n7. VERİ BÖLME STRATEJİSİ:")
    print(f"   • ÖNERİLER:")
    print(f"     - Train/Test: 80/20 veya 70/30 oranı kullanın")
    print(f"     - MUTLAKA Stratified Split kullanın (hedef değişken dengesini koru)")
    print(f"     - Cross-validation için K-Fold (k=5 veya k=10)")
    print(f"     - Küçük veri seti için Leave-One-Out CV düşünün")
    print(f"     - Random state sabitleyerek reproducibility sağlayın")
    
    # 8. Çoklu doğrusallık kontrolü
    print(f"\n8. ÇOKLU DOĞRUSALLIK (MULTICOLLINEARITY):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    correlation_matrix = df[numeric_cols].corr()
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Var1': correlation_matrix.columns[i],
                    'Var2': correlation_matrix.columns[j],
                    'Corr': correlation_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"   ⚠️  UYARI: Yüksek korelasyonlu değişken çiftleri bulundu (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(f"     - {pair['Var1']} vs {pair['Var2']}: r={pair['Corr']:.3f}")
        print(f"\n   • ÖNERİLER:")
        print(f"     - VIF (Variance Inflation Factor) hesaplayın")
        print(f"     - Yüksek korelasyonlu değişkenlerden birini çıkarın")
        print(f"     - PCA ile boyut indirgeme yapın")
        print(f"     - Ridge/Lasso regression kullanın (regularization)")
    else:
        print(f"   ✓ Ciddi çoklu doğrusallık problemi yok (|r| < 0.8)")
    
    # 9. Model seçimi önerileri
    print(f"\n9. MODEL SEÇİMİ ÖNERİLERİ:")
    print(f"   • Başlangıç için önerilen modeller:")
    print(f"     1. Logistic Regression (baseline)")
    print(f"     2. Random Forest (feature importance için)")
    print(f"     3. XGBoost (yüksek performans için)")
    print(f"     4. SVM (non-linear relationships için)")
    print(f"   • Model karşılaştırması için:")
    print(f"     - Accuracy, Precision, Recall, F1-Score")
    print(f"     - ROC-AUC (class imbalance varsa)")
    print(f"     - Confusion Matrix analizi")
    
    # 10. Validasyon stratejisi
    print(f"\n10. VALIDASYON STRATEJİSİ:")
    print(f"    • Overfitting kontrolü için:")
    print(f"      - Learning curves çizin")
    print(f"      - Train vs Test performance karşılaştırın")
    print(f"      - Cross-validation kullanın")
    print(f"      - Regularization uygulayın (L1/L2)")
    print(f"    • Underfitting kontrolü için:")
    print(f"      - Model complexity artırın")
    print(f"      - Feature engineering yapın")
    print(f"      - Ensemble methods deneyin")
    
    print("\n" + "="*80)
    print("✅ VERİ ÖNİŞLEME ÖNERİLERİ TAMAMLANDI")
    print("="*80)

# ============================================================================
# ANA ANALİZ FONKSIYONU
# ============================================================================

def main():
    """
    Tüm analizleri sırasıyla çalıştırır.
    """
    # Veri dosya yolu
    data_path = 'data/heart_disease_uci.csv'
    
    try:
        # 1. Veri yükleme
        df = load_and_explore_data(data_path)
        
        # 2. Veri kalitesi analizi
        missing_df = analyze_data_quality(df)
        
        # 3. Tanımlayıcı istatistikler
        stats_df, additional_stats = descriptive_statistics(df)
        
        # 4. Outlier analizi
        outlier_summary, z_outlier_summary = detect_outliers(df)
        
        # 5. Korelasyon analizi
        correlation_matrix, strong_corr_df = correlation_analysis(df)
        
        # 6. Kategorik değişken analizi
        categorical_analysis(df)
        
        # 7. İstatistiksel testler
        test_results = statistical_tests(df)
        
        # 8. Gelişmiş görselleştirmeler
        advanced_visualizations(df)
        
        # 9. Veri önişleme önerileri
        preprocessing_recommendations(df, outlier_summary)
        
        print("\n" + "="*80)
        print("🎉 ANALİZ TAMAMLANDI!")
        print("="*80)
        print("\n📁 Oluşturulan Görsel Dosyalar:")
        print("   1. 01_data_quality_analysis.png")
        print("   2. 02_descriptive_statistics_distributions.png")
        print("   3. 03_outlier_analysis_boxplots.png")
        print("   4. 04_correlation_analysis.png")
        print("   5. 05_categorical_analysis.png")
        print("   6. 06_pairplot_analysis.png")
        print("   7. 07_violin_plots.png")
        print("   8. 08_categorical_countplots.png")
        print("   9. 09_age_detailed_analysis.png")
        
        print("\n💾 Rapor için 'HEART_DISEASE_ANALYSIS_TECHNICAL_REPORT.md' dosyası oluşturulacak...")
        
        # Sonuçları return et (rapor için kullanılacak)
        return {
            'df': df,
            'missing_df': missing_df,
            'stats_df': stats_df,
            'additional_stats': additional_stats,
            'outlier_summary': outlier_summary,
            'z_outlier_summary': z_outlier_summary,
            'correlation_matrix': correlation_matrix,
            'strong_corr_df': strong_corr_df,
            'test_results': test_results
        }
        
    except FileNotFoundError:
        print(f"❌ HATA: '{data_path}' dosyası bulunamadı!")
        print("Lütfen dosya yolunu kontrol edin.")
        return None
    except Exception as e:
        print(f"❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# SCRIPT ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    results = main()
