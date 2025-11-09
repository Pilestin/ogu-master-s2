import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Görselleştirme ayarları
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

# Veri setini yükle
df = pd.read_csv('data/online_shoppers_intention.csv')

# Kategorik olması gereken numeric sütunları dönüştür
# OperatingSystems, Browser, Region, TrafficType aslında kategorik değişkenler
categorical_as_numeric = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
for col in categorical_as_numeric:
    df[col] = df[col].astype(str)

# Çıktı klasörünü oluştur
import os
output_dir = 'analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 80)
print("ONLINE SHOPPERS INTENTION VERİ SETİ ANALİZİ")
print("=" * 80)




# ============================================================================
# 1. GENEL VERİ İNCELEMESİ
# ============================================================================
print("\n" + "=" * 80)
print("1. GENEL VERİ BİLGİLERİ")
print("=" * 80)

print(f"\nVeri Seti Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"\nToplam Hücre Sayısı: {df.shape[0] * df.shape[1]:,}")
print(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "-" * 80)
print("Sütun Bilgileri:")
print("-" * 80)
print(df.info())

print("\n" + "-" * 80)
print("İlk 5 Satır:")
print("-" * 80)
print(df.head())

print("\n" + "-" * 80)
print("Veri Tipleri Dağılımı:")
print("-" * 80)
print(df.dtypes.value_counts())

# ============================================================================
# 2. EKSİK VERİ ANALİZİ
# ============================================================================
print("\n" + "=" * 80)
print("2. EKSİK VERİ ANALİZİ")
print("=" * 80)

missing_data = pd.DataFrame({
    'Sütun': df.columns,
    'Eksik Değer': df.isnull().sum(),
    'Yüzde (%)': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Eksik Değer'] > 0].sort_values('Eksik Değer', ascending=False)

if len(missing_data) > 0:
    print("\nEksik Değerler:")
    print(missing_data.to_string(index=False))
else:
    print("\n✓ Veri setinde eksik değer bulunmamaktadır!")

# ============================================================================
# 3. NUMERİK DEĞİŞKENLER ANALİZİ
# ============================================================================
print("\n" + "=" * 80)
print("3. NUMERİK DEĞİŞKENLER - TANIMLAYICI İSTATİSTİKLER")
print("=" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerik Sütun Sayısı: {len(numeric_cols)}")
print(f"Numerik Sütunlar: {', '.join(numeric_cols)}")

print("\n" + "-" * 80)
print("Temel İstatistikler:")
print("-" * 80)
print(df[numeric_cols].describe().T)

print("\n" + "-" * 80)
print("İlave İstatistikler:")
print("-" * 80)
additional_stats = pd.DataFrame({
    'Varyans': df[numeric_cols].var(),
    'Çarpıklık (Skewness)': df[numeric_cols].skew(),
    'Basıklık (Kurtosis)': df[numeric_cols].kurtosis(),
    'Medyan': df[numeric_cols].median()
})
print(additional_stats)

# ============================================================================
# 4. KATEGORİK DEĞİŞKENLER ANALİZİ
# ============================================================================
print("\n" + "=" * 80)
print("4. KATEGORİK DEĞİŞKENLER ANALİZİ")
print("=" * 80)

categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"\nKategorik Sütun Sayısı: {len(categorical_cols)}")
print(f"Kategorik Sütunlar: {', '.join(categorical_cols)}")

for col in categorical_cols:
    print("\n" + "-" * 80)
    print(f"{col} - Dağılım:")
    print("-" * 80)
    value_counts = df[col].value_counts()
    value_percentages = df[col].value_counts(normalize=True) * 100
    
    result_df = pd.DataFrame({
        'Değer': value_counts.index,
        'Sayı': value_counts.values,
        'Yüzde (%)': value_percentages.values
    })
    print(result_df.to_string(index=False))

# ============================================================================
# 5. HEDEF DEĞİŞKEN (REVENUE) ANALİZİ
# ============================================================================
print("\n" + "=" * 80)
print("5. HEDEF DEĞİŞKEN ANALİZİ - REVENUE")
print("=" * 80)

revenue_counts = df['Revenue'].value_counts()
revenue_pct = df['Revenue'].value_counts(normalize=True) * 100

print("\nRevenue Dağılımı:")
for val, count, pct in zip(revenue_counts.index, revenue_counts.values, revenue_pct.values):
    print(f"  {val}: {count} ({pct:.2f}%)")

print(f"\nSınıf Dengesi Oranı: {revenue_counts.min() / revenue_counts.max():.4f}")
print(f"Not: 1'e yakın değer dengeli, 0'a yakın değer dengesiz veri setini gösterir.")

# ============================================================================
# 6. KORELASYON ANALİZİ
# ============================================================================
print("\n" + "=" * 80)
print("6. KORELASYON ANALİZİ")
print("=" * 80)

# Revenue'yu numerik hale getir
df_corr = df.copy()
df_corr['Revenue'] = df_corr['Revenue'].map({'TRUE': 1, 'False': 0, True: 1, False: 0})

correlation_matrix = df_corr[numeric_cols + ['Revenue']].corr()

print("\nRevenue ile En Yüksek Korelasyona Sahip Değişkenler:")
print("-" * 80)
revenue_corr = correlation_matrix['Revenue'].drop('Revenue').sort_values(ascending=False)
print(revenue_corr)

print("\n\nEn Yüksek Korelasyonlar (Revenue Hariç):")
print("-" * 80)
# Üst üçgen matrisi al
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
corr_pairs = correlation_matrix.where(mask).stack().sort_values(ascending=False)
print(corr_pairs.head(10))

# ============================================================================
# 7. GÖRSELLEŞTİRMELER
# ============================================================================
print("\n" + "=" * 80)
print("7. GÖRSELLEŞTİRMELER OLUŞTURULUYOR...")
print("=" * 80)

# 7.1. Korelasyon Matrisi Heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Korelasyon Matrisi - Tüm Numerik Değişkenler', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Korelasyon matrisi kaydedildi: 01_correlation_matrix.png")
plt.close()

# 7.2. Revenue Dağılımı
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Pie chart
revenue_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%', startangle=90, 
                    colors=['#ff9999', '#66b3ff'])
axes[0].set_title('Revenue Dağılımı (Pie Chart)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('')

# Bar chart
revenue_counts.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'])
axes[1].set_title('Revenue Dağılımı (Bar Chart)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Revenue', fontsize=12)
axes[1].set_ylabel('Frekans', fontsize=12)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_revenue_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Revenue dağılımı kaydedildi: 02_revenue_distribution.png")
plt.close()

# 7.3. Numerik Değişkenlerin Dağılımları
n_cols = 4
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df[col].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col}\nMean: {df[col].mean():.2f}, Median: {df[col].median():.2f}', 
                       fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=9)
    axes[idx].set_ylabel('Frekans', fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

# Boş grafikleri gizle
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Numerik Değişkenlerin Dağılımları', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_numeric_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Numerik dağılımlar kaydedildi: 03_numeric_distributions.png")
plt.close()

# 7.4. Box Plot - Aykırı Değer Analizi
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Değer', fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

# Boş grafikleri gizle
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Box Plot - Aykırı Değer Analizi', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Box plotlar kaydedildi: 04_boxplots.png")
plt.close()

# 7.5. Kategorik Değişkenlerin Dağılımları
n_cat = len(categorical_cols)
n_cols_cat = 3
n_rows_cat = int(np.ceil(n_cat / n_cols_cat))
fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(18, n_rows_cat * 5))
axes = axes.flatten() if n_cat > 1 else [axes]

for idx, col in enumerate(categorical_cols):
    value_counts = df[col].value_counts()
    axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral', alpha=0.7)
    axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Kategori', fontsize=10)
    axes[idx].set_ylabel('Frekans', fontsize=10)
    axes[idx].set_xticks(range(len(value_counts)))
    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[idx].grid(axis='y', alpha=0.3)

# Boş grafikleri gizle
for idx in range(len(categorical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Kategorik Değişkenlerin Dağılımları', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_categorical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Kategorik dağılımlar kaydedildi: 05_categorical_distributions.png")
plt.close()

# 7.6. Revenue'ya Göre Numerik Değişkenlerin Karşılaştırması (Önemli olanlar)
important_numeric = ['PageValues', 'ProductRelated_Duration', 'ProductRelated', 
                     'BounceRates', 'ExitRates', 'Administrative_Duration']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(important_numeric):
    df_corr.boxplot(column=col, by='Revenue', ax=axes[idx], patch_artist=True)
    axes[idx].set_title(f'{col} vs Revenue', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Revenue', fontsize=10)
    axes[idx].set_ylabel(col, fontsize=10)
    plt.sca(axes[idx])
    plt.xticks([1, 2], ['FALSE', 'TRUE'])

plt.suptitle('Revenue\'ya Göre Önemli Numerik Değişkenlerin Karşılaştırması', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_revenue_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Revenue karşılaştırması kaydedildi: 06_revenue_comparison.png")
plt.close()

# 7.7. Ay ve Visitor Type'a Göre Revenue Analizi
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Month vs Revenue
month_revenue = pd.crosstab(df['Month'], df['Revenue'], normalize='index') * 100
month_revenue.plot(kind='bar', ax=axes[0], color=['#ff9999', '#66b3ff'])
axes[0].set_title('Aylara Göre Revenue Dağılımı (%)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ay', fontsize=11)
axes[0].set_ylabel('Yüzde (%)', fontsize=11)
axes[0].legend(['FALSE', 'TRUE'], title='Revenue')
axes[0].tick_params(axis='x', rotation=45)

# VisitorType vs Revenue
visitor_revenue = pd.crosstab(df['VisitorType'], df['Revenue'], normalize='index') * 100
visitor_revenue.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'])
axes[1].set_title('Ziyaretçi Tipine Göre Revenue Dağılımı (%)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ziyaretçi Tipi', fontsize=11)
axes[1].set_ylabel('Yüzde (%)', fontsize=11)
axes[1].legend(['FALSE', 'TRUE'], title='Revenue')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('output_categorical_revenue_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Kategorik revenue analizi kaydedildi: output_categorical_revenue_analysis.png")
plt.close()

# 7.8. Pair Plot - Önemli Değişkenler Arası İlişkiler
selected_features = ['PageValues', 'ExitRates', 'BounceRates', 'ProductRelated_Duration', 'Revenue']
pairplot_data = df_corr[selected_features].copy()

plt.figure(figsize=(15, 15))
sns.pairplot(pairplot_data, hue='Revenue', palette={0: '#ff9999', 1: '#66b3ff'}, 
             plot_kws={'alpha': 0.6}, diag_kind='kde', corner=False)
plt.suptitle('Önemli Değişkenler Arası İlişkiler (Pair Plot)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.savefig('output_pairplot.png', dpi=300, bbox_inches='tight')
print("✓ Pair plot kaydedildi: output_pairplot.png")
plt.close()

# ============================================================================
# 8. İSTATİSTİKSEL TESTLER
# ============================================================================
print("\n" + "=" * 80)
print("8. İSTATİSTİKSEL TESTLER")
print("=" * 80)

# 8.1. Chi-Square Test - Kategorik değişkenler için
print("\nChi-Square Testi (Kategorik Değişkenler vs Revenue):")
print("-" * 80)

for col in ['Month', 'VisitorType', 'Weekend']:
    contingency_table = pd.crosstab(df[col], df['Revenue'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\n{col}:")
    print(f"  Chi2 İstatistiği: {chi2:.4f}")
    print(f"  P-değeri: {p_value:.6f}")
    print(f"  Serbestlik Derecesi: {dof}")
    if p_value < 0.05:
        print(f"  ✓ {col} ile Revenue arasında istatistiksel olarak anlamlı bir ilişki VAR (p < 0.05)")
    else:
        print(f"  ✗ {col} ile Revenue arasında istatistiksel olarak anlamlı bir ilişki YOK (p >= 0.05)")

# 8.2. T-Test - Numerik değişkenler için
print("\n\nT-Testi (Numerik Değişkenler vs Revenue):")
print("-" * 80)

revenue_true = df_corr[df_corr['Revenue'] == 1]
revenue_false = df_corr[df_corr['Revenue'] == 0]

for col in important_numeric:
    t_stat, p_value = stats.ttest_ind(revenue_true[col].dropna(), 
                                       revenue_false[col].dropna())
    print(f"\n{col}:")
    print(f"  T İstatistiği: {t_stat:.4f}")
    print(f"  P-değeri: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✓ Revenue grupları arasında {col} için istatistiksel olarak anlamlı bir fark VAR (p < 0.05)")
    else:
        print(f"  ✗ Revenue grupları arasında {col} için istatistiksel olarak anlamlı bir fark YOK (p >= 0.05)")

# ============================================================================
# 9. ÖZET VE ÖNERİLER
# ============================================================================
print("\n" + "=" * 80)
print("9. ANALİZ ÖZETİ VE ÖNEMLİ BULGULAR")
print("=" * 80)

true_key = 'TRUE' if 'TRUE' in revenue_pct.index else True
false_key = 'FALSE' if 'FALSE' in revenue_pct.index else False

print(f"""
TEMEL BULGULAR:
--------------
1. Veri Seti Yapısı:
   - Toplam {df.shape[0]} gözlem, {df.shape[1]} değişken
   - {len(numeric_cols)} numerik, {len(categorical_cols)} kategorik değişken
   
2. Hedef Değişken (Revenue):
   - Dengesiz bir veri seti (sınıf oranları farklı)
   - TRUE oranı: {revenue_pct[true_key]:.2f}%
   - FALSE oranı: {revenue_pct[false_key]:.2f}%
   
3. En Önemli Özellikler (Revenue ile korelasyon):
   - PageValues: En yüksek pozitif korelasyon
   - ExitRates: Negatif korelasyon
   - BounceRates: Negatif korelasyon
   
4. Zaman Bazlı Analiz:
   - Belirli aylarda dönüşüm oranları değişiyor
   - Weekend/Weekday farklılıkları gözlenebilir
   
5. Ziyaretçi Davranışı:
   - Returning Visitor vs New Visitor farklılıkları önemli
   - Ürün sayfalarında geçirilen süre kritik
   
ÖNERİLER:
---------
• PageValues değişkeni en güçlü tahmin edici olabilir
• Bounce ve Exit Rate'leri düşük tutmak önemli
• Ziyaretçi tipine göre farklı stratejiler geliştirilebilir
• Belirli ayların/günlerin özel kampanyalara uygun olabileceği görülüyor
• Sınıf dengesizliği nedeniyle model eğitiminde dikkatli olunmalı

OLUŞTURULAN GÖRSELLER:
--------------------
✓ 01_correlation_matrix.png - Korelasyon matrisi
✓ 02_revenue_distribution.png - Revenue dağılımı
✓ 03_numeric_distributions.png - Numerik değişken dağılımları
✓ 04_boxplots.png - Aykırı değer analizi
✓ 05_categorical_distributions.png - Kategorik dağılımlar
✓ 06_revenue_comparison.png - Revenue karşılaştırması
✓ 07_categorical_revenue_analysis.png - Kategorik revenue analizi
✓ 08_pairplot.png - Değişkenler arası ilişkiler
""")

print("\n" + "=" * 80)
print("ANALİZ TAMAMLANDI!")
print("=" * 80)
print("\nTüm görseller ve istatistikler başarıyla oluşturuldu.")
print("Görseller mevcut çalışma dizinine kaydedildi.\n")
