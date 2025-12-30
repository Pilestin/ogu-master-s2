"""
UCI Heart Disease Veri Seti - İleri Seviye Analiz ve Modelleme
==============================================================

Bu script aim.md dosyasındaki önerileri uygular:
1. KNN Imputer ile eksik veri doldurma
2. Aykırı değer baskılama (Winsorizing)
3. Özellik mühendisliği
4. SMOTE ile sınıf dengeleme
5. Stratified 10-Fold CV ile model karşılaştırma
6. SHAP analizi ve görselleştirmeler

NOT: Sadece Cleveland veri seti kullanılmaktadır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

# Preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.stats import mstats

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Validasyon ve Metrikler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, f1_score, recall_score, 
    precision_score, accuracy_score, roc_auc_score
)

# Dengesiz Veri
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SHAP
import shap

import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')

# Global değişken: Sonuç klasörü
RESULTS_DIR = None

def create_results_folder():
    """Tarih ve saat damgalı sonuç klasörü oluştur"""
    global RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(__file__).parent / f"results_{timestamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Sonuç klasörü oluşturuldu: {RESULTS_DIR}")
    return RESULTS_DIR
sns.set_palette("husl")

# ============================================================
# 1. VERİ YÜKLEME VE İNCELEME
# ============================================================

def load_and_explore_data(filepath):
    """Veriyi yükle ve temel istatistikleri göster"""
    print("=" * 60)
    print("1. VERİ YÜKLEME VE İNCELEME")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"\n📊 Veri Seti Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"\n📋 Sütunlar: {list(df.columns)}")
    
    # Sadece Cleveland verilerini filtrele
    print("\n🔍 Sadece Cleveland verileri filtreleniyor...")
    df = df[df['dataset'] == 'Cleveland'].copy()
    print(f"   ✓ Cleveland veri sayısı: {len(df)} satır")
    
    # Hedef değişken dağılımı
    print("\n🎯 Hedef Değişken (num) Dağılımı:")
    print(df['num'].value_counts())
    
    # Binary sınıflandırma için: 0 = sağlıklı, 1+ = hasta
    df['target'] = (df['num'] > 0).astype(int)
    
    print("\n🎯 Binary Hedef (0=Sağlıklı, 1=Hasta) Dağılımı:")
    print(df['target'].value_counts())
    print(f"Hasta oranı: {df['target'].mean():.2%}")
    
    # Eksik değerler
    print("\n❓ Eksik Değerler:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Eksik': missing, '%': missing_pct})
    print(missing_df[missing_df['Eksik'] > 0])
    
    return df

# ============================================================
# 2. VERİ ÖNİŞLEME
# ============================================================

def preprocess_data(df):
    """Veri önişleme adımları"""
    print("\n" + "=" * 60)
    print("2. VERİ ÖNİŞLEME")
    print("=" * 60)
    
    df_processed = df.copy()
    
    # ----------------------------------------------------------
    # A. Kategorik değişkenleri sayısallaştır
    # ----------------------------------------------------------
    print("\n📝 Kategorik değişkenler encode ediliyor...")
    
    # Kategorik sütunları belirle
    categorical_cols = ['sex', 'dataset', 'cp', 'restecg', 'exang', 'slope', 'thal', 'fbs']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            # NaN değerleri geçici olarak 'missing' ile değiştir
            df_processed[col] = df_processed[col].fillna('missing')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"  ✓ {col}: {len(le.classes_)} kategori")
    
    # ----------------------------------------------------------
    # B. Sayısal sütunları belirle
    # ----------------------------------------------------------
    # id, num ve target sütunlarını hariç tut
    exclude_cols = ['id', 'num', 'target']
    numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    print(f"\n📊 Sayısal sütunlar ({len(numeric_cols)}): {numeric_cols}")
    
    # ----------------------------------------------------------
    # C. KNN Imputer ile eksik değer doldurma
    # ----------------------------------------------------------
    print("\n🔧 KNN Imputer uygulanıyor (n_neighbors=5)...")
    
    # Eksik değer olan sütunları bul
    missing_before = df_processed[numeric_cols].isnull().sum().sum()
    
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    missing_after = df_processed[numeric_cols].isnull().sum().sum()
    print(f"  ✓ Eksik değerler: {missing_before} → {missing_after}")
    
    # ----------------------------------------------------------
    # D. Aykırı değer baskılama (Winsorizing)
    # ----------------------------------------------------------
    print("\n📈 Aykırı değer baskılama (Winsorizing %5-%95)...")
    
    # Sadece sürekli sayısal değişkenlere uygula
    continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    
    for col in continuous_cols:
        if col in df_processed.columns:
            before_min, before_max = df_processed[col].min(), df_processed[col].max()
            df_processed[col] = mstats.winsorize(df_processed[col], limits=[0.05, 0.05])
            after_min, after_max = df_processed[col].min(), df_processed[col].max()
            print(f"  ✓ {col}: [{before_min:.1f}, {before_max:.1f}] → [{after_min:.1f}, {after_max:.1f}]")
    
    # ----------------------------------------------------------
    # E. Özellik Mühendisliği
    # ----------------------------------------------------------
    print("\n🔬 Özellik Mühendisliği...")
    
    # 1. Risk Skoru: Yaş × Kolesterol (normalize)
    df_processed['risk_score'] = (df_processed['age'] * df_processed['chol']) / 10000
    print("  ✓ risk_score = age × chol / 10000")
    
    # 2. Yaş Kategorileri (Binning)
    df_processed['age_group'] = pd.cut(
        df_processed['age'], 
        bins=[0, 40, 55, 70, 100],
        labels=[0, 1, 2, 3]  # 0=Genç, 1=Orta, 2=Risk, 3=Yüksek Risk
    ).astype(int)
    print("  ✓ age_group: 0=Genç(<40), 1=Orta(40-55), 2=Risk(55-70), 3=Yüksek(70+)")
    
    # 3. Kalp Hızı / Yaş Oranı
    df_processed['hr_age_ratio'] = df_processed['thalch'] / df_processed['age']
    print("  ✓ hr_age_ratio = thalch / age")
    
    # 4. Kan Basıncı × Kolesterol etkileşimi
    df_processed['bp_chol_interaction'] = (df_processed['trestbps'] * df_processed['chol']) / 10000
    print("  ✓ bp_chol_interaction = trestbps × chol / 10000")
    
    return df_processed, label_encoders

# ============================================================
# 3. ÖZELLİK VE HEDEF DEĞİŞKEN AYIRMA
# ============================================================

def prepare_features_target(df):
    """Özellik ve hedef değişkenleri ayır"""
    print("\n" + "=" * 60)
    print("3. ÖZELLİK VE HEDEF DEĞİŞKEN AYIRMA")
    print("=" * 60)
    
    # Hariç tutulacak sütunlar
    exclude_cols = ['id', 'num', 'target', 'dataset']
    
    # Özellik sütunları
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"\n📊 Özellik boyutu: {X.shape}")
    print(f"🎯 Hedef dağılımı: Sağlıklı={sum(y==0)}, Hasta={sum(y==1)}")
    print(f"📋 Kullanılan özellikler ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")
    
    return X, y, feature_cols

# ============================================================
# 4. MODEL TANIMLAMA
# ============================================================

def get_models():
    """Model sözlüğü oluştur"""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    return models

# ============================================================
# 5. CROSS-VALIDATION İLE MODEL DEĞERLENDİRME
# ============================================================

def evaluate_models(X, y, feature_names):
    """Stratified 10-Fold CV ile modelleri değerlendir"""
    print("\n" + "=" * 60)
    print("4. MODEL DEĞERLENDİRME (Stratified 10-Fold CV)")
    print("=" * 60)
    
    # RobustScaler kullan
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTE uygula
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    print(f"\n⚖️ SMOTE sonrası: Sağlıklı={sum(y_resampled==0)}, Hasta={sum(y_resampled==1)}")
    
    # Sonuç tablosu
    results = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    models = get_models()
    
    print("\n" + "-" * 80)
    print(f"{'Model':<25} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<10}")
    print("-" * 80)
    
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        # Cross-validation metrikleri
        acc_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
        prec_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='precision')
        rec_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='recall')
        f1_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1')
        auc_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='roc_auc')
        
        # Sonuçları yazdır
        print(f"{name:<25} "
              f"{acc_scores.mean():.3f}±{acc_scores.std():.3f}  "
              f"{prec_scores.mean():.3f}±{prec_scores.std():.3f}  "
              f"{rec_scores.mean():.3f}±{rec_scores.std():.3f}  "
              f"{f1_scores.mean():.3f}±{f1_scores.std():.3f}  "
              f"{auc_scores.mean():.3f}")
        
        results.append({
            'Model': name,
            'Accuracy': f"{acc_scores.mean():.3f}±{acc_scores.std():.3f}",
            'Precision': f"{prec_scores.mean():.3f}±{prec_scores.std():.3f}",
            'Recall': f"{rec_scores.mean():.3f}±{rec_scores.std():.3f}",
            'F1-Score': f"{f1_scores.mean():.3f}±{f1_scores.std():.3f}",
            'AUC': f"{auc_scores.mean():.3f}±{auc_scores.std():.3f}",
            'F1_mean': f1_scores.mean()
        })
        
        # En iyi modeli seç (F1-Score'a göre)
        if f1_scores.mean() > best_f1:
            best_f1 = f1_scores.mean()
            best_model = (name, model)
    
    print("-" * 80)
    print(f"\n🏆 En İyi Model (F1-Score): {best_model[0]} ({best_f1:.3f})")
    
    return results, X_scaled, X_resampled, y_resampled, best_model, scaler

# ============================================================
# 6. GÖRSELLEŞTİRMELER
# ============================================================

def plot_confusion_matrices(X, y, models):
    """Tüm modeller için confusion matrix"""
    print("\n" + "=" * 60)
    print("5. CONFUSION MATRIX GÖRSELLEŞTİRME")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Sağlıklı', 'Hasta'],
                   yticklabels=['Sağlıklı', 'Hasta'])
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Tahmin')
        axes[idx].set_ylabel('Gerçek')
        
        # Tip I ve Tip II hataları
        tn, fp, fn, tp = cm.ravel()
        axes[idx].text(0.5, -0.15, 
                       f'FP (Tip I): {fp} | FN (Tip II): {fn}',
                       transform=axes[idx].transAxes,
                       ha='center', fontsize=9, color='red')
    
    plt.suptitle('Confusion Matrix Karşılaştırması\n(Stratified 10-Fold CV)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = RESULTS_DIR / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")

def plot_roc_curves(X, y, models):
    """ROC eğrileri"""
    print("\n" + "=" * 60)
    print("6. ROC EĞRİLERİ")
    print("=" * 60)
    
    plt.figure(figsize=(10, 8))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, (name, model) in enumerate(models.items()):
        y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
    plt.title('ROC Eğrileri Karşılaştırması\n(Stratified 10-Fold CV)', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = RESULTS_DIR / 'roc_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")

def plot_feature_importance(X, y, feature_names):
    """Feature importance (Random Forest)"""
    print("\n" + "=" * 60)
    print("7. ÖZELLİK ÖNEMİ (Random Forest)")
    print("=" * 60)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_names)))
    
    plt.barh(range(len(feature_names)), importance[indices][::-1], 
            color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.title('Özellik Önem Sıralaması (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = RESULTS_DIR / 'feature_importance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")
    
    print("\n📊 Özellik Önem Sıralaması:")
    for i, idx in enumerate(indices[:10], 1):
        print(f"   {i:2d}. {feature_names[idx]:<25} {importance[idx]:.4f}")

# ============================================================
# 7. SHAP ANALİZİ
# ============================================================

def shap_analysis(X, y, feature_names):
    """SHAP değerleri ile model açıklanabilirliği"""
    print("\n" + "=" * 60)
    print("8. SHAP ANALİZİ (Model Açıklanabilirliği)")
    print("=" * 60)
    
    # X'i DataFrame'e çevir
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Random Forest modeli eğit
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_df, y)
    
    # SHAP değerlerini hesapla
    print("\n⏳ SHAP değerleri hesaplanıyor (bu biraz zaman alabilir)...")
    
    # Daha küçük bir örneklem al (hız için)
    sample_size = min(200, len(X_df))
    X_sample = X_df.sample(n=sample_size, random_state=42)
    
    # TreeExplainer kullan
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    
    # Özet grafiği
    plt.figure(figsize=(12, 8))
    # shap_values[1] hasta sınıfı için
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.title('SHAP Özet Grafiği (Hasta Sınıfı)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = RESULTS_DIR / 'shap_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")
    
    # Bar grafiği
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, plot_type='bar', show=False)
    plt.title('SHAP Özellik Önem Sıralaması', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = RESULTS_DIR / 'shap_importance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")
    
    return shap_values, explainer

# ============================================================
# 8. SONUÇ RAPORU
# ============================================================

def generate_report(results):
    """Sonuç raporu oluştur"""
    print("\n" + "=" * 60)
    print("9. SONUÇ RAPORU")
    print("=" * 60)
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                     UCI HEART DISEASE ANALİZİ                    ║
║                       SONUÇ RAPORU                               ║
╚══════════════════════════════════════════════════════════════════╝

📋 UYGULANAN TEKNİKLER:
   1. KNN Imputer ile eksik değer doldurma (k=5)
   2. Aykırı değer baskılama (Winsorizing %5-%95)
   3. Özellik Mühendisliği:
      - Risk skoru (yaş × kolesterol)
      - Yaş kategorileri (binning)
      - Kalp hızı/yaş oranı
      - Kan basıncı × kolesterol etkileşimi
   4. RobustScaler ile ölçekleme
   5. SMOTE ile sınıf dengeleme
   6. Stratified 10-Fold Cross Validation

📊 MODEL KARŞILAŞTIRMASI:
""")
    
    # Sonuç tablosu
    df_results = pd.DataFrame(results)
    print(df_results[['Model', 'Accuracy', 'Recall', 'F1-Score', 'AUC']].to_string(index=False))
    
    print("""
💡 ÖNEMLİ NOTLAR:
   - Recall (Duyarlılık) tıbbi çalışmalarda kritiktir
   - False Negative (Tip II hata) minimizasyonu önemli
   - SHAP değerleri model kararlarını açıklar
   
📈 ÖNERİLER:
   - Yüksek Recall'a sahip modeli tercih edin
   - Confusion matrix'teki FN sayısını minimize edin
   - SHAP grafiklerini raporda kullanın
""")
    
    # CSV olarak kaydet
    save_path = RESULTS_DIR / 'model_results.csv'
    df_results.to_csv(save_path, index=False)
    print(f"\n✓ {save_path.name} kaydedildi")

# ============================================================
# ANA FONKSİYON
# ============================================================

def main():
    """Ana çalıştırma fonksiyonu"""
    print("\n" + "=" * 60)
    print("  UCI HEART DISEASE - İLERİ SEVİYE ANALİZ PAKETİ")
    print("  (Sadece Cleveland Veri Seti)")
    print("=" * 60)
    
    # Sonuç klasörü oluştur
    create_results_folder()
    
    # Veri yolu
    data_path = Path(__file__).parent / 'data' / 'heart_disease_uci.csv'
    
    # 1. Veri yükleme
    df = load_and_explore_data(data_path)
    
    # 2. Veri önişleme
    df_processed, label_encoders = preprocess_data(df)
    
    # 3. Özellik ve hedef ayırma
    X, y, feature_names = prepare_features_target(df_processed)
    
    # 4. Model değerlendirme
    results, X_scaled, X_resampled, y_resampled, best_model, scaler = evaluate_models(X, y, feature_names)
    
    # 5. Confusion Matrix
    models = get_models()
    plot_confusion_matrices(X_resampled, y_resampled, models)
    
    # 6. ROC Eğrileri
    plot_roc_curves(X_resampled, y_resampled, models)
    
    # 7. Özellik önemi
    plot_feature_importance(X_resampled, y_resampled, feature_names)
    
    # 8. SHAP Analizi
    shap_values, explainer = shap_analysis(X_resampled, y_resampled, feature_names)
    
    # 9. Sonuç raporu
    generate_report(results)
    
    print("\n" + "=" * 60)
    print("  ANALİZ TAMAMLANDI!")
    print(f"  Sonuçlar: {RESULTS_DIR}")
    print("  Oluşturulan dosyalar:")
    print("    - confusion_matrices.png")
    print("    - roc_curves.png")
    print("    - feature_importance.png")
    print("    - shap_summary.png")
    print("    - shap_importance.png")
    print("    - model_results.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
