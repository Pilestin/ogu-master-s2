"""
UCI Heart Disease - Senaryo Karşılaştırmalı Analiz
===================================================

Bu script farklı senaryoları karşılaştırır:
- Senaryo 1: Temel model (varsayılan parametreler)
- Senaryo 2: Özellik mühendisliği ile
- Senaryo 3: PCA ile boyut indirgeme
- Senaryo 4: Optuna ile hiperparametre optimizasyonu
- Senaryo 5: Tüm iyileştirmeler birlikte

NOT: Sadece Cleveland veri seti kullanılmaktadır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mstats

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Validasyon ve Metrikler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)

# Dengesiz Veri
from imblearn.over_sampling import SMOTE

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Global değişken: Sonuç klasörü
RESULTS_DIR = None

def create_results_folder():
    """Tarih ve saat damgalı sonuç klasörü oluştur"""
    global RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(__file__).parent / f"scenario_results_{timestamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Sonuç klasörü: {RESULTS_DIR}")
    return RESULTS_DIR

# ============================================================
# VERİ YÜKLEME VE TEMEL İŞLEME
# ============================================================

def load_cleveland_data(filepath):
    """Cleveland verisini yükle ve temel temizleme yap"""
    df = pd.read_csv(filepath)
    
    # Sadece Cleveland
    df = df[df['dataset'] == 'Cleveland'].copy()
    
    # Binary hedef
    df['target'] = (df['num'] > 0).astype(int)
    
    print(f"📊 Cleveland Veri Seti: {len(df)} örnek")
    print(f"   Sağlıklı: {sum(df['target']==0)}, Hasta: {sum(df['target']==1)}")
    
    return df

def basic_preprocessing(df):
    """Temel veri önişleme (tüm senaryolar için ortak)"""
    df_processed = df.copy()
    
    # Kategorik değişkenleri encode et
    categorical_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal', 'fbs']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna('missing')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Sayısal sütunları belirle
    exclude_cols = ['id', 'num', 'target', 'dataset']
    numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    # KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    return df_processed

def add_feature_engineering(df):
    """Özellik mühendisliği ekle"""
    df_fe = df.copy()
    
    # Risk skoru
    df_fe['risk_score'] = (df_fe['age'] * df_fe['chol']) / 10000
    
    # Yaş kategorileri
    df_fe['age_group'] = pd.cut(
        df_fe['age'], 
        bins=[0, 40, 55, 70, 100],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(1).astype(int)
    
    # Kalp hızı / yaş oranı
    df_fe['hr_age_ratio'] = df_fe['thalch'] / (df_fe['age'] + 1)
    
    # Kan basıncı × kolesterol
    df_fe['bp_chol_interaction'] = (df_fe['trestbps'] * df_fe['chol']) / 10000
    
    return df_fe

def get_features_target(df, exclude_extra=[]):
    """Özellik ve hedef ayır"""
    exclude_cols = ['id', 'num', 'target', 'dataset'] + exclude_extra
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y, feature_cols

# ============================================================
# MODEL DEĞERLENDİRME
# ============================================================

def evaluate_model(model, X, y, cv=10):
    """Model performansını değerlendir"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
    rec = cross_val_score(model, X, y, cv=skf, scoring='recall')
    auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    
    return {
        'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
        'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
        'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
        'auc': f"{auc.mean():.3f}±{auc.std():.3f}",
        'f1_mean': f1.mean()
    }

# ============================================================
# SENARYO 1: TEMEL MODEL
# ============================================================

def scenario_1_baseline(df):
    """Senaryo 1: Temel model - varsayılan parametreler"""
    print("\n" + "="*60)
    print("📌 SENARYO 1: Temel Model (Varsayılan Parametreler)")
    print("="*60)
    
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    # Ölçekleme
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"   Özellik sayısı: {len(features)}")
    
    # Random Forest (varsayılan)
    model = RandomForestClassifier(random_state=42)
    results = evaluate_model(model, X_resampled, y_resampled)
    
    print(f"   ✓ Random Forest: F1={results['f1']}, AUC={results['auc']}")
    
    return results, features

# ============================================================
# SENARYO 2: ÖZELLİK MÜHENDİSLİĞİ
# ============================================================

def scenario_2_feature_engineering(df):
    """Senaryo 2: Özellik mühendisliği eklenmiş"""
    print("\n" + "="*60)
    print("📌 SENARYO 2: Özellik Mühendisliği ile")
    print("="*60)
    
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"   Özellik sayısı: {len(features)} (+4 yeni özellik)")
    
    model = RandomForestClassifier(random_state=42)
    results = evaluate_model(model, X_resampled, y_resampled)
    
    print(f"   ✓ Random Forest: F1={results['f1']}, AUC={results['auc']}")
    
    return results, features

# ============================================================
# SENARYO 3: PCA İLE BOYUT İNDİRGEME
# ============================================================

def scenario_3_pca(df):
    """Senaryo 3: PCA ile boyut indirgeme"""
    print("\n" + "="*60)
    print("📌 SENARYO 3: PCA ile Boyut İndirgeme")
    print("="*60)
    
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    scaler = StandardScaler()  # PCA için StandardScaler önerilir
    X_scaled = scaler.fit_transform(X)
    
    # PCA - %95 varyansı koruyacak şekilde
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"   Orijinal özellik: {X_scaled.shape[1]}")
    print(f"   PCA sonrası: {X_pca.shape[1]} bileşen (%95 varyans)")
    print(f"   Açıklanan varyans: {pca.explained_variance_ratio_.sum():.2%}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_pca, y)
    
    model = RandomForestClassifier(random_state=42)
    results = evaluate_model(model, X_resampled, y_resampled)
    
    print(f"   ✓ Random Forest: F1={results['f1']}, AUC={results['auc']}")
    
    return results, pca

# ============================================================
# SENARYO 4: OPTUNA HİPERPARAMETRE OPTİMİZASYONU
# ============================================================

def scenario_4_optuna(df, n_trials=50):
    """Senaryo 4: Optuna ile hiperparametre optimizasyonu"""
    print("\n" + "="*60)
    print("📌 SENARYO 4: Optuna Hiperparametre Optimizasyonu")
    print("="*60)
    
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    def objective(trial):
        # Hiperparametreler
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        score = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        return score
    
    print(f"   ⏳ {n_trials} deneme yapılıyor...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"\n   🏆 En iyi parametreler:")
    for k, v in best_params.items():
        print(f"      {k}: {v}")
    
    # En iyi parametrelerle final model
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    results = evaluate_model(best_model, X_resampled, y_resampled)
    
    print(f"   ✓ Optimize RF: F1={results['f1']}, AUC={results['auc']}")
    
    return results, best_params

# ============================================================
# SENARYO 5: TÜM İYİLEŞTİRMELER BİRLİKTE
# ============================================================

def scenario_5_all_combined(df, n_trials=50):
    """Senaryo 5: Tüm iyileştirmeler birlikte"""
    print("\n" + "="*60)
    print("📌 SENARYO 5: Tüm İyileştirmeler Birlikte")
    print("   (Özellik Müh. + PCA + Optuna)")
    print("="*60)
    
    # Özellik mühendisliği
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"   Özellikler: {len(features)} → PCA: {X_pca.shape[1]}")
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_pca, y)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        score = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        return score
    
    print(f"   ⏳ {n_trials} deneme yapılıyor...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"\n   🏆 En iyi parametreler:")
    for k, v in best_params.items():
        print(f"      {k}: {v}")
    
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    results = evaluate_model(best_model, X_resampled, y_resampled)
    
    print(f"   ✓ Full Pipeline: F1={results['f1']}, AUC={results['auc']}")
    
    return results, best_params

# ============================================================
# TÜM MODELLER KARŞILAŞTIRMASI (Optuna ile)
# ============================================================

def compare_all_models_with_optuna(df, n_trials=30):
    """En iyi senaryo üzerinde tüm modelleri Optuna ile optimize et ve karşılaştır"""
    print("\n" + "="*60)
    print("🔬 TÜM MODELLER KARŞILAŞTIRMASI (Optuna Optimizasyonu)")
    print("="*60)
    
    # Veri hazırlama (En iyi senaryo: Temel preprocessing)
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    model_results = {}
    best_params_all = {}
    
    # =========================================
    # 1. Logistic Regression
    # =========================================
    print("\n📌 1/4: Logistic Regression optimizasyonu...")
    
    def lr_objective(trial):
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        
        model = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        score = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        return score
    
    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(lr_objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params_lr = study_lr.best_params
    solver = 'saga' if best_params_lr.get('penalty') == 'l1' else 'lbfgs'
    best_lr = LogisticRegression(**best_params_lr, solver=solver, max_iter=1000, 
                                  random_state=42, class_weight='balanced')
    model_results['Logistic Regression'] = evaluate_model(best_lr, X_resampled, y_resampled)
    best_params_all['Logistic Regression'] = best_params_lr
    print(f"   ✓ LR: F1={model_results['Logistic Regression']['f1']}")
    
    # =========================================
    # 2. Random Forest
    # =========================================
    print("\n📌 2/4: Random Forest optimizasyonu...")
    
    def rf_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        score = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        return score
    
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(rf_objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params_rf = study_rf.best_params
    best_rf = RandomForestClassifier(**best_params_rf, random_state=42, n_jobs=-1, class_weight='balanced')
    model_results['Random Forest'] = evaluate_model(best_rf, X_resampled, y_resampled)
    best_params_all['Random Forest'] = best_params_rf
    print(f"   ✓ RF: F1={model_results['Random Forest']['f1']}")
    
    # =========================================
    # 3. SVM
    # =========================================
    print("\n📌 3/4: SVM optimizasyonu...")
    
    def svm_objective(trial):
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
        
        model = SVC(
            C=C, gamma=gamma, kernel=kernel,
            probability=True, random_state=42, class_weight='balanced'
        )
        score = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        return score
    
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(svm_objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params_svm = study_svm.best_params
    best_svm = SVC(**best_params_svm, probability=True, random_state=42, class_weight='balanced')
    model_results['SVM'] = evaluate_model(best_svm, X_resampled, y_resampled)
    best_params_all['SVM'] = best_params_svm
    print(f"   ✓ SVM: F1={model_results['SVM']['f1']}")
    
    # =========================================
    # 4. XGBoost
    # =========================================
    print("\n📌 4/4: XGBoost optimizasyonu...")
    
    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42, n_jobs=-1, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        score = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='f1').mean()
        return score
    
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params_xgb = study_xgb.best_params
    best_xgb = XGBClassifier(**best_params_xgb, random_state=42, n_jobs=-1, 
                             use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model_results['XGBoost'] = evaluate_model(best_xgb, X_resampled, y_resampled)
    best_params_all['XGBoost'] = best_params_xgb
    print(f"   ✓ XGBoost: F1={model_results['XGBoost']['f1']}")
    
    # =========================================
    # Sonuçları Görselleştir
    # =========================================
    print("\n" + "="*60)
    print("📊 MODEL KARŞILAŞTIRMASI SONUÇLARI")
    print("="*60)
    
    # Tablo
    print("\n" + "-"*80)
    print(f"{'Model':<22} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<10}")
    print("-"*80)
    
    for name, res in model_results.items():
        print(f"{name:<22} {res['accuracy']:<15} "
              f"{res.get('precision', 'N/A'):<15} {res['recall']:<15} "
              f"{res['f1']:<15} {res['auc']:<10}")
    print("-"*80)
    
    # En iyi model
    best_model_name = max(model_results.items(), key=lambda x: x[1]['f1_mean'])
    print(f"\n🏆 En İyi Model: {best_model_name[0]} (F1={best_model_name[1]['f1']})")
    
    # Görselleştirme
    plot_model_comparison(model_results)
    
    # En iyi parametreleri kaydet
    params_path = RESULTS_DIR / 'best_params_all_models.txt'
    with open(params_path, 'w', encoding='utf-8') as f:
        for model_name, params in best_params_all.items():
            f.write(f"\n{model_name}:\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
    print(f"\n✓ best_params_all_models.txt kaydedildi")
    
    return model_results, best_params_all

def plot_model_comparison(model_results):
    """Model karşılaştırma grafiği"""
    models = list(model_results.keys())
    f1_scores = [model_results[m]['f1_mean'] for m in models]
    
    # Renk paleti
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1 Score Bar Chart
    bars = axes[0].barh(models, f1_scores, color=colors, edgecolor='black')
    axes[0].set_xlabel('F1-Score', fontsize=12)
    axes[0].set_title('Model Karşılaştırması (Optuna Optimized)', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    
    for bar, score in zip(bars, f1_scores):
        axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # Tüm metrikler
    metrics = ['accuracy', 'recall', 'auc']
    metric_labels = ['Accuracy', 'Recall', 'AUC']
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [model_results[m]['f1_mean'] if metric == 'f1' else 
                  float(model_results[m][metric].split('±')[0]) for m in models]
        axes[1].bar(x + i*width, values, width, label=label, alpha=0.8)
    
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Metrik Karşılaştırması', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / 'model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")


# ============================================================
# SONUÇ KARŞILAŞTIRMA VE GÖRSELLEŞTİRME
# ============================================================

def compare_and_visualize(all_results):
    """Tüm senaryoları karşılaştır ve görselleştir"""
    print("\n" + "="*60)
    print("📊 SENARYO KARŞILAŞTIRMASI")
    print("="*60)
    
    # DataFrame oluştur
    comparison_data = []
    for scenario, data in all_results.items():
        comparison_data.append({
            'Senaryo': scenario,
            'F1-Score': data['f1'],
            'Accuracy': data['accuracy'],
            'Recall': data['recall'],
            'AUC': data['auc'],
            'F1_mean': data['f1_mean']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1 Score Bar Chart
    scenarios = list(all_results.keys())
    f1_scores = [all_results[s]['f1_mean'] for s in scenarios]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scenarios)))
    
    bars = axes[0].barh(scenarios, f1_scores, color=colors, edgecolor='black')
    axes[0].set_xlabel('F1-Score', fontsize=12)
    axes[0].set_title('Senaryo Karşılaştırması (F1-Score)', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    
    # Değerleri yaz
    for bar, score in zip(bars, f1_scores):
        axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10)
    
    # İyileştirme Yüzdesi
    baseline_f1 = all_results['1. Baseline']['f1_mean']
    improvements = [(all_results[s]['f1_mean'] - baseline_f1) / baseline_f1 * 100 
                   for s in scenarios]
    
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]
    bars2 = axes[1].barh(scenarios, improvements, color=colors_imp, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('İyileştirme (%)', fontsize=12)
    axes[1].set_title('Baseline\'a Göre İyileştirme', fontsize=14, fontweight='bold')
    
    for bar, imp in zip(bars2, improvements):
        x_pos = imp + 0.5 if imp > 0 else imp - 2
        axes[1].text(x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{imp:+.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / 'scenario_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ {save_path.name} kaydedildi")
    
    # CSV kaydet
    csv_path = RESULTS_DIR / 'scenario_results.csv'
    df_comparison.to_csv(csv_path, index=False)
    print(f"✓ {csv_path.name} kaydedildi")
    
    return df_comparison

# ============================================================
# DETAYLI GÖRSELLEŞTİRMELER (En İyi Senaryo İçin)
# ============================================================

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Confusion matrix görselleştirme"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Sağlıklı', 'Hasta'],
               yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    
    # Tip I ve Tip II hataları
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.12, 
             f'FP (Tip I Hata): {fp} | FN (Tip II Hata): {fn} | Doğruluk: {(tp+tn)/(tp+tn+fp+fn):.2%}',
             transform=plt.gca().transAxes,
             ha='center', fontsize=10, color='darkred')
    
    plt.tight_layout()
    save_path = RESULTS_DIR / 'confusion_matrix_best.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")

def plot_roc_curves_comparison(models_data):
    """Tüm senaryolar için ROC eğrileri"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (name, data) in enumerate(models_data.items()):
        if 'fpr' in data and 'tpr' in data:
            plt.plot(data['fpr'], data['tpr'], color=colors[idx % len(colors)], 
                    linewidth=2, label=f"{name} (AUC = {data['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
    plt.title('ROC Eğrileri - Senaryo Karşılaştırması', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'roc_curves_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")

def plot_feature_importance(model, feature_names, title="Özellik Önem Sıralaması"):
    """Feature importance görselleştirme"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_names)))
    
    plt.barh(range(len(feature_names)), importance[indices][::-1], 
            color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'feature_importance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")
    
    print("\n📊 En Önemli 10 Özellik:")
    for i, idx in enumerate(indices[:10], 1):
        print(f"   {i:2d}. {feature_names[idx]:<25} {importance[idx]:.4f}")
    
    return importance, indices

def generate_detailed_visualizations(df, best_params=None):
    """En iyi senaryo için detaylı görselleştirmeler"""
    print("\n" + "="*60)
    print("📈 DETAYLI GÖRSELLEŞTİRMELER")
    print("="*60)
    
    # Veri hazırlama
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Best params varsa kullan, yoksa varsayılan
    if best_params:
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Cross-validation predictions
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=cv)
    y_proba = cross_val_predict(model, X_resampled, y_resampled, cv=cv, method='predict_proba')[:, 1]
    
    # 1. Confusion Matrix
    print("\n1️⃣ Confusion Matrix oluşturuluyor...")
    plot_confusion_matrix(y_resampled, y_pred, "Confusion Matrix (En İyi Model - Optuna RF)")
    
    # 2. ROC Curve (tek model)
    print("\n2️⃣ ROC Eğrisi oluşturuluyor...")
    fpr, tpr, _ = roc_curve(y_resampled, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#4ECDC4', linewidth=2, label=f'Optuna RF (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    plt.fill_between(fpr, tpr, alpha=0.3, color='#4ECDC4')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Eğrisi - En İyi Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'roc_curve_best.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ {save_path.name} kaydedildi")
    
    # 3. Feature Importance
    print("\n3️⃣ Özellik Önemi hesaplanıyor...")
    model.fit(X_resampled, y_resampled)
    plot_feature_importance(model, features, "Özellik Önem Sıralaması (Optuna RF)")
    
    # 4. SHAP Analizi
    print("\n4️⃣ SHAP Analizi yapılıyor...")
    try:
        import shap
        
        X_df = pd.DataFrame(X_resampled, columns=features)
        sample_size = min(200, len(X_df))
        X_sample = X_df.sample(n=sample_size, random_state=42)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # SHAP Summary Plot
        plt.figure(figsize=(12, 8))
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
        
        # SHAP Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_sample, plot_type='bar', show=False)
        plt.title('SHAP Özellik Önem Sıralaması', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = RESULTS_DIR / 'shap_importance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ {save_path.name} kaydedildi")
        
    except ImportError:
        print("   ⚠️ SHAP kütüphanesi yüklü değil, atlanıyor...")
    except Exception as e:
        print(f"   ⚠️ SHAP hatası: {e}")

# ============================================================
# ANA FONKSİYON
# ============================================================

def main():
    """Ana çalıştırma fonksiyonu"""
    print("\n" + "="*60)
    print("  UCI HEART DISEASE - SENARYO KARŞILAŞTIRMASI")
    print("  (Cleveland Veri Seti + Optuna + PCA)")
    print("="*60)
    
    # Sonuç klasörü oluştur
    create_results_folder()
    
    # Veri yükle
    data_path = Path(__file__).parent / 'data' / 'heart_disease_uci.csv'
    df = load_cleveland_data(data_path)
    
    # Tüm senaryoları çalıştır
    all_results = {}
    
    # Senaryo 1: Baseline
    results_1, _ = scenario_1_baseline(df)
    all_results['1. Baseline'] = results_1
    
    # Senaryo 2: Feature Engineering
    results_2, _ = scenario_2_feature_engineering(df)
    all_results['2. Feature Eng.'] = results_2
    
    # Senaryo 3: PCA
    results_3, _ = scenario_3_pca(df)
    all_results['3. PCA'] = results_3
    
    # Senaryo 4: Optuna (50 deneme)
    results_4, best_params_4 = scenario_4_optuna(df, n_trials=50)
    all_results['4. Optuna'] = results_4
    
    # Senaryo 5: All Combined
    results_5, best_params_5 = scenario_5_all_combined(df, n_trials=50)
    all_results['5. All Combined'] = results_5
    
    # Karşılaştırma ve görselleştirme
    df_comparison = compare_and_visualize(all_results)
    
    # Detaylı görselleştirmeler (En iyi senaryo için)
    generate_detailed_visualizations(df, best_params_4)
    
    # Tüm modeller karşılaştırması (Optuna ile)
    model_results, best_params_all = compare_all_models_with_optuna(df, n_trials=30)
    
    # Özet
    print("\n" + "="*60)
    print("  ANALİZ TAMAMLANDI!")
    print(f"  Sonuçlar: {RESULTS_DIR}")
    print("  Oluşturulan dosyalar:")
    print("    - scenario_comparison.png")
    print("    - scenario_results.csv")
    print("    - confusion_matrix_best.png")
    print("    - roc_curve_best.png")
    print("    - feature_importance.png")
    print("    - shap_summary.png")
    print("    - shap_importance.png")
    print("    - model_comparison.png")
    print("    - best_params_all_models.txt")
    print("="*60)
    
    # En iyi senaryo
    best_scenario = max(all_results.items(), key=lambda x: x[1]['f1_mean'])
    print(f"\n🏆 En İyi Senaryo: {best_scenario[0]}")
    print(f"   F1-Score: {best_scenario[1]['f1']}")
    
    return all_results, model_results

if __name__ == "__main__":
    main()
