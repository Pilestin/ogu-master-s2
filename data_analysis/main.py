import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from datetime import datetime

# Sklearn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE

# Boosting imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, Dropout
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Veri setini yükler."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
    return df

def preprocess_data(df):
    """Veri ön işleme adımlarını gerçekleştirir."""
    df_processed = df.copy()

    # 1. 'Revenue' ve 'Weekend' boolean sütunlarını sayısal değerlere dönüştürün
    df_processed['Revenue'] = df_processed['Revenue'].astype(int)
    df_processed['Weekend'] = df_processed['Weekend'].astype(int)

    # 2. 'Month' ve 'VisitorType' gibi kategorik sütunları one-hot encoding kullanarak sayısal temsilcilere dönüştürün
    df_processed = pd.get_dummies(df_processed, columns=['Month', 'VisitorType'], drop_first=True)

    print("Veri ön işleme tamamlandı.")
    return df_processed

def perform_hypothesis_testing(df):
    """
    Veri seti üzerinde hipotez testleri uygular.
    Numerik değişkenler için T-Testi, Kategorik değişkenler için Chi-Square testi.
    """
    print("\n--- Hipotez Testleri Başlatılıyor ---")
    
    target = 'Revenue'
    if target not in df.columns:
        print("Hedef değişken bulunamadı.")
        return

    # Grupları ayır
    group_true = df[df[target] == 1]
    group_false = df[df[target] == 0]

    print(f"Grup True (Revenue=1) Sayısı: {len(group_true)}")
    print(f"Grup False (Revenue=0) Sayısı: {len(group_false)}")

    # Sütun tiplerini belirle
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Revenue ve one-hot encoded olmayan kategorik sütunları çıkar (eğer varsa)
    numeric_cols = [c for c in numeric_cols if c != target and 'Month' not in c and 'VisitorType' not in c] 
    
    # One-hot encoded sütunları kategorik olarak kabul edebiliriz ama orijinal halleri yoksa
    # binary oldukları için Chi-Square uygundur.
    # Ancak burada df_processed geldiği için one-hot sütunlar var.
    # Biz sadece temel numeriklere bakalım: Administrative, Administrative_Duration, etc.
    
    basic_numeric_cols = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]
    
    print("\n1. Numerik Değişkenler için T-Testi (Bağımsız Örneklem):")
    print(f"{'Değişken':<30} | {'T-Statistic':<12} | {'P-Value':<12} | {'Sonuç'}")
    print("-" * 70)
    
    for col in basic_numeric_cols:
        if col in df.columns:
            stat, p_value = stats.ttest_ind(group_true[col], group_false[col], equal_var=False)
            significance = "Anlamlı Fark Var (H0 Red)" if p_value < 0.05 else "Fark Yok (H0 Kabul)"
            print(f"{col:<30} | {stat:.4f}       | {p_value:.4e}   | {significance}")

    print("\n2. Kategorik/Binary Değişkenler için Chi-Square Testi:")
    # Weekend ve One-Hot encoded sütunlar
    categorical_cols = ['Weekend'] + [c for c in df.columns if 'Month_' in c or 'VisitorType_' in c]
    
    print(f"{'Değişken':<30} | {'Chi2 Stat':<12} | {'P-Value':<12} | {'Sonuç'}")
    print("-" * 70)
    
    for col in categorical_cols:
        if col in df.columns:
            contingency_table = pd.crosstab(df[col], df[target])
            stat, p, dof, expected = stats.chi2_contingency(contingency_table)
            significance = "Bağımlı (H0 Red)" if p < 0.05 else "Bağımsız (H0 Kabul)"
            print(f"{col:<30} | {stat:.4f}       | {p:.4e}   | {significance}")
            
    print("--- Hipotez Testleri Tamamlandı ---\n")

def perform_feature_selection(X_train, y_train, X_test):
    """
    Random Forest kullanarak özellik seçimi yapar.
    """
    print("\n--- Özellik Seçimi (Feature Selection) Başlatılıyor ---")
    print(f"Orijinal Özellik Sayısı: {X_train.shape[1]}")
    
    # Özellik seçimi için temel bir model kullan
    selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train, y_train)
    
    # Özellik önemlerini al
    importances = selector_model.feature_importances_
    feature_names = X_train.columns
    
    # Önem derecelerini göster
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\nEn Önemli 10 Özellik:")
    print(feature_imp_df.head(10))
    
    # SelectFromModel ile seçim yap (mean değerinden yüksek olanları seç)
    selector = SelectFromModel(selector_model, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = feature_names[selector.get_support()]
    print(f"\nSeçilen Özellik Sayısı: {X_train_selected.shape[1]}")
    print(f"Seçilen Özellikler: {list(selected_features)}")
    
    # DataFrame formatını korumak için (sütun isimleri kaybolmasın diye)
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    print("--- Özellik Seçimi Tamamlandı ---\n")
    return X_train_selected_df, X_test_selected_df

def apply_smote(X_train, y_train, verbose=True):
    """
    SMOTE (Synthetic Minority Over-sampling Technique) uygulayarak veri dengesizliğini giderir.
    """
    if verbose:
        print("\n--- SMOTE ile Veri Dengeleme Başlatılıyor ---")
        print("Önceki Sınıf Dağılımı:")
        print(y_train.value_counts())
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    if verbose:
        print("\nSonraki Sınıf Dağılımı:")
        print(y_train_resampled.value_counts())
        print("--- SMOTE Tamamlandı ---\n")
    
    return X_train_resampled, y_train_resampled


# =============================================================================
# OPTUNA HİPERPARAMETRE OPTİMİZASYONU
# =============================================================================

def optimize_logistic_regression_optuna(X_train, y_train, n_trials=30):
    """Logistic Regression için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear',
            'random_state': 42,
            'max_iter': 1000
        }
        
        model = LogisticRegression(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_decision_tree_optuna(X_train, y_train, n_trials=30):
    """Decision Tree için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        
        model = DecisionTreeClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_svm_optuna(X_train, y_train, n_trials=30):
    """SVM için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        }
        
        model = SVC(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_knn_optuna(X_train, y_train, n_trials=30):
    """K-Nearest Neighbors için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
            'p': trial.suggest_int('p', 1, 3)
        }
        
        model = KNeighborsClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_gradient_boosting_optuna(X_train, y_train, n_trials=30):
    """Gradient Boosting için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_catboost_optuna(X_train, y_train, n_trials=30):
    """CatBoost için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_state': 42,
            'verbose': 0
        }
        
        model = CatBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_xgboost_optuna(X_train, y_train, n_trials=30):
    """XGBoost için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_lightgbm_optuna(X_train, y_train, n_trials=30):
    """LightGBM için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_random_forest_optuna(X_train, y_train, n_trials=30):
    """Random Forest için Optuna ile hiperparametre optimizasyonu."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def run_optuna_optimization(X_train, y_train, n_trials=30):
    """Hızlı modeller için Optuna optimizasyonu yapar (XGBoost, LightGBM, Random Forest)."""
    print("\n" + "="*60)
    print("OPTUNA HİPERPARAMETRE OPTİMİZASYONU BAŞLIYOR")
    print("="*60)
    
    best_params = {}
    
    # 1. XGBoost
    print("\n[1/3] XGBoost optimizasyonu...")
    xgb_params, xgb_score = optimize_xgboost_optuna(X_train, y_train, n_trials)
    best_params['xgboost'] = xgb_params
    print(f"   En iyi CV F1 Skoru: {xgb_score:.4f}")
    
    # 2. LightGBM
    print("\n[2/3] LightGBM optimizasyonu...")
    lgbm_params, lgbm_score = optimize_lightgbm_optuna(X_train, y_train, n_trials)
    best_params['lightgbm'] = lgbm_params
    print(f"   En iyi CV F1 Skoru: {lgbm_score:.4f}")
    
    # 3. Random Forest
    print("\n[3/3] Random Forest optimizasyonu...")
    rf_params, rf_score = optimize_random_forest_optuna(X_train, y_train, n_trials)
    best_params['random_forest'] = rf_params
    print(f"   En iyi CV F1 Skoru: {rf_score:.4f}")
    
    print("\n--- Optuna Optimizasyonu Tamamlandı ---")
    return best_params


# =============================================================================
# KARŞILAŞTIRMALI ANALİZ
# =============================================================================

def evaluate_scenario(X_train, y_train, X_test, y_test, model, model_name):
    """Tek bir model için değerlendirme yapar."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    }
    
    return metrics


def evaluate_dl_models(X_train, y_train, X_test, y_test, scenario_name):
    """Deep Learning modelleri (Keras MLP ve TabNet) için değerlendirme yapar."""
    dl_results = []
    
    # Veri tipini numpy array olarak ayarla (her birini ayrı kontrol et)
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values
    else:
        X_train_np = np.array(X_train)
    
    if hasattr(X_test, 'values'):
        X_test_np = X_test.values
    else:
        X_test_np = np.array(X_test)
    
    if hasattr(y_train, 'values'):
        y_train_np = y_train.values
    else:
        y_train_np = np.array(y_train)
    
    if hasattr(y_test, 'values'):
        y_test_np = y_test.values
    else:
        y_test_np = np.array(y_test)
    
    # --- Keras Enhanced Deep Learning Model ---
    print("    Eğitiliyor: Enhanced Deep Learning...")
    model_keras = Sequential([
        InputLayer(input_shape=(X_train_np.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model_keras.fit(
        X_train_np, y_train_np,
        epochs=25,
        batch_size=16,
        validation_data=(X_test_np, y_test_np),
        verbose=0
    )
    
    y_pred_proba_keras = model_keras.predict(X_test_np, verbose=0).flatten()
    y_pred_keras = (y_pred_proba_keras > 0.5).astype(int)
    
    dl_results.append({
        "Model": "Enhanced Deep Learning",
        "Senaryo": scenario_name,
        "Accuracy": accuracy_score(y_test_np, y_pred_keras),
        "Precision": precision_score(y_test_np, y_pred_keras, zero_division=0),
        "Recall": recall_score(y_test_np, y_pred_keras),
        "F1-Score": f1_score(y_test_np, y_pred_keras),
        "ROC AUC": roc_auc_score(y_test_np, y_pred_proba_keras)
    })
    print(f"    Enhanced Deep Learning: F1={dl_results[-1]['F1-Score']:.4f}, ROC-AUC={dl_results[-1]['ROC AUC']:.4f}")
    
    # --- TabNet Model ---
    print("    Eğitiliyor: TabNet...")
    tabnet_model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        seed=42,
        verbose=0
    )
    
    tabnet_model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_test_np, y_test_np)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    y_pred_tabnet = tabnet_model.predict(X_test_np)
    y_pred_proba_tabnet = tabnet_model.predict_proba(X_test_np)[:, 1]
    
    dl_results.append({
        "Model": "TabNet",
        "Senaryo": scenario_name,
        "Accuracy": accuracy_score(y_test_np, y_pred_tabnet),
        "Precision": precision_score(y_test_np, y_pred_tabnet, zero_division=0),
        "Recall": recall_score(y_test_np, y_pred_tabnet),
        "F1-Score": f1_score(y_test_np, y_pred_tabnet),
        "ROC AUC": roc_auc_score(y_test_np, y_pred_proba_tabnet)
    })
    print(f"    TabNet: F1={dl_results[-1]['F1-Score']:.4f}, ROC-AUC={dl_results[-1]['ROC AUC']:.4f}")
    
    return dl_results


def run_comparative_analysis(X_train_orig, y_train_orig, X_test_orig, y_test, 
                              feature_names, output_dir):
    """
    4 farklı senaryo ile karşılaştırmalı analiz yapar:
    1. Baseline (ham veri)
    2. Sadece SMOTE
    3. Sadece Feature Selection
    4. SMOTE + Feature Selection + Optuna
    """
    
    print("\n" + "="*70)
    print("KARŞILAŞTIRMALI ANALİZ BAŞLIYOR")
    print("="*70)
    
    results = []
    scenario_best_models = {}
    
    # Kullanılacak temel modeller (11 model)
    def get_base_models():
        return {
            "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "SVM (RBF kernel)": SVC(kernel="rbf", C=10, gamma=0.1, class_weight="balanced", probability=True, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=0, iterations=100)
        }
    
    # =========================================================================
    # SENARYO 1: BASELINE (Ham Veri)
    # =========================================================================
    print("\n" + "-"*50)
    print("SENARYO 1: BASELINE (Ham Veri, Varsayılan Parametreler)")
    print("-"*50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    
    scenario1_results = []
    for name, model in get_base_models().items():
        metrics = evaluate_scenario(X_train_scaled, y_train_orig, X_test_scaled, y_test, model, name)
        metrics["Senaryo"] = "1. Baseline"
        scenario1_results.append(metrics)
        print(f"  {name}: F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC AUC']:.4f}")
    
    # Deep Learning modelleri
    dl_s1 = evaluate_dl_models(X_train_scaled, y_train_orig, X_test_scaled, y_test, "1. Baseline")
    scenario1_results.extend(dl_s1)
    
    results.extend(scenario1_results)
    best_s1 = max(scenario1_results, key=lambda x: x['F1-Score'])
    scenario_best_models['Baseline'] = best_s1
    
    # =========================================================================
    # SENARYO 2: SADECE SMOTE
    # =========================================================================
    print("\n" + "-"*50)
    print("SENARYO 2: SADECE SMOTE")
    print("-"*50)
    
    X_train_smote, y_train_smote = apply_smote(
        pd.DataFrame(X_train_scaled, columns=feature_names), 
        y_train_orig, 
        verbose=False
    )
    
    scenario2_results = []
    for name, model in get_base_models().items():
        metrics = evaluate_scenario(X_train_smote, y_train_smote, X_test_scaled, y_test, model, name)
        metrics["Senaryo"] = "2. SMOTE Only"
        scenario2_results.append(metrics)
        print(f"  {name}: F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC AUC']:.4f}")
    
    # Deep Learning modelleri
    dl_s2 = evaluate_dl_models(X_train_smote, y_train_smote, X_test_scaled, y_test, "2. SMOTE Only")
    scenario2_results.extend(dl_s2)
    
    results.extend(scenario2_results)
    best_s2 = max(scenario2_results, key=lambda x: x['F1-Score'])
    scenario_best_models['SMOTE Only'] = best_s2
    
    # =========================================================================
    # SENARYO 3: SADECE FEATURE SELECTION
    # =========================================================================
    print("\n" + "-"*50)
    print("SENARYO 3: SADECE FEATURE SELECTION")
    print("-"*50)
    
    # Feature selection uygula
    X_train_fs, X_test_fs = perform_feature_selection(
        pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train_orig.index),
        y_train_orig,
        pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test_orig.index)
    )
    
    scenario3_results = []
    for name, model in get_base_models().items():
        metrics = evaluate_scenario(X_train_fs, y_train_orig, X_test_fs, y_test, model, name)
        metrics["Senaryo"] = "3. Feature Selection Only"
        scenario3_results.append(metrics)
        print(f"  {name}: F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC AUC']:.4f}")
    
    # Deep Learning modelleri
    dl_s3 = evaluate_dl_models(X_train_fs, y_train_orig, X_test_fs, y_test, "3. Feature Selection Only")
    scenario3_results.extend(dl_s3)
    
    results.extend(scenario3_results)
    best_s3 = max(scenario3_results, key=lambda x: x['F1-Score'])
    scenario_best_models['Feature Selection Only'] = best_s3
    
    # =========================================================================
    # SENARYO 4: SMOTE + FEATURE SELECTION + OPTUNA
    # =========================================================================
    print("\n" + "-"*50)
    print("SENARYO 4: SMOTE + FEATURE SELECTION + OPTUNA")
    print("-"*50)
    
    # Önce SMOTE uygula, sonra Feature Selection
    X_train_smote_fs, y_train_smote_fs = apply_smote(X_train_fs, y_train_orig, verbose=False)
    
    scenario4_results = []
    
    # Önce tüm temel modelleri varsayılan parametrelerle dene (adil karşılaştırma için)
    print("\n  [Temel Modeller - Varsayılan Parametreler]")
    for name, model in get_base_models().items():
        metrics = evaluate_scenario(X_train_smote_fs, y_train_smote_fs, X_test_fs, y_test, model, name)
        metrics["Senaryo"] = "4. SMOTE + FS + Optuna"
        scenario4_results.append(metrics)
        print(f"  {name}: F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC AUC']:.4f}")
    
    # Optuna ile hiperparametre optimizasyonu
    best_params = run_optuna_optimization(X_train_smote_fs, y_train_smote_fs, n_trials=25)
    
    # Optuna ile optimize edilmiş modeller (sadece hızlı modeller)
    print("\n  [Optuna Optimize Modeller]")
    optimized_models = {
        "XGBoost (Optuna)": XGBClassifier(
            **best_params['xgboost'], 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        ),
        "LightGBM (Optuna)": LGBMClassifier(
            **best_params['lightgbm'], 
            random_state=42, 
            verbose=-1
        ),
        "Random Forest (Optuna)": RandomForestClassifier(
            **best_params['random_forest'], 
            random_state=42, 
            n_jobs=-1
        )
    }
    
    for name, model in optimized_models.items():
        metrics = evaluate_scenario(X_train_smote_fs, y_train_smote_fs, X_test_fs, y_test, model, name)
        metrics["Senaryo"] = "4. SMOTE + FS + Optuna"
        scenario4_results.append(metrics)
        print(f"  {name}: F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC AUC']:.4f}")
    
    # Not: Senaryo 4'te Deep Learning modelleri çalıştırılmıyor (hız optimizasyonu)
    # Temel ML modelleri + Optuna optimize modeller yeterli karşılaştırma sağlıyor
    
    results.extend(scenario4_results)
    best_s4 = max(scenario4_results, key=lambda x: x['F1-Score'])
    scenario_best_models['Full Pipeline'] = best_s4
    
    return pd.DataFrame(results), scenario_best_models, best_params


def generate_comparison_report(results_df, scenario_best_models, output_dir, best_params=None):
    """Karşılaştırma raporu ve görselleri oluşturur."""
    
    print("\n" + "="*70)
    print("SONUÇ RAPORU")
    print("="*70)
    
    # 1. Her Senaryo için en iyi sonuçlar (TÜM METRİKLER)
    print("\n--- Her Senaryo İçin En İyi Model (Tüm Metrikler) ---")
    scenario_summary = []
    for scenario, metrics in scenario_best_models.items():
        print(f"\n{scenario}:")
        print(f"   Model: {metrics['Model']}")
        print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall:    {metrics['Recall']:.4f}")
        print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"   ROC AUC:   {metrics['ROC AUC']:.4f}")
        scenario_summary.append({
            'Senaryo': scenario,
            'En İyi Model': metrics['Model'],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score'],
            'ROC AUC': metrics['ROC AUC']
        })
    
    summary_df = pd.DataFrame(scenario_summary)
    
    # 2. Tekniklerin Katkısı (birden fazla metrik için)
    print("\n--- TEKNİKLERİN KATKISI (Accuracy ve F1 bazında) ---")
    baseline_f1 = scenario_best_models['Baseline']['F1-Score']
    baseline_acc = scenario_best_models['Baseline']['Accuracy']
    
    contributions = {}
    for scenario, metrics in scenario_best_models.items():
        if scenario != 'Baseline':
            f1_improvement = ((metrics['F1-Score'] - baseline_f1) / baseline_f1) * 100
            acc_improvement = ((metrics['Accuracy'] - baseline_acc) / baseline_acc) * 100
            contributions[scenario] = {'F1': f1_improvement, 'Accuracy': acc_improvement}
            print(f"{scenario}:")
            print(f"   F1-Score:  {f1_improvement:+.2f}%")
            print(f"   Accuracy:  {acc_improvement:+.2f}%")
    
    # 3. Sonuçları kaydet
    results_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "scenario_summary.csv"), index=False)
    print(f"\nSonuçlar kaydedildi: {output_dir}")
    
    # 4. Görselleştirmeler
    create_visualizations(summary_df, contributions, output_dir, results_df)
    
    # 5. Optuna Parametre Tablosu Görseli
    if best_params:
        create_parameter_table(best_params, output_dir)
    
    return summary_df, contributions


def create_parameter_table(best_params, output_dir):
    """Optuna ile bulunan en iyi parametrelerin tablo görselini oluşturur."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Parametre verilerini hazırla
    table_data = []
    for model_name, params in best_params.items():
        for param, value in params.items():
            if isinstance(value, float):
                table_data.append([model_name.upper(), param, f"{value:.4f}"])
            else:
                table_data.append([model_name.upper(), param, str(value)])
    
    # Tablo oluştur
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Parametre', 'Değer'],
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.35, 0.4]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Başlık renklerini ayarla
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Satır renklerini model bazında ayarla
    colors = {'XGBOOST': '#e8f4fd', 'LIGHTGBM': '#e8fdf4', 'RANDOM_FOREST': '#fdf4e8'}
    for i, row in enumerate(table_data):
        model = row[0]
        bg_color = colors.get(model, '#ffffff')
        for j in range(3):
            table[(i+1, j)].set_facecolor(bg_color)
    
    ax.set_title('Optuna ile Bulunan En İyi Hiperparametreler', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optuna_parameters.png"), dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Parametre tablosu kaydedildi: {os.path.join(output_dir, 'optuna_parameters.png')}")


def create_visualizations(summary_df, contributions, output_dir, results_df=None):
    """Karşılaştırma grafiklerini oluşturur."""
    
    # Stil ayarları
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    # 1. F1-Score Karşılaştırması ve Teknik Katkıları
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sol: F1 ve ROC-AUC bar chart
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, summary_df['F1-Score'], width, label='F1-Score', color=colors[0])
    bars2 = axes[0].bar(x + width/2, summary_df['ROC AUC'], width, label='ROC AUC', color=colors[1])
    
    axes[0].set_xlabel('Senaryo', fontsize=11)
    axes[0].set_ylabel('Skor', fontsize=11)
    axes[0].set_title('Senaryo Bazlı Performans Karşılaştırması', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df['Senaryo'], rotation=15, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    # Sağ: Teknik Katkıları (F1 ve Accuracy)
    if contributions:
        scenarios = list(contributions.keys())
        f1_values = [contributions[s]['F1'] for s in scenarios]
        acc_values = [contributions[s]['Accuracy'] for s in scenarios]
        
        y_pos = np.arange(len(scenarios))
        height = 0.35
        
        bars1 = axes[1].barh(y_pos - height/2, f1_values, height, label='F1-Score', 
                            color=[colors[1] if v >= 0 else colors[3] for v in f1_values])
        bars2 = axes[1].barh(y_pos + height/2, acc_values, height, label='Accuracy', 
                            color=[colors[0] if v >= 0 else colors[3] for v in acc_values], alpha=0.7)
        
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(scenarios)
        axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=1)
        axes[1].set_xlabel('% İyileşme (Baseline\'a Göre)', fontsize=11)
        axes[1].set_title('Her Tekniğin Katkısı (F1 ve Accuracy)', fontsize=13, fontweight='bold')
        axes[1].legend()
        
        for bar, val in zip(bars1, f1_values):
            axes[1].text(val + 0.3 if val >= 0 else val - 0.3, bar.get_y() + bar.get_height()/2,
                        f'{val:+.1f}%', ha='left' if val >= 0 else 'right', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_barplot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Tüm metriklerin heatmap'i
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    heatmap_data = summary_df[metrics_cols].values
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=metrics_cols, yticklabels=summary_df['Senaryo'],
                ax=ax, vmin=0.3, vmax=1.0, cbar_kws={'label': 'Skor'})
    
    ax.set_title('Tüm Metrikler - Senaryo Karşılaştırması', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Her Metrik için Senaryo Bazlı Değişim Grafikleri (5 ayrı grafik)
    if results_df is not None:
        create_detailed_metric_plots(results_df, output_dir)
    
    print(f"\nGörseller kaydedildi: {output_dir}")


def create_detailed_metric_plots(results_df, output_dir):
    """Her metrik için senaryo bazlı detaylı grafikler oluşturur."""
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    scenarios = results_df['Senaryo'].unique()
    models = results_df[results_df['Senaryo'] == scenarios[0]]['Model'].unique()
    
    # Renk paleti
    n_models = len(models)
    colors = plt.cm.tab20(np.linspace(0, 1, n_models))
    
    # 3.1 Tüm metriklerin senaryolara göre değişimi (Line Plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for i, model in enumerate(models):
            model_data = results_df[results_df['Model'] == model]
            if len(model_data) > 0:
                scenario_order = ['1. Baseline', '2. SMOTE Only', '3. Feature Selection Only', '4. SMOTE + FS + Optuna']
                model_data_sorted = model_data.set_index('Senaryo').reindex([s for s in scenario_order if s in model_data['Senaryo'].values])
                
                ax.plot(range(len(model_data_sorted)), model_data_sorted[metric].values, 
                       marker='o', label=model, color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Senaryo', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{metric} - Senaryo Değişimi', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(scenario_order)))
        ax.set_xticklabels(['Baseline', 'SMOTE', 'FS', 'Full'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    # Son subplot'u legend için kullan
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=9, ncol=2, title='Modeller')
    
    plt.suptitle('Tüm Metriklerin Senaryolara Göre Değişimi', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_scenario_lines.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3.2 Her senaryo için model karşılaştırma bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    scenario_short = {
        '1. Baseline': 'Baseline',
        '2. SMOTE Only': 'SMOTE Only',
        '3. Feature Selection Only': 'Feature Selection',
        '4. SMOTE + FS + Optuna': 'Full Pipeline'
    }
    
    for idx, scenario in enumerate(scenarios[:4]):
        ax = axes[idx]
        scenario_data = results_df[results_df['Senaryo'] == scenario].copy()
        scenario_data = scenario_data.sort_values('F1-Score', ascending=True)
        
        y_pos = np.arange(len(scenario_data))
        
        bars = ax.barh(y_pos, scenario_data['F1-Score'], color=plt.cm.viridis(scenario_data['F1-Score']))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenario_data['Model'], fontsize=9)
        ax.set_xlabel('F1-Score', fontsize=10)
        ax.set_title(f'{scenario_short.get(scenario, scenario)}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        for bar, val in zip(bars, scenario_data['F1-Score']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
    
    plt.suptitle('Her Senaryo İçin Model F1-Score Karşılaştırması', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "models_by_scenario_bars.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3.3 Tüm sonuçlar için büyük heatmap
    pivot_f1 = results_df.pivot_table(values='F1-Score', index='Model', columns='Senaryo', aggfunc='first')
    
    # Sütun sırasını düzenle
    col_order = ['1. Baseline', '2. SMOTE Only', '3. Feature Selection Only', '4. SMOTE + FS + Optuna']
    pivot_f1 = pivot_f1[[c for c in col_order if c in pivot_f1.columns]]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax, vmin=0.3, vmax=0.8, cbar_kws={'label': 'F1-Score'},
                linewidths=0.5)
    
    ax.set_title('Tüm Modeller - F1-Score Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xlabel('Senaryo', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_models_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()





def perform_clustering(df_processed):
    """K-Means kümeleme işlemini gerçekleştirir."""
    print("Kümeleme işlemi başlatılıyor...")
    
    # 1. Hedef değişken 'Revenue' hariç tüm özelliklerden oluşan bir DataFrame oluşturun
    X_cluster = df_processed.drop('Revenue', axis=1)

    # 2. X_cluster verisini standartlaştırın
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 3. Seçilen optimal küme sayısını (örneğin 4) kullanarak KMeans modelini eğitin
    optimal_k = 4 
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_clustered = df_processed.copy()
    df_clustered['Cluster'] = clusters

    print(f"K-Means clustering completed with {optimal_k} clusters.")
    print("Cluster distribution:")
    print(df_clustered['Cluster'].value_counts())
    
    return df_clustered

def get_data_splits(df_processed):
    """Veriyi eğitim ve test setlerine ayırır."""
    # Not: Notebook'ta kümeleme yapıldıktan sonra 'Cluster' sütunu eğitim verisinden çıkarılıyor.
    # Eğer df_processed içinde 'Cluster' varsa çıkaralım.
    
    drop_cols = ['Revenue']
    if 'Cluster' in df_processed.columns:
        drop_cols.append('Cluster')
        
    X = df_processed.drop(drop_cols, axis=1)
    y = df_processed['Revenue']

    # Stratifiye edilmiş bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Veri eğitim ve test setlerine ayrıldı.")
    print(f"X_train boyutu: {X_train.shape}, y_train boyutu: {y_train.shape}")
    print(f"X_test boyutu: {X_test.shape}, y_test boyutu: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(name, y_test, y_pred, y_pred_proba=None):
    """Model performansını değerlendirir ve sonuçları döndürür."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    roc_auc = 0
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0

    print(f"\n----- {name} Performans Metrikleri -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }

def train_ml_models(X_train, y_train, X_test, y_test):
    """Klasik makine öğrenimi modellerini eğitir ve değerlendirir."""
    results = {}
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, iterations=100),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # SVM Özel durumu (olasılık ve parametreler)
    print("\nEğitiliyor: SVM (RBF kernel)...")
    svm_model = SVC(kernel="rbf", C=10, gamma=0.1, class_weight="balanced", probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
    y_pred_svm = (y_pred_proba_svm >= 0.3).astype(int) # Threshold 0.3
    results["SVM (RBF kernel)"] = evaluate_model("SVM (RBF kernel)", y_test, y_pred_svm, y_pred_proba_svm)

    for name, model in models.items():
        print(f"\nEğitiliyor: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
        results[name] = evaluate_model(name, y_test, y_pred, y_pred_proba)
        
    return results

def train_dl_models(X_train, y_train, X_test, y_test):
    """Derin öğrenme modellerini (Keras MLP ve TabNet) eğitir ve değerlendirir."""
    results = {}
    
    # Veriyi ölçeklendirme (DL modelleri için önemli)
    scaler_dl = StandardScaler()
    X_train_scaled = scaler_dl.fit_transform(X_train)
    X_test_scaled = scaler_dl.transform(X_test)
    
    # --- Keras MLP Model ---
    print("\nEğitiliyor: Enhanced Deep Learning (Keras)...")
    model_enhanced = Sequential([
        InputLayer(input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model_enhanced.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model_enhanced.fit(
        X_train_scaled, y_train,
        epochs=25,
        batch_size=16,
        validation_data=(X_test_scaled, y_test),
        verbose=0 # Çıktıyı temiz tutmak için 0
    )
    
    y_pred_proba_dl = model_enhanced.predict(X_test_scaled, verbose=0)
    y_pred_dl = (y_pred_proba_dl > 0.5).astype(int).flatten()
    y_pred_proba_dl = y_pred_proba_dl.flatten()
    
    results["Enhanced Deep Learning"] = evaluate_model("Enhanced Deep Learning", y_test, y_pred_dl, y_pred_proba_dl)
    
    # --- TabNet Model ---
    print("\nEğitiliyor: TabNet...")
    tabnet_model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        seed=42,
        verbose=0
    )
    
    tabnet_model.fit(
        X_train_scaled,
        y_train.values,
        eval_set=[(X_test_scaled, y_test.values)],
        eval_metric=['auc'],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    y_pred_tabnet = tabnet_model.predict(X_test_scaled)
    y_pred_proba_tabnet = tabnet_model.predict_proba(X_test_scaled)[:, 1]
    
    results["TabNet"] = evaluate_model("TabNet", y_test, y_pred_tabnet, y_pred_proba_tabnet)
    
    return results

def main():
    """
    Ana işlem akışı - Karşılaştırmalı Analiz Pipeline
    
    4 Senaryo ile her tekniğin katkısını ölçer:
    1. Baseline (ham veri, varsayılan parametreler)
    2. Sadece SMOTE
    3. Sadece Feature Selection
    4. SMOTE + Feature Selection + Optuna
    """
    
    print("\n" + "="*70)
    print("ONLINE SHOPPERS INTENTION - KARŞILAŞTIRMALI ANALİZ")
    print("SMOTE | Feature Selection | Optuna Hiperparametre Optimizasyonu")
    print("="*70)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dosya yolları
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "online_shoppers_intention.csv")
    output_dir = os.path.join(script_dir, "results")
    
    # Çıktı dizini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nVeri seti: {dataset_path}")
    print(f"Çıktı dizini: {output_dir}")
    
    # 1. Veri Yükleme
    try:
        df = load_data(dataset_path)
        print(f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
        print(f"Hedef değişken dağılımı:\n{df['Revenue'].value_counts()}")
    except Exception as e:
        print(f"Hata: {e}")
        return

    # 2. Ön İşleme
    df_processed = preprocess_data(df)
    
    # 3. Hipotez Testleri
    perform_hypothesis_testing(df_processed)
    
    # 4. Kümeleme (Analiz amaçlı, model eğitiminde kullanılmayacak)
    df_clustered = perform_clustering(df_processed)
    
    # 5. Veri Bölme
    X_train, X_test, y_train, y_test = get_data_splits(df_clustered)
    feature_names = X_train.columns.tolist()
    
    # ==========================================================================
    # 6. KARŞILAŞTIRMALI ANALİZ - 4 SENARYO
    # ==========================================================================
    results_df, scenario_best_models, best_params = run_comparative_analysis(
        X_train, y_train, X_test, y_test, 
        feature_names, output_dir
    )
    
    # 7. Raporlama ve Görselleştirme
    summary_df, contributions = generate_comparison_report(
        results_df, scenario_best_models, output_dir, best_params
    )
    
    # ==========================================================================
    # 8. SONUÇLARIN ÖZETİ
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALİZ TAMAMLANDI")
    print("="*70)
    
    print("\n📊 SENARYO KARŞILAŞTIRMA TABLOSU:")
    print(summary_df.to_string(index=False))
    
    print("\n📈 TEKNİKLERİN KATKI ANALİZİ:")
    for technique, contrib in contributions.items():
        f1_emoji = "✅" if contrib['F1'] > 0 else "⚠️"
        acc_emoji = "✅" if contrib['Accuracy'] > 0 else "⚠️"
        print(f"  {technique}:")
        print(f"    {f1_emoji} F1-Score: {contrib['F1']:+.2f}%")
        print(f"    {acc_emoji} Accuracy: {contrib['Accuracy']:+.2f}%")
    
    # En iyi sonucu belirle
    best_scenario = max(scenario_best_models.items(), key=lambda x: x[1]['F1-Score'])
    print(f"\n🏆 EN İYİ PERFORMANS:")
    print(f"   Senaryo: {best_scenario[0]}")
    print(f"   Model: {best_scenario[1]['Model']}")
    print(f"   Accuracy:  {best_scenario[1]['Accuracy']:.4f}")
    print(f"   Precision: {best_scenario[1]['Precision']:.4f}")
    print(f"   Recall:    {best_scenario[1]['Recall']:.4f}")
    print(f"   F1-Score:  {best_scenario[1]['F1-Score']:.4f}")
    print(f"   ROC AUC:   {best_scenario[1]['ROC AUC']:.4f}")
    
    # Optuna parametreleri
    print("\n🔧 OPTUNA İLE BULUNAN EN İYİ HİPERPARAMETRELER:")
    for model_name, params in best_params.items():
        print(f"\n   {model_name.upper()}:")
        for param, value in params.items():
            print(f"      - {param}: {value}")
    
    print(f"\n📁 Çıktı dosyaları '{output_dir}' dizinine kaydedildi:")
    print("   - comparison_results.csv (Tüm sonuçlar)")
    print("   - scenario_summary.csv (Senaryo özetleri)")
    print("   - comparison_barplot.png (Karşılaştırma grafiği)")
    print("   - metrics_heatmap.png (Metrik heatmap)")
    print("   - optuna_parameters.png (Parametre tablosu)")
    
    print(f"\nBitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return results_df, summary_df, best_params


if __name__ == "__main__":
    main()

