import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Sklearn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def apply_smote(X_train, y_train):
    """
    SMOTE (Synthetic Minority Over-sampling Technique) uygulayarak veri dengesizliğini giderir.
    """
    print("\n--- SMOTE ile Veri Dengeleme Başlatılıyor ---")
    print("Önceki Sınıf Dağılımı:")
    print(y_train.value_counts())
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("\nSonraki Sınıf Dağılımı:")
    print(y_train_resampled.value_counts())
    print("--- SMOTE Tamamlandı ---\n")
    
    return X_train_resampled, y_train_resampled

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
    # Dosya yolu
    # Scriptin bulunduğu dizine göre data klasörünü bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "online_shoppers_intention.csv")
    
    print(f"Veri seti yolu: {dataset_path}")
    
    # 1. Veri Yükleme
    try:
        df = load_data(dataset_path)
    except Exception as e:
        print(f"Hata: {e}")
        return

    # 2. Ön İşleme
    df_processed = preprocess_data(df)
    
    # 3. Hipotez Testleri (YENİ)
    perform_hypothesis_testing(df_processed)
    
    # 4. Kümeleme (Analiz amaçlı)
    df_clustered = perform_clustering(df_processed)
    
    # 5. Veri Bölme
    X_train, X_test, y_train, y_test = get_data_splits(df_clustered)
    
    # 6. Özellik Seçimi (YENİ)
    X_train_selected, X_test_selected = perform_feature_selection(X_train, y_train, X_test)
    
    # 7. Dengesiz Veri Yönetimi - SMOTE (YENİ)
    # Not: SMOTE sadece eğitim setine uygulanır!
    X_train_resampled, y_train_resampled = apply_smote(X_train_selected, y_train)
    
    print("\n--- Model Eğitimi Başlıyor (Seçilmiş Özellikler ve SMOTE ile) ---")
    
    # 8. ML Modelleri
    ml_results = train_ml_models(X_train_resampled, y_train_resampled, X_test_selected, y_test)
    
    # 9. DL Modelleri
    dl_results = train_dl_models(X_train_resampled, y_train_resampled, X_test_selected, y_test)
    
    # 10. Sonuçları Birleştirme ve Gösterme
    all_results = {**ml_results, **dl_results}
    
    results_df = pd.DataFrame(all_results).T
    print("\n\n==================================================")
    print("TÜM MODELLERİN PERFORMANS SONUÇLARI (SMOTE + Feature Selection)")
    print("==================================================")
    print(results_df)
    
    # Sonuçları CSV olarak kaydetmek isterseniz:
    # results_df.to_csv(os.path.join(script_dir, "model_comparison_results_improved.csv"))

if __name__ == "__main__":
    main()
