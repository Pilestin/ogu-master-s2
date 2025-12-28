import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import optuna

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Grafik ayarları (Akademik görünüm için)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# Klasörleri oluştur
os.makedirs('results', exist_ok=True)
os.makedirs('img', exist_ok=True)

def load_and_preprocess_data(filepath):
    print("Veri yükleniyor ve ön işleme yapılıyor...")
    df = pd.read_csv(filepath)
    
    # Sadece Cleveland veri setini kullan
    if 'dataset' in df.columns:
        print(f"Ham veri sayısı: {len(df)}")
        df = df[df['dataset'] == 'Cleveland']
        print(f"Cleveland veri seti filtrelendi. Kalan satır sayısı: {len(df)}")
    
    # Gereksiz sütunları çıkar (id, dataset)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'dataset' in df.columns:
        df = df.drop('dataset', axis=1)
        
    # Hedef değişkeni ikili sınıflandırmaya çevir (0: Yok, 1-4: Var)
    # 'num' sütunu hedef değişkendir
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop('num', axis=1)
    
    # Kategorik ve sayısal sütunları ayır
    # Verisetindeki bazı sütunlar object tipinde olabilir, bunları sayısal hale getirelim
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    
    le = LabelEncoder()
    for col in categorical_cols:
        # Eksik değerleri string olarak doldurup encode edelim, sonra tekrar NaN yapabiliriz veya imputer halleder
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Eksik verileri doldurma (Imputation)
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df_imputed

def evaluate_model(model, X_test, y_test, model_name, scenario_name):
    y_pred = model.predict(X_test)
    # Olasılık değerleri varsa al, yoksa None
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
    
    print(f"--- {model_name} ({scenario_name}) ---")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Confusion Matrix Görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}\n({scenario_name})')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen')
    plt.tight_layout()
    plt.savefig(f'img/cm_{scenario_name}_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    return {
        'Model': model_name, 
        'Scenario': scenario_name, 
        'Accuracy': acc, 
        'Precision': prec, 
        'Recall': rec, 
        'F1': f1, 
        'AUC': auc,
        'y_test': y_test,
        'y_prob': y_prob
    }

def plot_roc_curves(results_list):
    plt.figure(figsize=(10, 8))
    
    for res in results_list:
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
            plt.plot(fpr, tpr, label=f"{res['Model']} ({res['Scenario']}) - AUC: {res['AUC']:.3f}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Rastgele Tahmin')
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title('ROC Eğrileri Karşılaştırması')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/roc_curves_comparison.png', dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, title_suffix=""):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # İlk 20 özelliği göster (eğer çoksa)
        top_n = min(20, len(feature_names))
        indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.title(f'Özellik Önem Düzeyleri {title_suffix}')
        plt.xlabel('Önem Skoru')
        plt.ylabel('Özellikler')
        plt.tight_layout()
        plt.savefig(f'img/feature_importance{title_suffix.replace(" ", "_")}.png', dpi=300)
        plt.close()

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    # Sadece sayısal sütunları al
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
    plt.title('Özellikler Arası Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig('img/correlation_matrix.png', dpi=300)
    plt.close()

def plot_optuna_params(best_params, model_name):
    # Tablo için figür oluştur
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Tablo verisi
    table_data = [[k, v] for k, v in best_params.items()]
    
    # Tabloyu çiz
    table = ax.table(cellText=table_data, colLabels=["Parametre", "En İyi Değer"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Başlık ve sütun renkleri
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')
            
    plt.title(f"{model_name} - Optuna Optimizasyon Sonuçları", pad=20)
    plt.tight_layout()
    plt.savefig(f'img/optuna_params_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_scenario_1(X_train, X_test, y_train, y_test):
    print("\n--- Senaryo 1: Temel Modeller Başlatılıyor ---")
    results = []
    
    # Verileri ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        res = evaluate_model(model, X_test_scaled, y_test, name, "Senaryo 1 (Temel)")
        results.append(res)
        
    return results

def objective_rf(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    return cross_val_score(clf, X, y, n_jobs=-1, cv=3).mean()

def run_scenario_2(X_train, X_test, y_train, y_test):
    print("\n--- Senaryo 2: Gelişmiş Yöntemler (SMOTE + Feature Selection + Optuna) Başlatılıyor ---")
    results = []
    
    # 1. SMOTE ile veri dengeleme
    print("SMOTE uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 2. Feature Selection (En iyi 10 özellik)
    print("Feature Selection uygulanıyor...")
    selector = SelectKBest(f_classif, k=10)
    X_train_sel = selector.fit_transform(X_train_res, y_train_res)
    X_test_sel = selector.transform(X_test)
    
    # Seçilen özellikleri kaydetme (bilgi amaçlı)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    print(f"Seçilen Özellikler: {list(selected_features)}")
    
    # 3. Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # --- Modellerin Eğitilmesi (Tüm Modeller) ---
    
    # 1. Logistic Regression (SMOTE+FS)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train_res)
    res_lr = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression (SMOTE+FS)", "Senaryo 2 (Gelişmiş)")
    results.append(res_lr)

    # 2. SVM (SMOTE+FS)
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train_res)
    res_svm = evaluate_model(svm, X_test_scaled, y_test, "SVM (SMOTE+FS)", "Senaryo 2 (Gelişmiş)")
    results.append(res_svm)

    # 3. Optuna ile Random Forest Optimizasyonu
    print("Optuna ile Random Forest optimize ediliyor...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_rf(trial, X_train_scaled, y_train_res), n_trials=20)
    
    print(f"En iyi parametreler: {study.best_params}")
    
    # Optuna parametrelerini tablo olarak kaydet
    plot_optuna_params(study.best_params, "Random Forest")
    
    best_rf = RandomForestClassifier(**study.best_params, random_state=42)
    best_rf.fit(X_train_scaled, y_train_res)
    
    # Feature Importance Çizimi (Random Forest için)
    plot_feature_importance(best_rf, X_train.columns[selected_indices], " (Random Forest - Optuna)")

    res_rf = evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest (Optuna+SMOTE)", "Senaryo 2 (Gelişmiş)")
    results.append(res_rf)
    
    # XGBoost için de SMOTE ve Feature Selection ile deneyelim (Optuna olmadan hızlı sonuç için)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_scaled, y_train_res)
    
    # Feature Importance Çizimi (XGBoost için)
    plot_feature_importance(xgb, X_train.columns[selected_indices], " (XGBoost)")

    res_xgb = evaluate_model(xgb, X_test_scaled, y_test, "XGBoost (SMOTE+FS)", "Senaryo 2 (Gelişmiş)")
    results.append(res_xgb)
    
    return results

def plot_comparison(results_df):
    # Accuracy Karşılaştırması
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Scenario', palette='viridis')
    plt.title('Model Doğruluk (Accuracy) Karşılaştırması', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    
    # Değerleri çubukların üzerine yaz
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig('img/model_accuracy_comparison.png', dpi=300)
    plt.close()
    
    # F1 Score Karşılaştırması
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=results_df, x='Model', y='F1', hue='Scenario', palette='magma')
    plt.title('Model F1 Skoru Karşılaştırması', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    
    # Değerleri çubukların üzerine yaz
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig('img/model_f1_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Veri Yolu
    data_path = os.path.join('data', 'heart_disease_uci.csv')
    
    if not os.path.exists(data_path):
        print(f"Hata: {data_path} bulunamadı.")
    else:
        # Veriyi hazırla
        df = load_and_preprocess_data(data_path)
        
        # Korelasyon Matrisi Çiz
        plot_correlation_matrix(df)

        X = df.drop('target', axis=1)
        y = df['target']
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Senaryoları Çalıştır
        results1 = run_scenario_1(X_train, X_test, y_train, y_test)
        results2 = run_scenario_2(X_train, X_test, y_train, y_test)
        
        # Tüm sonuçları birleştir
        all_results_list = results1 + results2
        
        # ROC Eğrilerini Çiz
        plot_roc_curves(all_results_list)

        # Sonuçları DataFrame'e çevir (y_test ve y_prob sütunlarını çıkararak kaydet)
        all_results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_prob']} for r in all_results_list])
        
        print("\n--- Tüm Sonuçlar ---")
        print(all_results_df)
        
        all_results_df.to_csv('results/tum_sonuclar.csv', index=False)
        
        # Grafikleri Çiz
        plot_comparison(all_results_df)
        
        print("\nİşlem tamamlandı. Sonuçlar 'results' klasörüne, grafikler 'img' klasörüne kaydedildi.")
