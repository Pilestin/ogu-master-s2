"""
Veri Önişleme Modülü
UCI Online Shoppers Intention Dataset için veri önişleme fonksiyonları
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class DataPreprocessor:
    """Veri önişleme sınıfı - tüm modeller için ortak kullanılacak"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_cols = None
        self.numerical_cols = None
        
    def load_data(self, filepath='data/online_shoppers_intention.csv'):
        """Veriyi yükle"""
        print(f"Veri yükleniyor: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✓ Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    
    def identify_columns(self, df):
        """Sütun tiplerini belirle"""
        # Revenue hedef değişkenimiz
        target_col = 'Revenue'
        
        # Kategorik sütunlar
        categorical_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 
                          'TrafficType', 'VisitorType', 'Weekend']
        
        # Numerik sütunlar (hedef hariç)
        numerical_cols = [col for col in df.columns 
                         if col not in categorical_cols and col != target_col]
        
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        print(f"\n✓ Numerik sütunlar ({len(numerical_cols)}): {numerical_cols[:5]}...")
        print(f"✓ Kategorik sütunlar ({len(categorical_cols)}): {categorical_cols}")
        
        return numerical_cols, categorical_cols
    
    def handle_missing_values(self, df):
        """Eksik değerleri işle"""
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"\n⚠ {missing_count} eksik değer bulundu, işleniyor...")
            # Numerik sütunlar için medyan ile doldur
            for col in self.numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            # Kategorik sütunlar için mod ile doldur
            for col in self.categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            print("\n✓ Eksik değer yok")
        return df
    
    def encode_categorical(self, df, fit=True):
        """Kategorik değişkenleri encode et"""
        print("\nKategorik değişkenler encode ediliyor...")
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    raise ValueError(f"LabelEncoder for {col} not fitted yet!")
        
        print(f"✓ {len(self.categorical_cols)} kategorik sütun encode edildi")
        return df_encoded
    
    def encode_target(self, y, fit=True):
        """Hedef değişkeni encode et"""
        if fit:
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
        else:
            y_encoded = self.target_encoder.transform(y)
        return y_encoded
    
    def scale_features(self, X, fit=True):
        """Özellikleri ölçeklendir (standardizasyon)"""
        print("\nÖzellikler ölçeklendiriliyor (StandardScaler)...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        print("✓ Ölçeklendirme tamamlandı")
        return X_scaled
    
    def preprocess_pipeline(self, df, test_size=0.2, random_state=42, scale=True):
        """Tam önişleme pipeline'ı"""
        print("\n" + "="*80)
        print("VERİ ÖNİŞLEME BAŞLATILIYOR")
        print("="*80)
        
        # 1. Sütunları belirle
        self.identify_columns(df)
        
        # 2. Eksik değerleri işle
        df = self.handle_missing_values(df)
        
        # 3. X ve y'yi ayır
        X = df.drop('Revenue', axis=1)
        y = df['Revenue']
        
        print(f"\n✓ X boyutu: {X.shape}")
        print(f"✓ y boyutu: {y.shape}")
        print(f"✓ Revenue dağılımı:\n{y.value_counts()}")
        
        # 4. Kategorik değişkenleri encode et
        X_encoded = self.encode_categorical(X, fit=True)
        y_encoded = self.encode_target(y, fit=True)
        
        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        print(f"\n✓ Train set: {X_train.shape[0]} örneklem")
        print(f"✓ Test set: {X_test.shape[0]} örneklem")
        print(f"✓ Train set Revenue dağılımı: {np.bincount(y_train)}")
        print(f"✓ Test set Revenue dağılımı: {np.bincount(y_test)}")
        
        # 6. Ölçeklendirme (isteğe bağlı)
        if scale:
            X_train_scaled = self.scale_features(X_train, fit=True)
            X_test_scaled = self.scale_features(X_test, fit=False)
            self.feature_names = X_encoded.columns.tolist()
            
            print("\n" + "="*80)
            print("ÖNİŞLEME TAMAMLANDI")
            print("="*80)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, X_encoded.columns.tolist()
        else:
            self.feature_names = X_encoded.columns.tolist()
            
            print("\n" + "="*80)
            print("ÖNİŞLEME TAMAMLANDI (Ölçeklendirme yapılmadı)")
            print("="*80)
            
            return X_train.values, X_test.values, y_train, y_test, X_encoded.columns.tolist()
    
    def save_preprocessor(self, filepath='preprocessor.pkl'):
        """Preprocessor'ı kaydet"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'target_encoder': self.target_encoder,
                'feature_names': self.feature_names,
                'categorical_cols': self.categorical_cols,
                'numerical_cols': self.numerical_cols
            }, f)
        print(f"\n✓ Preprocessor kaydedildi: {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.pkl'):
        """Preprocessor'ı yükle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoders = data['label_encoders']
            self.target_encoder = data['target_encoder']
            self.feature_names = data['feature_names']
            self.categorical_cols = data['categorical_cols']
            self.numerical_cols = data['numerical_cols']
        print(f"\n✓ Preprocessor yüklendi: {filepath}")


def get_feature_importance_df(model, feature_names, top_n=None):
    """Model için feature importance DataFrame'i oluştur"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
    
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    if top_n:
        df_importance = df_importance.head(top_n)
    
    return df_importance


if __name__ == "__main__":
    # Test kodu
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(df)
    
    print(f"\n✓ X_train shape: {X_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ y_test shape: {y_test.shape}")
    print(f"✓ Feature count: {len(feature_names)}")
    
    # Preprocessor'ı kaydet
    preprocessor.save_preprocessor()
