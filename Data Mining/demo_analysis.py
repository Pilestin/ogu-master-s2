#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UCI Heart Disease - Basit Demo Analizi
=======================================

Bu script temel analizleri çalıştırarak sistemin çalıştığını doğrular.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 UCI Heart Disease Demo Analizi Başlatılıyor...")
    
    try:
        # 1. Veri yükleme
        print("\n1. Veri yükleniyor...")
        df = pd.read_csv('heart_disease_uci.csv')
        print(f"✓ Veri başarıyla yüklendi: {df.shape}")
        
        # 2. Temel bilgiler
        print(f"\n2. Veri seti özeti:")
        print(f"   - Boyut: {df.shape}")
        print(f"   - Sütunlar: {list(df.columns)}")
        print(f"   - Eksik değer: {df.isnull().sum().sum()}")
        
        # 3. Hedef değişken oluştur
        print(f"\n3. Hedef değişken hazırlanıyor...")
        df['heart_disease'] = (df['num'] > 0).astype(int)
        print(f"   - Hasta sayısı: {df['heart_disease'].sum()}")
        print(f"   - Sağlıklı sayısı: {(df['heart_disease'] == 0).sum()}")
        
        # 4. Basit görselleştirme
        print(f"\n4. Basit görselleştirmeler...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Yaş dağılımı
        axes[0].hist(df['age'], bins=20, alpha=0.7, color='skyblue')
        axes[0].set_title('Yaş Dağılımı')
        axes[0].set_xlabel('Yaş')
        axes[0].set_ylabel('Frekans')
        
        # Cinsiyet dağılımı
        df['sex'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
        axes[1].set_title('Cinsiyet Dağılımı')
        
        # Kalp hastalığı dağılımı
        df['heart_disease'].value_counts().plot(kind='bar', ax=axes[2], color=['lightblue', 'lightcoral'])
        axes[2].set_title('Kalp Hastalığı Dağılımı')
        axes[2].set_xlabel('0: Sağlıklı, 1: Hasta')
        axes[2].set_ylabel('Sayı')
        axes[2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('demo_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("   ✓ Grafikler oluşturuldu ve kaydedildi")
        
        # 5. Basit makine öğrenmesi
        print(f"\n5. Basit makine öğrenmesi modeli...")
        
        # Kategorik değişkenleri encode et
        df_encoded = df.copy()
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        # Özellik ve hedef ayır
        feature_cols = [col for col in df_encoded.columns if col not in ['id', 'num', 'heart_disease', 'dataset']]
        X = df_encoded[feature_cols].fillna(df_encoded[feature_cols].median())
        y = df_encoded['heart_disease']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Tahmin ve değerlendirme
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   - Eğitim seti: {X_train.shape}")
        print(f"   - Test seti: {X_test.shape}")
        print(f"   - Model doğruluğu: {accuracy:.3f}")
        
        # 6. Özellik önemliliği
        print(f"\n6. Özellik önemliliği analizi...")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("   En önemli 5 özellik:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"   - {row['feature']}: {row['importance']:.3f}")
        
        # 7. Basit istatistikler
        print(f"\n7. Temel istatistikler:")
        
        # Cinsiyete göre hastalık oranları
        gender_stats = df.groupby('sex')['heart_disease'].agg(['mean', 'count'])
        print(f"\n   Cinsiyete göre hastalık oranları:")
        for gender, stats in gender_stats.iterrows():
            print(f"   - {gender}: %{stats['mean']*100:.1f} ({stats['count']} kişi)")
        
        # Yaş gruplarına göre analiz
        df['age_group'] = pd.cut(df['age'], bins=[0, 45, 55, 65, 100], labels=['≤45', '46-55', '56-65', '>65'])
        age_stats = df.groupby('age_group')['heart_disease'].mean()
        print(f"\n   Yaş gruplarına göre hastalık oranları:")
        for age_group, rate in age_stats.items():
            print(f"   - {age_group}: %{rate*100:.1f}")
        
        print(f"\n✅ Demo analiz başarıyla tamamlandı!")
        print(f"📊 Grafik dosyası: demo_visualization.png")
        print(f"🎯 Model performansı: %{accuracy*100:.1f} doğruluk")
        
        return {
            'data_shape': df.shape,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'gender_stats': gender_stats
        }
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n🎉 Analiz başarılı!")
        print(f"📋 Sonuçlar: {type(results)} tipinde obje döndürüldü")
    else:
        print(f"\n💥 Analiz başarısız!")
