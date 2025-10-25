#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gerekli kütüphanelerin test edilmesi
"""

try:
    import pandas as pd
    print("✓ pandas başarıyla yüklendi")
except ImportError as e:
    print(f"✗ pandas yüklenemedi: {e}")

try:
    import numpy as np
    print("✓ numpy başarıyla yüklendi")
except ImportError as e:
    print(f"✗ numpy yüklenemedi: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib başarıyla yüklendi")
except ImportError as e:
    print(f"✗ matplotlib yüklenemedi: {e}")

try:
    import seaborn as sns
    print("✓ seaborn başarıyla yüklendi")
except ImportError as e:
    print(f"✗ seaborn yüklenemedi: {e}")

try:
    from sklearn.model_selection import train_test_split
    print("✓ scikit-learn başarıyla yüklendi")
except ImportError as e:
    print(f"✗ scikit-learn yüklenemedi: {e}")

try:
    from scipy import stats
    print("✓ scipy başarıyla yüklendi")
except ImportError as e:
    print(f"✗ scipy yüklenemedi: {e}")

# Veri dosyası kontrolü
import os
if os.path.exists('heart_disease_uci.csv'):
    print("✓ heart_disease_uci.csv dosyası bulundu")
    
    # Dosyanın okunabilirliğini test et
    try:
        df = pd.read_csv('heart_disease_uci.csv')
        print(f"✓ Veri seti başarıyla okundu: {df.shape} boyut")
        print(f"✓ Sütunlar: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Veri seti okunamadı: {e}")
else:
    print("✗ heart_disease_uci.csv dosyası bulunamadı")

print("\n🚀 Test tamamlandı!")
