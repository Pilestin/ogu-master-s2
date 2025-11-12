"""
Autoencoder Model for Wood Anomaly Detection
Derin öğrenme tabanlı autoencoder ile anomali tespiti
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.metrics import roc_auc_score, f1_score
from utils import (load_image, normalize_image, create_anomaly_map, 
                  calculate_metrics, plot_metrics, visualize_anomaly_detection,
                  save_model_results, preprocess_for_anomaly_detection)

class ConvolutionalAutoencoder:
    """Konvolüsyonel Autoencoder sınıfı"""
    
    def __init__(self, input_shape=(256, 256, 3), latent_dim=128):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        
    def build_model(self):
        """Autoencoder modelini oluştur"""
        # Encoder
        encoder_input = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Bottleneck
        shape_before_flattening = keras.backend.int_shape(x)[1:]
        x = layers.Flatten()(x)
        encoded = layers.Dense(self.latent_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        decoder_input = layers.Dense(np.prod(shape_before_flattening), activation='relu')(encoded)
        x = layers.Reshape(shape_before_flattening)(decoder_input)
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Modeli oluştur
        self.model = keras.Model(encoder_input, decoded)
        self.encoder = keras.Model(encoder_input, encoded)
        
        # Optimizer ve loss function
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.model
    
    def train(self, train_images, validation_split=0.2, epochs=50, batch_size=8):
        """Modeli eğit"""
        print("Autoencoder eğitimi başlıyor...")
        print(f"Eğitim verisi: {train_images.shape}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                'autoencoder_best.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Eğitim
        self.history = self.model.fit(
            train_images, train_images,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Eğitim tamamlandı!")
        return self.history
    
    def predict_anomaly_scores(self, images):
        """Anomali skorlarını hesapla"""
        reconstructed = self.model.predict(images, verbose=0)
        
        # Reconstruction error hesapla
        mse = np.mean(np.square(images - reconstructed), axis=(1, 2, 3))
        
        # Anomali haritalarını hesapla
        anomaly_maps = []
        for i in range(len(images)):
            anomaly_map = create_anomaly_map(images[i], reconstructed[i])
            anomaly_maps.append(anomaly_map)
        
        return mse, np.array(anomaly_maps), reconstructed
    
    def plot_training_history(self):
        """Eğitim geçmişini görselleştir"""
        if self.history is None:
            print("Model henüz eğitilmemiş!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        self.model.save(filepath)
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """Modeli yükle"""
        self.model = keras.models.load_model(filepath)
        print(f"Model yüklendi: {filepath}")

def load_dataset(data_dir, target_size=(256, 256)):
    """Dataset'i yükle ve hazırla"""
    train_good_dir = os.path.join(data_dir, "train", "good")
    test_good_dir = os.path.join(data_dir, "test", "good")
    test_defect_dir = os.path.join(data_dir, "test", "defect")
    
    # Training verisi (sadece good örnekler)
    train_images = []
    train_files = [f for f in os.listdir(train_good_dir) if f.endswith('.bmp')]
    
    print(f"Training verisi yükleniyor: {len(train_files)} dosya...")
    for filename in train_files:
        img_path = os.path.join(train_good_dir, filename)
        img = load_image(img_path, target_size)
        if img is not None:
            img = preprocess_for_anomaly_detection(img, target_size)
            train_images.append(img)
    
    # Test verisi (good + defect)
    test_images = []
    test_labels = []
    
    # Good örnekler
    test_good_files = [f for f in os.listdir(test_good_dir) if f.endswith('.bmp')]
    print(f"Test good verisi yükleniyor: {len(test_good_files)} dosya...")
    for filename in test_good_files:
        img_path = os.path.join(test_good_dir, filename)
        img = load_image(img_path, target_size)
        if img is not None:
            img = preprocess_for_anomaly_detection(img, target_size)
            test_images.append(img)
            test_labels.append(0)  # 0 = normal
    
    # Defect örnekler
    test_defect_files = [f for f in os.listdir(test_defect_dir) if f.endswith('.bmp')]
    print(f"Test defect verisi yükleniyor: {len(test_defect_files)} dosya...")
    for filename in test_defect_files:
        img_path = os.path.join(test_defect_dir, filename)
        img = load_image(img_path, target_size)
        if img is not None:
            img = preprocess_for_anomaly_detection(img, target_size)
            test_images.append(img)
            test_labels.append(1)  # 1 = anomaly
    
    return np.array(train_images), np.array(test_images), np.array(test_labels)

def main():
    """Ana fonksiyon"""
    print("=== AUTOENCODER ANOMALY DETECTION ===\n")
    
    # Dataset yükle
    data_dir = "wood"
    target_size = (256, 256)
    
    train_images, test_images, test_labels = load_dataset(data_dir, target_size)
    
    print(f"\nDataset Bilgileri:")
    print(f"Training set: {train_images.shape}")
    print(f"Test set: {test_images.shape}")
    print(f"Test labels: {test_labels.shape}")
    print(f"Normal samples: {np.sum(test_labels == 0)}")
    print(f"Anomaly samples: {np.sum(test_labels == 1)}")
    
    # Model oluştur
    autoencoder = ConvolutionalAutoencoder(
        input_shape=(target_size[0], target_size[1], 3),
        latent_dim=128
    )
    
    model = autoencoder.build_model()
    print(f"\nModel Özeti:")
    model.summary()
    
    # Modeli eğit
    print("\n" + "="*50)
    print("MODEL EĞİTİMİ BAŞLIYOR...")
    print("="*50)
    
    history = autoencoder.train(
        train_images,
        validation_split=0.2,
        epochs=30,  # İlk test için az epoch
        batch_size=4   # Memory için küçük batch
    )
    
    # Eğitim geçmişini görselleştir
    autoencoder.plot_training_history()
    
    # Test seti üzerinde tahmin yap
    print("\n" + "="*50)
    print("TEST SONUÇLARI HESAPLANIYOR...")
    print("="*50)
    
    anomaly_scores, anomaly_maps, reconstructed = autoencoder.predict_anomaly_scores(test_images)
    
    # Skorları normalize et (0-1 aralığına)
    anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Metrikleri hesapla
    results = calculate_metrics(test_labels, anomaly_scores_norm, threshold=0.5)
    
    # Sonuçları görselleştir
    plot_metrics(results, "Autoencoder")
    
    # Anomali tespit örneklerini göster
    visualize_anomaly_detection(
        test_images, test_labels, anomaly_scores_norm, anomaly_maps,
        "Autoencoder", num_samples=6
    )
    
    # Sonuçları kaydet
    save_model_results("Autoencoder", results)
    autoencoder.save_model("autoencoder_final.h5")
    
    print(f"\n=== AUTOENCODER SONUÇLARI ===")
    print(f"AUC Score: {results['auc_score']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
