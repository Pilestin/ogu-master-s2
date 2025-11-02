"""
Ahşap Anomali Tespiti - Gelişmiş Veri Önişleme
Bu script literatürde yaygın kullanılan görüntü önişleme tekniklerini içerir:
- Normalizasyon ve standardizasyon
- Gürültü azaltma (denoising)
- Kenar (edge) iyileştirme
- Histogram eşitleme
- Gereksiz bölgelerin kesilmesi (cropping)
- Veri artırma (augmentation)
- Adaptif eşikleme
- Morfolojik işlemler
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageEnhance
from skimage import exposure, filters, morphology
from scipy import ndimage
import json
from tqdm import tqdm
import shutil

# Dataset yolları
DATASET_ROOT = "dataset/wood"
TRAIN_GOOD_PATH = os.path.join(DATASET_ROOT, "train", "good")
TEST_GOOD_PATH = os.path.join(DATASET_ROOT, "test", "good")
TEST_DEFECT_PATH = os.path.join(DATASET_ROOT, "test", "defect")

# Önişlenmiş veri kayıt yolu
PREPROCESSED_ROOT = "dataset/wood_preprocessed"

class ImagePreprocessor:
    """Görüntü önişleme sınıfı"""
    
    def __init__(self, target_size=(256, 256), apply_crop=True):
        """
        Args:
            target_size: Hedef görüntü boyutu (height, width)
            apply_crop: Gereksiz bölgeleri otomatik kırp
        """
        self.target_size = target_size
        self.apply_crop = apply_crop
        
    def load_image(self, image_path):
        """Görüntü yükle"""
        image = cv2.imread(image_path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None
    
    def auto_crop_borders(self, image, threshold=10):
        """
        Görüntünün kenarlarındaki gereksiz boş/uniform bölgeleri otomatik kırp
        Bu ahşap görüntülerinde sıkça görülen kenar gürültüsünü temizler
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Kenarları tespit et
        edges = cv2.Canny(gray, 50, 150)
        
        # Kenar piksellerinin koordinatlarını bul
        coords = np.column_stack(np.where(edges > 0))
        
        if len(coords) == 0:
            return image
        
        # Bounding box hesapla
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Biraz margin ekle
        margin = 10
        h, w = image.shape[:2]
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(h, y_max + margin)
        x_max = min(w, x_max + margin)
        
        return image[y_min:y_max, x_min:x_max]

    def crop_with_otsu(self, image):
        """
        Görüntüyü Otsu's thresholding kullanarak ana nesneyi bulacak şekilde kırpar.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Gürültüyü azaltmak için blur uygulama
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's thresholding (Ahşabın arka plandan daha açık olduğu varsayılır)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Konturları bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image  # Kontur bulunamazsa orijinali döndür
            
        # En büyük konturu bul
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Sınırlayıcı kutuyu al
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Küçük bir margin ekle
        margin = 10
        h_img, w_img = image.shape[:2]
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(w_img, x + w + margin)
        y_end = min(h_img, y + h + margin)
        
        return image[y_start:y_end, x_start:x_end]

    def apply_gabor(self, image, ksize=(31, 31), sigma=5.0, theta=np.pi/4, lambd=10.0, gamma=0.5):
        """
        Doku analizi için Gabor filtresi uygular.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(gray, cv2.CV_8U, kernel)
        return cv2.cvtColor(gabor_response, cv2.COLOR_GRAY2RGB)
    
    def denoise(self, image, method='bilateral'):
        """
        Gürültü azaltma
        Methods: 'bilateral', 'gaussian', 'median', 'nlm' (Non-local Means)
        """
        if method == 'bilateral':
            # Kenarları koruyarak gürültü azaltır
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'nlm':
            # Non-local means denoising (en etkili ama en yavaş)
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return image
    
    def enhance_contrast(self, image, method='clahe'):
        """
        Kontrast iyileştirme
        Methods: 'clahe', 'histogram_eq', 'adaptive'
        """
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Her kanal için ayrı ayrı uygula
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        elif method == 'histogram_eq':
            # Global histogram eşitleme
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        elif method == 'adaptive':
            # Adaptive contrast using scikit-image
            return exposure.equalize_adapthist(image, clip_limit=0.03)
        
        return image
    
    def sharpen(self, image, strength=1.0):
        """
        Görüntüyü keskinleştir (detayları vurgula)
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        return cv2.filter2D(image, -1, kernel)
    
    def normalize_image(self, image, method='minmax'):
        """
        Görüntü normalizasyonu
        Methods: 'minmax', 'zscore', 'robust'
        """
        image_float = image.astype(np.float32)
        
        if method == 'minmax':
            # Min-Max normalizasyon [0, 1]
            normalized = (image_float - image_float.min()) / (image_float.max() - image_float.min() + 1e-8)
        
        elif method == 'zscore':
            # Z-score normalizasyon (mean=0, std=1)
            mean = image_float.mean()
            std = image_float.std() + 1e-8
            normalized = (image_float - mean) / std
            # [0, 1] aralığına getir
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        
        elif method == 'robust':
            # Robust normalizasyon (outlier'lara karşı dayanıklı)
            p2, p98 = np.percentile(image_float, (2, 98))
            normalized = np.clip((image_float - p2) / (p98 - p2 + 1e-8), 0, 1)
        
        return (normalized * 255).astype(np.uint8)
    
    def apply_morphological_operations(self, image, operation='opening'):
        """
        Morfolojik işlemler (küçük gürültüleri temizler)
        Operations: 'opening', 'closing', 'gradient'
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        if operation == 'opening':
            # Küçük gürültüleri kaldır
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            # Küçük delikleri kapat
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            # Kenarları vurgula
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        return image
    
    def resize_image(self, image, target_size=None):
        """Görüntüyü yeniden boyutlandır"""
        if target_size is None:
            target_size = self.target_size
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def preprocess_pipeline(self, image, config=None):
        """
        Tam önişleme pipeline'ı
        
        Args:
            image: Giriş görüntüsü
            config: Önişleme konfigürasyonu (dict)
        """
        if config is None:
            config = {
                'crop_method': 'otsu',  # 'canny', 'otsu', or None
                'denoise': 'bilateral',
                'contrast': 'clahe',
                'gabor': None, # {'ksize':(31,31), ...} or None
                'sharpen': True,
                'sharpen_strength': 0.5,
                'normalize': 'minmax',
                'morphology': None,  # 'opening', 'closing', None
                'resize': True
            }
        
        processed = image.copy()
        
        # 1. Otomatik kırpma
        crop_method = config.get('crop_method')
        if crop_method == 'canny':
            processed = self.auto_crop_borders(processed)
        elif crop_method == 'otsu':
            processed = self.crop_with_otsu(processed)
        
        # 2. Gürültü azaltma
        if config.get('denoise'):
            processed = self.denoise(processed, method=config['denoise'])
        
        # 3. Kontrast iyileştirme
        if config.get('contrast'):
            processed = self.enhance_contrast(processed, method=config['contrast'])

        # 4. Gabor Filtresi (Doku analizi)
        gabor_params = config.get('gabor')
        if gabor_params:
            processed = self.apply_gabor(processed, **gabor_params)
        
        # 5. Keskinleştirme
        if config.get('sharpen', False):
            strength = config.get('sharpen_strength', 0.5)
            processed = self.sharpen(processed, strength=strength)
        
        # 6. Morfolojik işlemler
        if config.get('morphology'):
            processed = self.apply_morphological_operations(
                processed, operation=config['morphology']
            )
        
        # 7. Normalizasyon
        if config.get('normalize'):
            processed = self.normalize_image(processed, method=config['normalize'])
        
        # 8. Yeniden boyutlandırma
        if config.get('resize', True):
            processed = self.resize_image(processed)
        
        return processed

class DataAugmentation:
    """Veri artırma sınıfı (training data için)"""
    
    @staticmethod
    def rotate(image, angle):
        """Görüntüyü döndür"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    @staticmethod
    def flip(image, mode='horizontal'):
        """Görüntüyü çevir"""
        if mode == 'horizontal':
            return cv2.flip(image, 1)
        elif mode == 'vertical':
            return cv2.flip(image, 0)
        elif mode == 'both':
            return cv2.flip(image, -1)
        return image
    
    @staticmethod
    def adjust_brightness(image, factor=1.2):
        """Parlaklık ayarla"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    @staticmethod
    def add_gaussian_noise(image, mean=0, std=5):
        """Gaussian gürültü ekle"""
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    @staticmethod
    def random_crop(image, crop_ratio=0.9):
        """Rastgele kırpma"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        cropped = image[top:top+new_h, left:left+new_w]
        return cv2.resize(cropped, (w, h))
    
    def augment_dataset(self, image, num_augmentations=5):
        """
        Bir görüntüden çoklu augmented versiyonlar oluştur
        """
        augmented_images = [image]  # Orijinal
        
        # Çeşitli augmentation kombinasyonları
        if num_augmentations >= 1:
            augmented_images.append(self.flip(image, 'horizontal'))
        
        if num_augmentations >= 2:
            augmented_images.append(self.rotate(image, 90))
        
        if num_augmentations >= 3:
            augmented_images.append(self.rotate(image, 180))
        
        if num_augmentations >= 4:
            augmented_images.append(self.adjust_brightness(image, 1.2))
        
        if num_augmentations >= 5:
            augmented_images.append(self.adjust_brightness(image, 0.8))
        
        return augmented_images[:num_augmentations + 1]

def visualize_preprocessing_steps(image_path, preprocessor):
    """Önişleme adımlarını görselleştir"""
    original = preprocessor.load_image(image_path)
    
    if original is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return
    
    # Farklı işlemler uygula
    cropped_otsu = preprocessor.crop_with_otsu(original)
    clahe_enhanced = preprocessor.enhance_contrast(cropped_otsu.copy(), method='clahe')
    gabor_filtered = preprocessor.apply_gabor(cropped_otsu.copy(), theta=np.pi/2)
    final_processed = preprocessor.preprocess_pipeline(original.copy())

    # Görselleştirme
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Önişleme Adımları", fontsize=16)

    steps = {
        "Orijinal": original,
        "Otsu Kırpma": cropped_otsu,
        "CLAHE": clahe_enhanced,
        "Gabor Filtresi": gabor_filtered,
        "Sonuç (Pipeline)": final_processed
    }

    for ax, (title, img) in zip(axes, steps.items()):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def process_and_save_dataset(preprocessor, config=None, apply_augmentation=False):
    """
    Tüm dataset'i işle ve kaydet
    """
    print("="*60)
    print("VERİ ÖNİŞLEME BAŞLIYOR...")
    print("="*60)
    
    # Önişlenmiş klasörleri oluştur
    os.makedirs(os.path.join(PREPROCESSED_ROOT, "train", "good"), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESSED_ROOT, "test", "good"), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESSED_ROOT, "test", "defect"), exist_ok=True)
    
    augmentor = DataAugmentation()
    stats = {
        'train_good': 0,
        'test_good': 0,
        'test_defect': 0,
        'augmented_train': 0
    }
    
    # Training set işle (isteğe bağlı augmentation ile)
    print("\n[1/3] Training set işleniyor...")
    train_files = [f for f in os.listdir(TRAIN_GOOD_PATH) if f.endswith('.bmp')]
    
    for filename in tqdm(train_files, desc="Train Good"):
        input_path = os.path.join(TRAIN_GOOD_PATH, filename)
        image = preprocessor.load_image(input_path)
        
        if image is not None:
            # Önişleme uygula
            processed = preprocessor.preprocess_pipeline(image, config)
            
            # Orijinali kaydet
            output_path = os.path.join(PREPROCESSED_ROOT, "train", "good", filename)
            cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            stats['train_good'] += 1
            
            # Augmentation uygula (sadece training data için)
            if apply_augmentation:
                augmented_images = augmentor.augment_dataset(processed, num_augmentations=3)
                for idx, aug_img in enumerate(augmented_images[1:], 1):  # İlk orijinal
                    aug_filename = filename.replace('.bmp', f'_aug{idx}.bmp')
                    aug_path = os.path.join(PREPROCESSED_ROOT, "train", "good", aug_filename)
                    cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    stats['augmented_train'] += 1
    
    # Test good set işle
    print("\n[2/3] Test good set işleniyor...")
    test_good_files = [f for f in os.listdir(TEST_GOOD_PATH) if f.endswith('.bmp')]
    
    for filename in tqdm(test_good_files, desc="Test Good"):
        input_path = os.path.join(TEST_GOOD_PATH, filename)
        image = preprocessor.load_image(input_path)
        
        if image is not None:
            processed = preprocessor.preprocess_pipeline(image, config)
            output_path = os.path.join(PREPROCESSED_ROOT, "test", "good", filename)
            cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            stats['test_good'] += 1
    
    # Test defect set işle
    print("\n[3/3] Test defect set işleniyor...")
    test_defect_files = [f for f in os.listdir(TEST_DEFECT_PATH) if f.endswith('.bmp')]
    
    for filename in tqdm(test_defect_files, desc="Test Defect"):
        input_path = os.path.join(TEST_DEFECT_PATH, filename)
        image = preprocessor.load_image(input_path)
        
        if image is not None:
            processed = preprocessor.preprocess_pipeline(image, config)
            output_path = os.path.join(PREPROCESSED_ROOT, "test", "defect", filename)
            cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            stats['test_defect'] += 1
    
    # Konfigürasyonu kaydet
    config_path = os.path.join(PREPROCESSED_ROOT, "preprocessing_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*60)
    print("ÖNİŞLEME TAMAMLANDI!")
    print("="*60)
    print(f"\nİstatistikler:")
    print(f"  Train Good: {stats['train_good']} görüntü")
    if apply_augmentation:
        print(f"  Train Good (Augmented): {stats['augmented_train']} ek görüntü")
        print(f"  Toplam Train: {stats['train_good'] + stats['augmented_train']} görüntü")
    print(f"  Test Good: {stats['test_good']} görüntü")
    print(f"  Test Defect: {stats['test_defect']} görüntü")
    print(f"\nKaydedilen konum: {PREPROCESSED_ROOT}")
    print(f"Konfigürasyon: {config_path}")
    
    return stats

def compare_preprocessing_configs(image_path):
    """Farklı önişleme konfigürasyonlarını karşılaştır"""
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    original = preprocessor.load_image(image_path)
    
    if original is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return
    
    # Farklı konfigürasyonlar
    configs = {
        'Minimal': {
            'auto_crop': True,
            'denoise': None,
            'contrast': None,
            'sharpen': False,
            'normalize': 'minmax',
            'resize': True
        },
        'Balanced': {
            'auto_crop': True,
            'denoise': 'bilateral',
            'contrast': 'clahe',
            'sharpen': True,
            'sharpen_strength': 0.5,
            'normalize': 'minmax',
            'resize': True
        },
        'Aggressive': {
            'auto_crop': True,
            'denoise': 'nlm',
            'contrast': 'clahe',
            'sharpen': True,
            'sharpen_strength': 1.0,
            'normalize': 'robust',
            'morphology': 'opening',
            'resize': True
        }
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Farklı Önişleme Konfigürasyonları', fontsize=16)
    
    axes[0].imshow(preprocessor.resize_image(original))
    axes[0].set_title('Orijinal', fontsize=12)
    axes[0].axis('off')
    
    for idx, (name, config) in enumerate(configs.items(), 1):
        processed = preprocessor.preprocess_pipeline(original, config)
        axes[idx].imshow(processed)
        axes[idx].set_title(name, fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Ana fonksiyon"""
    print("="*60)
    print("AHŞAP ANOMALİ TESPİTİ - GELİŞMİŞ VERİ ÖNİŞLEME")
    print("="*60)
    
    # Örnek görüntü seç
    train_files = [f for f in os.listdir(TRAIN_GOOD_PATH) if f.endswith('.bmp')]
    sample_path = os.path.join(TRAIN_GOOD_PATH, train_files[0])
    
    print(f"\nÖrnek görüntü: {os.path.basename(sample_path)}")
    
    # Preprocessor oluştur
    preprocessor = ImagePreprocessor(target_size=(256, 256), apply_crop=True)
    
    # 1. Önişleme adımlarını görselleştir
    print("\n[1] Önişleme adımlarını görselleştiriliyor...")
    visualize_preprocessing_steps(sample_path, preprocessor)
    
    # 2. Farklı konfigürasyonları karşılaştır
    print("\n[2] Farklı konfigürasyonlar karşılaştırılıyor...")
    compare_preprocessing_configs(sample_path)
    
    # 3. Kullanıcıdan seçim al
    print("\n" + "="*60)
    print("Önişleme Konfigürasyonu Seçin:")
    print("="*60)
    print("1. Minimal     - Sadece temel işlemler (hızlı)")
    print("2. Balanced    - Dengeli işlemler (önerilen)")
    print("3. Aggressive  - Kapsamlı işlemler (yavaş ama detaylı)")
    print("4. Custom      - Özel konfigürasyon")
    print("="*60)
    
    choice = input("\nSeçiminiz (1-4) [varsayılan: 2]: ").strip() or "2"
    
    # Konfigürasyon seç
    if choice == "1":
        config = {
            'auto_crop': True,
            'denoise': None,
            'contrast': None,
            'sharpen': False,
            'normalize': 'minmax',
            'resize': True
        }
        config_name = "Minimal"
    elif choice == "3":
        config = {
            'auto_crop': True,
            'denoise': 'nlm',
            'contrast': 'clahe',
            'sharpen': True,
            'sharpen_strength': 1.0,
            'normalize': 'robust',
            'morphology': 'opening',
            'resize': True
        }
        config_name = "Aggressive"
    elif choice == "4":
        # Custom konfigürasyon
        print("\nÖzel konfigürasyon için varsayılan değerler kullanılacak.")
        config = {
            'auto_crop': True,
            'denoise': 'bilateral',
            'contrast': 'clahe',
            'sharpen': True,
            'sharpen_strength': 0.5,
            'normalize': 'minmax',
            'resize': True
        }
        config_name = "Custom"
    else:  # Default: 2
        config = {
            'auto_crop': True,
            'denoise': 'bilateral',
            'contrast': 'clahe',
            'sharpen': True,
            'sharpen_strength': 0.5,
            'normalize': 'minmax',
            'resize': True
        }
        config_name = "Balanced"
    
    print(f"\nSeçilen konfigürasyon: {config_name}")
    
    # Augmentation sorusu
    aug_choice = input("\nTraining data için augmentation uygulansın mı? (e/h) [varsayılan: e]: ").strip().lower() or "e"
    apply_augmentation = (aug_choice == "e")
    
    # Dataset'i işle ve kaydet
    print("\n" + "="*60)
    stats = process_and_save_dataset(preprocessor, config, apply_augmentation)
    
    print("\n" + "="*60)
    print("İŞLEM TAMAMLANDI!")
    print("="*60)
    print(f"\nÖnişlenmiş veriler '{PREPROCESSED_ROOT}' klasörüne kaydedildi.")
    print("\nBu verileri model eğitiminde kullanabilirsiniz.")
    print("Model scriptlerinizde DATASET_ROOT değişkenini değiştirmeyi unutmayın:")
    print(f'  DATASET_ROOT = "{PREPROCESSED_ROOT}"')

if __name__ == "__main__":
    main()
