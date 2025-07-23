# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:04:40 2025

@author: MONSTER
"""

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def plaka_on_isleme(img):
    """Plaka tespiti için görüntü ön işleme"""
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azaltma
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Histogram eşitleme ile kontrast artırma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def kenar_tespit(img):
    """Gelişmiş kenar tespiti"""
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Adaptif threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Canny kenar tespiti
    edges = cv2.Canny(morph, 50, 150, apertureSize=3)
    
    return edges

def plaka_kontrol(contour, img_shape):
    """Konturun plaka olup olmadığını kontrol et"""
    # Minimum alan kontrolü
    area = cv2.contourArea(contour)
    if area < 1000:
        return False, 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Aspect ratio kontrolü (Türk plakaları için)
    aspect_ratio = w / h
    if not (2.5 < aspect_ratio < 6.0):
        return False, 0
    
    # Görüntü içinde kalma kontrolü
    if x < 0 or y < 0 or x + w > img_shape[1] or y + h > img_shape[0]:
        return False, 0
    
    # Boyut kontrolü
    img_area = img_shape[0] * img_shape[1]
    if area < img_area * 0.01 or area > img_area * 0.3:
        return False, 0
    
    # Dikdörtgensellik kontrolü
    rect_area = w * h
    extent = area / rect_area
    if extent < 0.6:
        return False, 0
    
    # Skor hesapla
    score = extent * min(aspect_ratio / 4.0, 1.0) * min(area / 5000, 1.0)
    
    return True, score

def plaka_dogrula(roi):
    """ROI'nin gerçekten plaka olup olmadığını doğrula"""
    if roi.shape[0] < 20 or roi.shape[1] < 50:
        return False
    
    # Histogram analizi
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    
    # Çok karanlık veya çok aydınlık bölgeleri reddet
    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[200:])
    total_pixels = roi.shape[0] * roi.shape[1]
    
    if dark_pixels / total_pixels > 0.7 or bright_pixels / total_pixels > 0.7:
        return False
    
    # Kenar yoğunluğu kontrolü
    edges = cv2.Canny(roi, 50, 150)
    edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
    
    return 0.1 < edge_density < 0.5

def plaka_tespit_et(img_path):
    """Ana plaka tespit fonksiyonu"""
    print(f"\n🔍 Analiz ediliyor: {os.path.basename(img_path)}")
    
    # Görüntüyü yükle
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Görüntü yüklenemedi!")
        return None
    
    # Orijinal boyutları sakla
    original_height, original_width = img.shape[:2]
    
    # Görüntüyü yeniden boyutlandır (çok büyükse)
    if original_width > 1200:
        scale = 1200 / original_width
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    # Ön işleme
    processed = plaka_on_isleme(img)
    
    # Kenar tespiti
    edges = kenar_tespit(processed)
    
    # Kontur bulma
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Konturları filtrele ve skorla
    plaka_adaylari = []
    
    for contour in contours:
        is_valid, score = plaka_kontrol(contour, img.shape)
        if is_valid:
            x, y, w, h = cv2.boundingRect(contour)
            roi = processed[y:y+h, x:x+w]
            
            if plaka_dogrula(roi):
                plaka_adaylari.append((contour, score, (x, y, w, h)))
    
    # En iyi adayı seç
    if plaka_adaylari:
        # Skora göre sırala
        plaka_adaylari.sort(key=lambda x: x[1], reverse=True)
        
        # En iyi 3 adayı göster
        result_img = img.copy()
        
        for i, (contour, score, (x, y, w, h)) in enumerate(plaka_adaylari[:3]):
            if i == 0:
                # En iyi aday - yeşil
                color = (0, 255, 0)
                thickness = 3
                label = f"PLAKA (Skor: {score:.2f})"
            else:
                # Diğer adaylar - sarı
                color = (0, 255, 255)
                thickness = 2
                label = f"Aday {i+1} (Skor: {score:.2f})"
            
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            print(f"  ✅ {label} - Konum: ({x},{y}) Boyut: {w}x{h}")
        
        return result_img
    else:
        print("  ❌ Plaka tespit edilemedi")
        return img

def main():
    """Ana fonksiyon"""
    klasor = "plaka"
    
    # Klasör kontrolü
    if not os.path.exists(klasor):
        print(f"❌ '{klasor}' klasörü bulunamadı!")
        print("Lütfen 'plaka' klasörü oluşturun ve içine test görüntülerini koyun.")
        return
    
    # Desteklenen dosya formatları
    desteklenen_formatlar = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Görüntü dosyalarını bul
    resim_dosyalari = [f for f in os.listdir(klasor) 
                       if f.lower().endswith(desteklenen_formatlar)]
    
    if not resim_dosyalari:
        print(f"❌ '{klasor}' klasöründe görüntü dosyası bulunamadı!")
        return
    
    print(f"📁 {len(resim_dosyalari)} görüntü dosyası bulundu")
    
    # Her görüntüyü işle
    for resim_adi in resim_dosyalari:
        resim_yolu = os.path.join(klasor, resim_adi)
        
        # Plaka tespiti yap
        sonuc = plaka_tespit_et(resim_yolu)
        
        if sonuc is not None:
            # Sonucu göster
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
            plt.title(f"Plaka Tespit Sonucu: {resim_adi}", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        print("-" * 50)

if __name__ == "__main__":
    main()