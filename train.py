"""
MNIST Handwritten Digit Recognizer
===================================
Bu script bir neural network eğitir ve modeli kaydeder.

Ne öğreneceksin:
- Veriyi yükleme ve hazırlama (preprocessing)
- Basit bir sinir ağı kurma (neural network)
- Modeli eğitme ve değerlendirme
- Modeli kaydetme

Çalıştırmak için: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os


# ─────────────────────────────────────────
# 1) VERİYİ YÜKLE
# ─────────────────────────────────────────
print("📥 MNIST verisi yükleniyor...")

# MNIST: 70.000 adet el yazısı rakam görüntüsü
# 60.000 eğitim, 10.000 test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"   Eğitim verisi: {x_train.shape}  →  {x_train.shape[0]} görüntü, her biri {x_train.shape[1]}x{x_train.shape[2]} piksel")
print(f"   Test verisi  : {x_test.shape}   →  {x_test.shape[0]} görüntü")


# ─────────────────────────────────────────
# 2) VERİYİ HAZIRLA (Preprocessing)
# ─────────────────────────────────────────
print("\n⚙️  Veri hazırlanıyor...")

# Piksel değerleri 0-255 arasında → 0.0-1.0 arasına normalize et
# Neden? Küçük sayılar neural network'ün daha hızlı öğrenmesini sağlar
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# 28x28 matrisini 784 elemanlı düz bir vektöre dönüştür (flatten)
# Neden? Dense layer tek boyutlu girdi bekler
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# Etiketleri one-hot encoding'e çevir
# Örnek: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

print(f"   Eğitim verisi şekli: {x_train.shape}")
print(f"   Etiket şekli       : {y_train.shape}")


# ─────────────────────────────────────────
# 3) MODELİ OLUŞTUR
# ─────────────────────────────────────────
print("\n🧠 Model oluşturuluyor...")

model = keras.Sequential([
    # Giriş katmanı: 784 piksel alır, 128 nöron
    layers.Dense(128, activation="relu", input_shape=(784,)),

    # Dropout: eğitim sırasında %20 nöronu rastgele kapat
    # Neden? Overfitting'i önler, model daha genelleyici olur
    layers.Dropout(0.2),

    # Gizli katman: 64 nöron
    layers.Dense(64, activation="relu"),

    layers.Dropout(0.2),

    # Çıkış katmanı: 10 rakam (0-9) için 10 nöron
    # Softmax: çıktıları olasılığa çevirir (toplamları 1'dir)
    layers.Dense(10, activation="softmax"),
])

# Modeli derle
model.compile(
    optimizer="adam",          # Ağırlıkları güncelleyen algoritma
    loss="categorical_crossentropy",  # Sınıflandırma için kayıp fonksiyonu
    metrics=["accuracy"],      # Takip edeceğimiz metrik
)

model.summary()


# ─────────────────────────────────────────
# 4) MODELİ EĞİT
# ─────────────────────────────────────────
print("\n🚀 Model eğitiliyor...")

history = model.fit(
    x_train, y_train,
    epochs=10,           # Tüm veriyi 10 kez gör
    batch_size=128,      # Her adımda 128 örnek kullan
    validation_split=0.1, # Eğitim verisinin %10'unu doğrulama için ayır
    verbose=1,
)


# ─────────────────────────────────────────
# 5) DEĞERLENDİR
# ─────────────────────────────────────────
print("\n📊 Model değerlendiriliyor...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"   Test Accuracy (Doğruluk): {test_acc:.4f}  →  %{test_acc*100:.2f}")
print(f"   Test Loss    (Kayıp)    : {test_loss:.4f}")


# ─────────────────────────────────────────
# 6) SONUÇLARI GRAFİKLE
# ─────────────────────────────────────────
print("\n📈 Grafik kaydediliyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy grafiği
ax1.plot(history.history["accuracy"],     label="Eğitim", color="blue")
ax1.plot(history.history["val_accuracy"], label="Doğrulama", color="orange")
ax1.set_title("Model Doğruluğu (Accuracy)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss grafiği
ax2.plot(history.history["loss"],     label="Eğitim", color="blue")
ax2.plot(history.history["val_loss"], label="Doğrulama", color="orange")
ax2.set_title("Model Kaybı (Loss)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("   → training_history.png kaydedildi")


# ─────────────────────────────────────────
# 7) MODELİ KAYDET
# ─────────────────────────────────────────
model.save("mnist_model.h5")
print("\n✅ Model 'mnist_model.h5' olarak kaydedildi!")
print(f"\n🎉 Eğitim tamamlandı! Test doğruluğu: %{test_acc*100:.2f}")
