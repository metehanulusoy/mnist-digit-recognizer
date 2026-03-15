# 🔢 MNIST Handwritten Digit Recognizer

El yazısı rakamları tanıyan bir Neural Network — **%98+ doğruluk** ile.

> 2. sınıf Bilgisayar Mühendisliği öğrencisi olarak yaptığım ilk ML projesi.

---

## 📌 Proje Hakkında

Bu proje, [MNIST](http://yann.lecun.com/exdb/mnist/) veri setini kullanarak 0-9 arası el yazısı rakamları tanıyan bir sinir ağı modeli eğitmektedir.

**Veri seti:** 70.000 görüntü (60.000 eğitim + 10.000 test), 28×28 piksel, gri tonlamalı.

---

## 🧠 Model Mimarisi

```
Girdi (784)  →  Dense(128, ReLU)  →  Dropout(0.2)
             →  Dense(64, ReLU)   →  Dropout(0.2)
             →  Dense(10, Softmax)
```

| Katman | Nöron | Aktivasyon | Açıklama |
|--------|-------|-----------|----------|
| Giriş  | 784   | —         | 28×28 piksel düzleştirildi |
| Gizli 1 | 128  | ReLU      | Ana özellik öğrenimi |
| Gizli 2 | 64   | ReLU      | Özellik sıkıştırma |
| Çıkış  | 10    | Softmax   | 0-9 arası olasılıklar |

---

## 📊 Sonuçlar

| Metrik | Değer |
|--------|-------|
| Test Accuracy | ~%98.2 |
| Test Loss | ~0.06 |
| Eğitim Süresi | ~2 dakika (CPU) |

---

## 🚀 Kurulum ve Kullanım

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Modeli Eğit

```bash
python train.py
```

→ `mnist_model.h5` ve `training_history.png` oluşturulur.

### 3. Tahmin Yap

```bash
# Test setinden rastgele 10 örnek göster
python predict.py

# Belirli bir örneği tahmin et (index: 0-9999)
python predict.py --image 42

# Yanlış tahminleri analiz et
python predict.py --show-errors
```

### 4. Detaylı Değerlendirme

```bash
python evaluate.py
```

→ Confusion matrix ve rakam bazında doğruluk raporu.

---

## 📁 Dosya Yapısı

```
mnist-digit-recognizer/
├── train.py            # Model eğitimi
├── predict.py          # Tahmin ve görselleştirme
├── evaluate.py         # Detaylı değerlendirme
├── requirements.txt    # Gerekli kütüphaneler
├── README.md           # Bu dosya
│
# Eğitim sonrası oluşur:
├── mnist_model.h5      # Kaydedilmiş model
├── training_history.png
├── predictions.png
└── confusion_matrix.png
```

---

## 📚 Öğrenilen Kavramlar

- **Neural Network** temelleri (Dense layers, aktivasyon fonksiyonları)
- **Overfitting** ve Dropout ile önlenmesi
- **One-hot encoding** — etiket dönüşümü
- **Normalizasyon** — veri ön işleme
- **Confusion Matrix** ile model analizi
- **train/test split** — model değerlendirme

---

## 🔧 Teknolojiler

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-green)

---

## 📈 Geliştirme Fikirleri

- [ ] CNN (Convolutional Neural Network) ile %99+ doğruluk
- [ ] Flask/Gradio ile web arayüzü — tarayıcıda çizin, tahmin edin
- [ ] Kendi el yazınızla test etme

---

## 👤 Yazar

**Metehan Ulusoy** · Bilgisayar Mühendisliği 2. Sınıf
