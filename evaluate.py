"""
MNIST - Detaylı Değerlendirme
================================
Modelin her rakam için ne kadar başarılı olduğunu analiz eder.

Çalıştırmak için: python evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import sys


def main():
    # ── Model ve veriyi yükle ──────────────────────
    try:
        model = keras.models.load_model("mnist_model.h5")
    except FileNotFoundError:
        print("❌ mnist_model.h5 bulunamadı! Önce 'python train.py' çalıştır.")
        sys.exit(1)

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test_flat = x_test.astype("float32").reshape(-1, 784) / 255.0

    # ── Tahminleri al ──────────────────────────────
    print("🔍 Tahminler hesaplanıyor...")
    pred_probs  = model.predict(x_test_flat, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    # ── Sınıflandırma Raporu ───────────────────────
    print("\n📊 Sınıflandırma Raporu:")
    print("─" * 55)
    print(classification_report(
        y_test, pred_labels,
        target_names=[f"Rakam {i}" for i in range(10)]
    ))

    # ── Confusion Matrix ───────────────────────────
    cm = confusion_matrix(y_test, pred_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("MNIST Model Değerlendirmesi", fontsize=14, fontweight="bold")

    # Ham sayılar
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10), ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Sayılar)")
    axes[0].set_xlabel("Tahmin Edilen")
    axes[0].set_ylabel("Gerçek")

    # Yüzdeler (normalize)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10), ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Yüzdeler)")
    axes[1].set_xlabel("Tahmin Edilen")
    axes[1].set_ylabel("Gerçek")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("📈 confusion_matrix.png kaydedildi")

    # ── Her rakam için accuracy ────────────────────
    print("\n📋 Rakam Bazında Doğruluk:")
    print("─" * 30)
    for digit in range(10):
        mask = y_test == digit
        acc  = np.mean(pred_labels[mask] == digit)
        bar  = "█" * int(acc * 20)
        print(f"  Rakam {digit}: {bar:<20} {acc:.2%}")

    # ── Genel doğruluk ─────────────────────────────
    overall_acc = np.mean(pred_labels == y_test)
    print(f"\n✅ Genel Test Doğruluğu: {overall_acc:.4f}  →  %{overall_acc*100:.2f}")

    plt.show()


if __name__ == "__main__":
    main()
