"""
MNIST - Tahmin Scripti
========================
Eğitilmiş modeli yükler ve tahmin yapar.

Kullanım:
  python predict.py               → Test setinden rastgele 10 örnek gösterir
  python predict.py --image 5     → Test setinden 5. örneği tahmin eder
  python predict.py --show-errors → Yanlış tahminleri gösterir

Çalıştırmadan önce: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse
import sys


def load_model_and_data():
    """Modeli ve test verisini yükle."""
    try:
        model = keras.models.load_model("mnist_model.h5")
        print("✅ Model yüklendi: mnist_model.h5")
    except FileNotFoundError:
        print("❌ Hata: mnist_model.h5 bulunamadı!")
        print("   Önce 'python train.py' çalıştır.")
        sys.exit(1)

    # Test verisini yükle
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test_norm = x_test.astype("float32") / 255.0
    x_test_flat = x_test_norm.reshape(-1, 784)

    return model, x_test, x_test_flat, y_test


def show_predictions(model, x_test, x_test_flat, y_test, n=10):
    """Test setinden n adet rastgele örnek seç ve tahmin et."""
    indices = np.random.choice(len(x_test), n, replace=False)

    predictions = model.predict(x_test_flat[indices], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = y_test[indices]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("MNIST Tahminleri", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[indices[i]], cmap="gray")

        pred  = pred_labels[i]
        true  = true_labels[i]
        conf  = predictions[i][pred] * 100  # Güven yüzdesi

        color = "green" if pred == true else "red"
        ax.set_title(
            f"Tahmin: {pred} ({conf:.0f}%)\nGerçek: {true}",
            color=color, fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()
    print("\n📊 predictions.png kaydedildi")

    # Doğruluk oranını hesapla
    correct = np.sum(pred_labels == true_labels)
    print(f"✅ {n} örnekten {correct} doğru tahmin ({correct/n*100:.0f}%)")


def show_wrong_predictions(model, x_test, x_test_flat, y_test, n=20):
    """Modelin yanlış tahmin ettiği örnekleri göster."""
    all_preds = model.predict(x_test_flat, verbose=0)
    pred_labels = np.argmax(all_preds, axis=1)
    wrong_mask = pred_labels != y_test
    wrong_indices = np.where(wrong_mask)[0]

    print(f"\n❌ Toplam yanlış tahmin: {len(wrong_indices)} / {len(y_test)}")
    print(f"   Doğruluk: %{(1 - len(wrong_indices)/len(y_test))*100:.2f}")

    # İlk n yanlışı göster
    sample = wrong_indices[:n]
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle("Yanlış Tahminler", fontsize=14, fontweight="bold", color="red")

    for i, ax in enumerate(axes.flat):
        if i < len(sample):
            idx = sample[i]
            ax.imshow(x_test[idx], cmap="gray")
            pred = pred_labels[idx]
            true = y_test[idx]
            conf = all_preds[idx][pred] * 100
            ax.set_title(f"Tahmin: {pred} ({conf:.0f}%)\nGerçek: {true}", color="red", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("wrong_predictions.png", dpi=150)
    plt.show()
    print("📊 wrong_predictions.png kaydedildi")


def predict_single(model, x_test, x_test_flat, y_test, index):
    """Tek bir örnek için detaylı tahmin göster."""
    x = x_test_flat[index:index+1]
    pred_probs = model.predict(x, verbose=0)[0]
    pred_label = np.argmax(pred_probs)
    true_label = y_test[index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Sol: görüntü
    ax1.imshow(x_test[index], cmap="gray")
    color = "green" if pred_label == true_label else "red"
    ax1.set_title(
        f"Tahmin: {pred_label} | Gerçek: {true_label}",
        color=color, fontsize=12, fontweight="bold"
    )
    ax1.axis("off")

    # Sağ: olasılık bar grafiği
    bars = ax2.bar(range(10), pred_probs, color=["green" if i == pred_label else "steelblue" for i in range(10)])
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Rakam")
    ax2.set_ylabel("Olasılık")
    ax2.set_title("Her Rakam İçin Tahmin Olasılığı")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    # En yüksek barın üstüne değer yaz
    ax2.text(pred_label, pred_probs[pred_label] + 0.02, f"{pred_probs[pred_label]:.2%}",
             ha="center", fontweight="bold", color="green")

    plt.tight_layout()
    plt.savefig(f"single_prediction_{index}.png", dpi=150)
    plt.show()
    print(f"\n📊 single_prediction_{index}.png kaydedildi")
    print(f"   Tahmin: {pred_label}  |  Gerçek: {true_label}  |  Güven: {pred_probs[pred_label]:.2%}")


# ─────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Tahmin Aracı")
    parser.add_argument("--image", type=int, default=None, help="Test setinden belirli bir örneğin indexi")
    parser.add_argument("--show-errors", action="store_true", help="Yanlış tahminleri göster")
    parser.add_argument("--n", type=int, default=10, help="Gösterilecek örnek sayısı (varsayılan: 10)")
    args = parser.parse_args()

    model, x_test, x_test_flat, y_test = load_model_and_data()

    if args.show_errors:
        show_wrong_predictions(model, x_test, x_test_flat, y_test, n=args.n)
    elif args.image is not None:
        predict_single(model, x_test, x_test_flat, y_test, args.image)
    else:
        show_predictions(model, x_test, x_test_flat, y_test, n=args.n)
