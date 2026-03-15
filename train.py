
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os

print("📥 MNIST verisi yükleniyor...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# CNN için 28x28x1 shape lazım (flatten etmiyoruz)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

print("🧠 CNN modeli oluşturuluyor...")
model = keras.Sequential([
    # 1. Conv katmanı: görüntüdeki kenarları ve şekilleri öğrenir
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    # 2. Conv katmanı: daha karmaşık şekilleri öğrenir
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("\n🚀 CNN eğitiliyor...")
history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                    validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Doğruluğu: %{test_acc*100:.2f}")

model.save("mnist_model.h5")
print("Model kaydedildi!")
