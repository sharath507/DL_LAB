import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



def standard_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def depthwise_separable_cnn():
    model = models.Sequential([
        layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def dilated_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2, input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def random_features_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu', kernel_initializer='random_normal'),  # Random initialization
        layers.Dense(10, activation='softmax')
    ])
    return model


def train_and_evaluate(model, name):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"\nTraining {name} model...")
    history = model.fit(x_train, y_train, epochs=4, batch_size=64, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"{name} Test accuracy: {test_acc:.4f}")
    return history


models_dict = {
    "Standard CNN": standard_cnn(),
    "Depthwise CNN": depthwise_separable_cnn(),
    "Dilated CNN": dilated_cnn(),
    "Random Feature CNN": random_features_cnn()
}

histories = {}
for name, model in models_dict.items():
    histories[name] = train_and_evaluate(model, name)

plt.figure(figsize=(10, 5))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.title("Comparison of Different CNN Architectures")
plt.show()






# output:

# Training Standard CNN model...
# Epoch 1/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 119s 150ms/step - accuracy: 0.2526 - loss: 1.9622 - val_accuracy: 0.4329 - val_loss: 1.5528
# Epoch 2/4

# 782/782 ━━━━━━━━━━━━━━━━━━━━ 143s 152ms/step - accuracy: 0.4374 - loss: 1.5298 - val_accuracy: 0.4896 - val_loss: 1.3970
# Epoch 3/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 142s 152ms/step - accuracy: 0.4954 - loss: 1.3815 - val_accuracy: 0.5136 - val_loss: 1.3318
# Epoch 4/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 139s 148ms/step - accuracy: 0.5460 - loss: 1.2515 - val_accuracy: 0.5809 - val_loss: 1.1711
# 313/313 - 6s - 20ms/step - accuracy: 0.5809 - loss: 1.1711
# Standard CNN Test accuracy: 0.5809

# Training Depthwise CNN model...
# Epoch 1/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 71s 87ms/step - accuracy: 0.1687 - loss: 2.1621 - val_accuracy: 0.2847 - val_loss: 1.8654
# Epoch 2/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 80s 85ms/step - accuracy: 0.2912 - loss: 1.8509 - val_accuracy: 0.3378 - val_loss: 1.7600
# Epoch 3/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 69s 89ms/step - accuracy: 0.3393 - loss: 1.7477 - val_accuracy: 0.3626 - val_loss: 1.7004
# Epoch 4/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 85s 93ms/step - accuracy: 0.3725 - loss: 1.6770 - val_accuracy: 0.4028 - val_loss: 1.6152
# 313/313 - 5s - 15ms/step - accuracy: 0.4028 - loss: 1.6152
# Depthwise CNN Test accuracy: 0.4028

# Training Dilated CNN model...
# Epoch 1/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 134s 169ms/step - accuracy: 0.2630 - loss: 1.9277 - val_accuracy: 0.4265 - val_loss: 1.5351
# Epoch 2/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 131s 167ms/step - accuracy: 0.4363 - loss: 1.5198 - val_accuracy: 0.4984 - val_loss: 1.3589
# Epoch 3/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 140s 165ms/step - accuracy: 0.5038 - loss: 1.3644 - val_accuracy: 0.5243 - val_loss: 1.2804
# Epoch 4/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 129s 164ms/step - accuracy: 0.5494 - loss: 1.2401 - val_accuracy: 0.5828 - val_loss: 1.1503
# 313/313 - 12s - 38ms/step - accuracy: 0.5828 - loss: 1.1503
# Dilated CNN Test accuracy: 0.5828

# Training Random Feature CNN model...
# Epoch 1/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 124s 155ms/step - accuracy: 0.2326 - loss: 2.0016 - val_accuracy: 0.4048 - val_loss: 1.6233
# Epoch 2/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 137s 149ms/step - accuracy: 0.4298 - loss: 1.5426 - val_accuracy: 0.5034 - val_loss: 1.3531
# Epoch 3/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 145s 153ms/step - accuracy: 0.4970 - loss: 1.3678 - val_accuracy: 0.5084 - val_loss: 1.3311
# Epoch 4/4
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 143s 155ms/step - accuracy: 0.5330 - loss: 1.2845 - val_accuracy: 0.5746 - val_loss: 1.1787
# 313/313 - 8s - 25ms/step - accuracy: 0.5746 - loss: 1.1787
# Random Feature CNN Test accuracy: 0.5746
