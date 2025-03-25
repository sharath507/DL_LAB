import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images to a vector of size 28*28
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build the model
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(28*28,)),  # input shape should be 28*28
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # output layer has 10 neurons for 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32)  # batch_size updated to 32 for better performance

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])
actual_label = np.argmax(y_test[0])

# Plot the first test image with its prediction
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {predicted_label}, Actual: {actual_label}')
plt.show()