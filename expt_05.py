import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

def generate_sequence(seq_length = 5, num_samples = 1000):
  x, y = [], []
  for _ in range(num_samples):
    seq = np.random.choice([0, 1], size = seq_length)
    x.append(seq)
    y.append(seq[-1] + 0.1)
  return np.array(x), np.array(y)

seq_length = 5

x, y = generate_sequence(seq_length)

x = x.reshape((x.shape[0], seq_length, 1))

split = int(0.8 * len(x))
x_train, y_train = x[ :split], y[:split]
x_test, y_test = x[split:], y[split:]

model = Sequential([
    SimpleRNN(10, activation = 'relu', input_shape = (seq_length, 1)),
    Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse')

model.summary()

history = model.fit(x_train, y_train, epochs = 50, batch_size = 16, validation_data = (x_test, y_test))

test_loss = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')

sample_input = np.array([0, 1, 0, 0, 0]).reshape(1, seq_length, 1) # Input should match the sequence length
predicted_value = model.predict(sample_input)
print(f"Predicted next value: {predicted_value}")

plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()