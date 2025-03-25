


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivation(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

W1 = np.random.rand(2, 4)
W2 = np.random.rand(4, 1)

learning_rate = 0.1

for epoch in range(200000):
    hidden_layer = sigmoid(X @ W1)
    output_layer = sigmoid(hidden_layer @ W2)

    error = y - output_layer

    output_layer_delta = error * sigmoid_derivation(output_layer)
    hidden_layer_error = output_layer_delta @ W2.T
    hidden_layer_delta = hidden_layer_error * sigmoid_derivation(hidden_layer)

    W2 += hidden_layer.T @ output_layer_delta * learning_rate
    W1 += X.T @ hidden_layer_delta * learning_rate

    if epoch % 50000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

print("Final output after training:")
print(output_layer)