# Artificial Neural Network using Backpropagation
# Dataset: Iris Dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Normalize the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Convert target labels to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# 5. Initialize network parameters
np.random.seed(42)

input_neurons = 4
hidden_neurons = 8
output_neurons = 3
learning_rate = 0.1
epochs = 1000

W1 = np.random.randn(input_neurons, hidden_neurons) * 0.1
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons) * 0.1
b2 = np.zeros((1, output_neurons))

# 6. Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 7. Train the ANN using Backpropagation
loss_history = []

for epoch in range(epochs):

    # Forward propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Calculate cross-entropy loss
    loss = -np.mean(
        np.sum(y_train * np.log(A2 + 1e-9), axis=1)
    )
    loss_history.append(loss)

    # Backpropagation
    m = X_train.shape[0]

    dZ2 = (A2 - y_train) / m
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# 8. Test the trained ANN
Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

y_pred = np.argmax(A2_test, axis=1)
y_actual = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_actual, y_pred)

print("ANN Classification Results")
print("---------------------------")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(
    y_actual, y_pred, target_names=iris.target_names
))

# 9. Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("ANN Training Loss using Backpropagation")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()
