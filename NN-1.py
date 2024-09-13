# the raw version of this code was implemented by me from scratch, it had some errors that were resolved using perplexing.ai
import numpy as np
import pandas as pd
from scipy.special import softmax

# Load the MNIST dataset
data = pd.read_csv('train.csv')
print(data.shape)

data = np.array(data)
X = data[:, 1:]
Y = data[:, 0]
print(X.shape)
print(Y.shape)

# Normalization
X = (X - np.mean(X)) / np.std(X)

# He Initialization
W1 = np.random.randn(15, 784) * np.sqrt(2. / 784)  # Use np.random.randn for normal distribution
B1 = np.zeros((1, 15))  # Initialize biases to zero
W2 = np.random.randn(10, 15) * np.sqrt(2. / 15)  # Adjusted for the number of neurons in the previous layer
B2 = np.zeros((1, 10))  # Initialize biases to zero

def ReLU(Z):
    return np.maximum(Z, 0)

def deriv_relu(Z):
    return Z > 0

def compute_softmax(Z):
    return softmax(Z, axis=1)

def forward_prop(W1, W2, B1, B2, X):
    Z1 = np.dot(X, W1.T) + B1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2.T) + B2
    A2 = compute_softmax(Z2)
    return Z1, Z2, A1, A2

def one_hot(y):
    arr = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        index = y[i]  # No need to subtract 1
        arr[i, index] = 1
    return arr

def backward_prop(alpha, A2, A1, Z1, Z2, W2, W1, B2, B1, y):
    dA2 = A2 - one_hot(y)
    dZ2 = dA2  # Softmax derivative
    dW2 = dZ2.T.dot(A1) / A1.shape[0]  # Average over batch size
    dB2 = np.mean(dZ2, axis=0, keepdims=True)
    dA1 = dZ2.dot(W2)
    dZ1 = dA1 * deriv_relu(Z1)
    dW1 = dZ1.T.dot(X) / X.shape[0]  # Average over batch size
    dB1 = np.mean(dZ1, axis=0, keepdims=True)

    W1 -= alpha * dW1
    B1 -= alpha * dB1
    W2 -= alpha * dW2
    B2 -= alpha * dB2

    return W1, B1, W2, B2

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.shape[0]

def get_predictions(Z):
    max_indices = np.argmax(Z, axis=1)
    return max_indices

def gradient_descent(iter, W1, W2, B1, B2, alpha, X, y):
    for i in range(iter):
        Z1, Z2, A1, A2 = forward_prop(W1, W2, B1, B2, X)
        W1, B1, W2, B2 = backward_prop(alpha, A2, A1, Z1, Z2, W2, W1, B2, B1, y)
        
        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, y)
        loss = -np.mean(np.sum(one_hot(y) * np.log(A2 + 1e-12), axis=1))  # Cross-entropy loss
        
        if i % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {i}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

# Run gradient descent
gradient_descent(6001, W1, W2, B1, B2, 0.05, X, Y)  # Adjusted learning rate