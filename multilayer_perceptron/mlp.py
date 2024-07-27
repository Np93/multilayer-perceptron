import numpy as np
from multilayer_perceptron.layers import DenseLayer
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.train_losses = []
        self.val_losses = []

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backpropagate(self, X, y, learning_rate):
        m = X.shape[0]
        y_pred = self.forward(X)
        delta = y_pred - y.reshape(-1, 1)

        for layer in reversed(self.layers):
            if layer.activation == 'relu':
                delta = delta * (layer.a > 0)  # Derivative of ReLU
            grad_w = np.dot(layer.inputs.T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            delta = np.dot(delta, layer.weights.T)
            layer.weights -= learning_rate * grad_w
            layer.bias -= learning_rate * grad_b

    def fit(self, X_train, y_train, X_valid, y_valid, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagate(X_train, y_train, learning_rate)
            y_train_pred = self.forward(X_train)
            y_valid_pred = self.forward(X_valid)
            train_loss = self.compute_loss(y_train, y_train_pred)
            val_loss = self.compute_loss(y_valid, y_valid_pred)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def plot_learning_curves(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()