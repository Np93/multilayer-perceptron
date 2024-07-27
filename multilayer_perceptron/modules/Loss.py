import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-15

    def loss(self, y_hat, y):
        y_hat = np.clip(y_hat, self.epsilon, 1 - self.epsilon)
        y_one_hot = np.eye(y_hat.shape[1])[y.astype(int)]
        return -np.mean(np.sum(y_one_hot * np.log(y_hat), axis=1))

    def loss_derivative(self, y_hat, y_one_hot):
        y_hat = np.clip(y_hat, self.epsilon, 1 - self.epsilon)
        return (y_hat - y_one_hot) / y_hat.shape[0]