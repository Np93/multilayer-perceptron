import numpy as np
from .Loss import CrossEntropyLoss

class Optimizer:
    def __init__(self, learning_rate=0.01, loss=CrossEntropyLoss()):
        self.loss = loss
        self.lr = learning_rate

    def update_weights(self, gradient, bias_gradient, layer):
        layer.w -= self.lr * gradient
        layer.b -= self.lr * bias_gradient

    def fit(self, layers, y):
        y = y.reshape(-1, 1)
        local_gradient = self.loss.loss_derivative(layers[-1].a, y)
        for l in reversed(layers):
            local_gradient, weights_gradient, bias_gradient = l.backward(local_gradient)
            self.update_weights(weights_gradient, bias_gradient, l)

class NAGOptimizer(Optimizer):
    def __init__(self, learning_rate=0.005, momentum=0.9, loss=CrossEntropyLoss()):
        super().__init__(learning_rate, loss)
        self.momentum = momentum
        self.velocity = None

    def update_weights(self, gradient, bias_gradient, layer, velocity_w, velocity_b):
        if velocity_w.shape != gradient.shape:
            velocity_w = np.zeros_like(gradient)
        if velocity_b.shape != bias_gradient.shape:
            velocity_b = np.zeros_like(bias_gradient)
        velocity_w = self.momentum * velocity_w - self.lr * gradient
        velocity_b = self.momentum * velocity_b - self.lr * bias_gradient
        layer.w += velocity_w
        layer.b += velocity_b
        # print(f"Updated weights: w shape {layer.w.shape}, b shape {layer.b.shape}, velocity_w shape {velocity_w.shape}, velocity_b shape {velocity_b.shape}")
        return velocity_w, velocity_b

    def fit(self, layers, y):
        y_one_hot = np.eye(layers[-1].w.shape[1])[y.astype(int)]
        if self.velocity == None:
            self.velocity = [(np.zeros_like(l.w), np.zeros_like(l.b)) for l in layers]
        local_gradient = self.loss.loss_derivative(layers[-1].a, y_one_hot)
        for i, l in enumerate(reversed(layers)):
            local_gradient, weights_gradient, bias_gradient = l.backward(local_gradient)
            self.velocity[-(i+1)] = self.update_weights(weights_gradient, bias_gradient, l, *self.velocity[-(i+1)])