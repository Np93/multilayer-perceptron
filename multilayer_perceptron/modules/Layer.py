import numpy as np
from utils import add_bias_units, xavier_init, get_activation_function

class Layer:
    def __init__(self, in_size, out_size, activation='sigmoid'):
        self.activation, self.activation_derivative = get_activation_function(activation)
        self.in_size = in_size
        self.out_size = out_size
        self.w = np.random.randn(in_size + 1, out_size) * np.sqrt(2 / (in_size + out_size))  # Xavier initialization
        self.b = np.zeros((1, out_size))  # Bias term

    def forward(self, x):
        self.x = np.hstack([x, np.ones((x.shape[0], 1))])  # Add bias unit
        self.z = np.dot(self.x, self.w)
        self.a = self.activation(self.z)
        print(f"Forward pass: x shape {self.x.shape}, w shape {self.w.shape}, z shape {self.z.shape}, a shape {self.a.shape}")
        return self.a

    def backward(self, djda):
        dadz = self.activation_derivative(self.z, self.a)
        djdz = djda * dadz
        djdw = np.dot(self.x.T, djdz) / self.x.shape[0]
        next_djda = np.dot(djdz, self.w[:-1].T)  # Exclude bias term from backpropagation
        print(f"Backward pass: djda shape {djda.shape}, dadz shape {dadz.shape}, djdz shape {djdz.shape}, djdw shape {djdw.shape}, next_djda shape {next_djda.shape}")
        return next_djda, djdw, np.mean(djdz, axis=0)  # Include mean bias gradient

    def __str__(self):
        return f"Layer(in_size={self.in_size}, out_size={self.out_size}, activation={self.activation.__name__})"