import numpy as np
from .Layer import Layer
from .Optimizer import Optimizer, NAGOptimizer
from .Loss import CrossEntropyLoss
from .Grapher import Grapher
from utils import delete_dir_and_contents
import os
import pickle
import pandas as pd
from collections import deque

class Model:
    def __init__(self, sizes, activations, optimizer=Optimizer(learning_rate=0.1, loss=CrossEntropyLoss())):
        self.layers = self.make_layer_list_from_sizes_and_activations(sizes, activations)
        self.optimizer = optimizer
        self.optimizer.layers = self.layers
        self.sizes = sizes
        self.activations = activations
        self.grapher = Grapher()

    def make_layer_list_from_sizes_and_activations(self, sizes, activations):
        layers = []
        input_size = sizes[0]
        for size, activation in zip(sizes[1:], activations):
            layers.append(Layer(input_size, size, activation))
            input_size = size
        return layers

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, x, y):
        y_hat = self.feed_forward(x)
        # print(f"y_hat shape: {y_hat.shape}, y shape: {y.shape}")
        self.optimizer.fit(self.layers, y)

    def train(self, dataset, batch_size=64, epochs=300, early_stopping_patience=5):
        losses = deque(maxlen=early_stopping_patience)
        for epoch in range(epochs):
            for x_batch, y_batch in dataset.batchiterator(batch_size):
                self.fit(x_batch, y_batch)
            train_loss = self.grapher.calculate_loss(self, dataset.x, dataset.y)
            val_loss = self.grapher.calculate_loss(self, dataset.x_valid, dataset.y_valid)
            losses.append(val_loss)
            # print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
            self.grapher.update_metrics(epoch + 1, train_loss, val_loss)
            # Print predictions for debugging
            predictions = self.feed_forward(dataset.x_valid)
            predicted_classes = np.argmax(predictions, axis=1)
            # print(f"Predictions: {predicted_classes}")
            
            if len(losses) == early_stopping_patience and all(l > val_loss for l in list(losses)[:-1]):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

    def save(self, model_name="mymodel"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        path = os.path.join(models_dir, model_name)
        print(f"Saving model to: {path}")
        delete_dir_and_contents(path)
        os.mkdir(path)
        np.savetxt(os.path.join(path, "activations.csv"), self.activations, delimiter=",", fmt="%s")
        np.savetxt(os.path.join(path, "sizes.csv"), self.sizes, delimiter=",", fmt="%d")
        with open(os.path.join(path, "optimizer.pkl"), "wb") as f:
            pickle.dump(self.optimizer, f)
        for i, layer in enumerate(self.layers):
            np.savetxt(os.path.join(path, f"weights_{i}.csv"), layer.w, delimiter=",")
            np.savetxt(os.path.join(path, f"biases_{i}.csv"), layer.b, delimiter=",")

    @staticmethod
    def load(model_name="mymodel"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        path = os.path.join(models_dir, model_name)
        activations = np.genfromtxt(os.path.join(path, "activations.csv"), delimiter=",", dtype=str)
        sizes = np.genfromtxt(os.path.join(path, "sizes.csv"), delimiter=",", dtype=int)
        with open(os.path.join(path, "optimizer.pkl"), "rb") as f:
            optimizer = pickle.load(f)
        model = Model(sizes, activations, optimizer=optimizer)
        for i in range(len(sizes) - 1):
            model.layers[i].w = np.genfromtxt(os.path.join(path, f"weights_{i}.csv"), delimiter=",")
            model.layers[i].b = np.genfromtxt(os.path.join(path, f"biases_{i}.csv"), delimiter=",")
        return model