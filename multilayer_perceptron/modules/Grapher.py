import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .Loss import CrossEntropyLoss

class Grapher:
    def __init__(self):
        self.metrics = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    def calculate_loss(self, model, x, y):
        y_hat = model.feed_forward(x)
        # print(f"calculate_loss: y_hat shape: {y_hat.shape}, y shape: {y.shape}")
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.loss(y_hat, y)
        return loss

    def update_metrics(self, epoch, train_loss, val_loss):
        # print(f"Updating metrics: Epoch {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")
        new_metrics = pd.DataFrame({"epoch": [epoch], "train_loss": [train_loss], "val_loss": [val_loss]})
        self.metrics = pd.concat([self.metrics, new_metrics], ignore_index=True)

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["epoch"], self.metrics["train_loss"], label='Train Loss')
        plt.plot(self.metrics["epoch"], self.metrics["val_loss"], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()