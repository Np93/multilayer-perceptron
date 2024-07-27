import os
import numpy as np
import pandas as pd
from modules import Model, Dataset, NAGOptimizer, CrossEntropyLoss, Grapher
from utils import load_dataset, is_overfitting
from visualization import calculate_metrics, plot_metrics  # Importer les nouvelles fonctions

# Configurations
folds = 5
reset_between_folds = False
epochs = 300
batchsize = 64
loss = CrossEntropyLoss()
optimizer = NAGOptimizer(learning_rate=0.03, momentum=0.9)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(base_dir, "../data/train.csv")
valid_file_path = os.path.join(base_dir, "../data/valid.csv")
model_name = "mymodel"

def main():
    X_train, y_train = load_dataset(train_file_path)
    X_valid, y_valid = load_dataset(valid_file_path)
    
    train_dataset = Dataset(X_train, y_train, X_valid, y_valid)
    valid_dataset = Dataset(X_valid, y_valid)
    
    model = Model(sizes=[X_train.shape[1], 15, 8, 2], activations=['sigmoid', 'sigmoid', 'softmax'], optimizer=optimizer)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train(train_dataset, batch_size=batchsize, epochs=1)
        
        train_loss, train_accuracy, train_predictions = calculate_metrics(model, train_dataset, loss)
        val_loss, val_accuracy, val_predictions = calculate_metrics(model, valid_dataset, loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} - Train accuracy: {train_accuracy:.4f} - Val accuracy: {val_accuracy:.4f}")
        
        # Display predictions for a sample
        sample_size = 10
        print(f"Sample predictions: {train_predictions[:sample_size]} - Actual: {train_dataset.y[:sample_size]}")
    
    # Plotting the metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    model.save(model_name)

if __name__ == "__main__":
    main()























    # def save_model(layers, file_path='data/saved_model.npy'):
    # model_data = [(layer.weights, layer.bias) for layer in layers]
    # np.save(file_path, model_data)