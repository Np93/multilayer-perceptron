import numpy as np
import os
import pandas as pd
import sys
from modules import Model, Dataset, NAGOptimizer, CrossEntropyLoss, Grapher
from utils import load_dataset, is_overfitting
from visualization import calculate_metrics, plot_metrics  # Importer les nouvelles fonctions

# Configurations
folds = 10
epochs = 700
batchsize = 64
loss = CrossEntropyLoss()
optimizer = NAGOptimizer(learning_rate=0.35, momentum=0.9)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(base_dir, "../data/train.csv")
valid_file_path = os.path.join(base_dir, "../data/valid.csv")
model_name = "mymodel"
best_model_name = "mymodel_best"

def main():
    if not os.path.exists(train_file_path):
        print(f"Error: Training file not found: {train_file_path}")
        sys.exit(1)
    if not os.path.exists(valid_file_path):
        print(f"Error: Validation file not found: {valid_file_path}")
        sys.exit(1)
    
    X_train, y_train = load_dataset(train_file_path)
    X_valid, y_valid = load_dataset(valid_file_path)
    print(f"X_train shape : {X_train.shape}")
    print(f"X_valid shape : {X_valid.shape}")
    
    train_dataset = Dataset(X_train, y_train, X_valid, y_valid)
    valid_dataset = Dataset(X_valid, y_valid)
    
    model = Model(sizes=[X_train.shape[1], 15, 8, 2], activations=['relu', 'relu', 'softmax'], optimizer=optimizer)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model = None
    metrics = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])

    for epoch in range(epochs):
        model.train(train_dataset, batch_size=batchsize, epochs=1, early_stopping_patience=folds)
        
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
        
        metrics = metrics._append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        }, ignore_index=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_model.save(best_model_name, metrics, passe=1)
            # print(f"New best model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")


        if len(val_losses) > folds and val_losses[-1] > min(val_losses[:-folds]):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Plotting the metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    model.save(model_name, metrics, passe=0)
    print(f"Best model saved as {model_name}")

if __name__ == "__main__":
    main()























    # def save_model(layers, file_path='data/saved_model.npy'):
    # model_data = [(layer.weights, layer.bias) for layer in layers]
    # np.save(file_path, model_data)