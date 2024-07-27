import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(model, dataset, loss_fn):
    y_hat = model.feed_forward(dataset.x)
    loss = loss_fn.loss(y_hat, dataset.y)  # Utiliser la méthode correcte
    predictions = np.argmax(y_hat, axis=1)
    accuracy = np.mean(predictions == dataset.y)
    return loss, accuracy, predictions

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    
    plt.show()