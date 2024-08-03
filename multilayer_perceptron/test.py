import os
import numpy as np
import pandas as pd
import argparse
from modules import Model
from utils import load_dataset
import sys

# Utiliser le chemin relatif pour accéder au répertoire 'data' à la racine du projet
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..'))
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(base_dir, "modules", "models")
default_test_file_path = os.path.join(data_dir, "test.csv")
default_model_name = "mymodel"

def load_test_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    X, y = load_dataset(file_path)
    return X, y

def evaluate_binary_classifier(model, x, y):
    yhat = model.feed_forward(x)
    yhatmax = (yhat == yhat.max(axis=1, keepdims=True)).astype(int)
    
    # Convert y to one-hot encoding
    y_one_hot = np.zeros_like(yhat)
    y_one_hot[np.arange(y.shape[0]), y] = 1
    
    e = (2 * y_one_hot) + yhatmax
    tp = (e[:, 1] == 3).astype(int).sum()
    tn = (e[:, 1] == 0).astype(int).sum()
    fn = (e[:, 1] == 2).astype(int).sum()
    fp = (e[:, 1] == 1).astype(int).sum()
    return tp, fp, tn, fn

def calculate_metrics(tp, fp, tn, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1 = 2.0 * (sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) != 0 else 0
    return (sensitivity, specificity, precision, f1)

def print_metrics(tp, fp, tn, fn):
    sensitivity, specificity, precision, f1 = calculate_metrics(tp, fp, tn, fn)
    print(f"{sensitivity = :.3f}, {specificity = :.3f}, {precision = :.3f}, {f1 = :.3f}\n")

def calculate_binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def main():
    # Configurer l'analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Evaluate a neural network model.")
    parser.add_argument("model_name", nargs='?', default=default_model_name, help="The name of the model to load")
    parser.add_argument("test_file_path", nargs='?', default=default_test_file_path, help="The path to the test dataset")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    test_file_path = args.test_file_path

    model_path = os.path.join(models_dir, model_name)

    # Vérifiez si les fichiers existent
    if not os.path.exists(test_file_path):
        print(f"Error: file for Prediction not found: {test_file_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    # Vérifiez les fichiers disponibles dans le répertoire data
    print("Files in data directory:", os.listdir(data_dir))
    
    model = Model.load(model_name)
    X_test, y_test = load_test_data(test_file_path)
    
    predictions = model.feed_forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    correct_predictions = np.sum(predicted_classes == y_test)
    total_predictions = len(y_test)
    
    for i in range(total_predictions):
        expected = '1' if y_test[i] == 1 else '0'
        predicted = '1' if predicted_classes[i] == 1 else '0'
        neuron_outputs = predictions[i]
        print(f"Sample {i + 1}: Expected: {expected}, Predicted: {predicted}")
        print(f"Neuron outputs: {neuron_outputs[0]:.4f} (0), {neuron_outputs[1]:.4f} (1)")
    
    print(f"Correct predictions: {correct_predictions} / {total_predictions}")

    # Calcul et affichage des métriques
    tp, fp, tn, fn = evaluate_binary_classifier(model, X_test, y_test)
    print_metrics(tp, fp, tn, fn)

    # Afficher la matrice de confusion
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

    # Calculer et afficher la perte d'entropie croisée binaire
    y_test_one_hot = np.zeros_like(predictions)
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1
    bce_loss = calculate_binary_cross_entropy(y_test_one_hot, predictions)
    print(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")

if __name__ == "__main__":
    main()