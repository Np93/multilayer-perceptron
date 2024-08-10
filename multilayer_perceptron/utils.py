import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_dataset(file_path):
    data = verify_file(file_path)

    y = data.iloc[:, 0].values  # Extraire les étiquettes sans transformation
    X = data.iloc[:, 1:].values  # Utiliser les colonnes restantes pour les caractéristiques
    
    # Utiliser LabelEncoder pour encoder les étiquettes de manière ordonnée
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    unique_labels = np.unique(y)
    num_unique_labels = len(unique_labels)
    
    print(f"Nombre de symboles différents dans y: {num_unique_labels}")
    return X_scaled, y, num_unique_labels

def verify_file(file_path):
    # Vérifier que le fichier est un .csv
    if not file_path.endswith('.csv'):
        raise ValueError("Le fichier doit être un fichier .csv.")

    # Vérifier que le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    # Lire le fichier sans en-tête pour vérification
    data = pd.read_csv(file_path, header=None)
    
    # Vérification des valeurs manquantes
    if data.isnull().values.any():
        raise ValueError("Le fichier contient des valeurs manquantes.")
    
    return data

def add_bias_units(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def xavier_init(in_size, out_size):
    return np.random.randn(in_size, out_size) * np.sqrt(2. / (in_size + out_size))

def delete_dir_and_contents(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                delete_dir_and_contents(file_path)
        os.rmdir(path)

def get_activation_function(name: str):
    if name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif name == 'relu':
        return relu, relu_derivative
    elif name == 'softmax':
        return softmax, softmax_derivative
    else:
        raise ValueError(f"Unknown activation function: {name}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z, a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z, a):
    return np.where(z > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(z, a):
    return a * (1 - a)

def is_overfitting(losses, patience=5):
    """
    Check if the model is overfitting based on validation losses.
    
    Args:
    losses (deque): A deque containing the validation losses.
    patience (int): Number of epochs to consider for early stopping.
    
    Returns:
    bool: True if the model is overfitting, False otherwise.
    """
    if len(losses) < patience:
        return False
    recent_losses = list(losses)[-patience:]
    return all(recent_losses[i] > recent_losses[i + 1] for i in range(patience - 1))