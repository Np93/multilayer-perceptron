import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset(file_path):
    data = pd.read_csv(file_path, header=None)
    y = data.iloc[:, 0].apply(lambda x: 1 if x == 'M' else 0).values  # Correctement extraire les étiquettes
    X = data.iloc[:, 1:].values  # Utiliser les colonnes restantes pour les caractéristiques
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # print("y (étiquettes):", y)
    # print("X_scaled (caractéristiques normalisées):", X_scaled)
    return X_scaled, y

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

def get_activation_function(name):
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



# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# def load_dataset(file_path):
#     data = pd.read_csv(file_path, header=None)
#     y = data.iloc[:, 0].apply(lambda x: 1 if x == 'M' else 0).values  # Correctement extraire les étiquettes
#     X = data.iloc[:, 1:].values  # Utiliser les colonnes restantes pour les caractéristiques
    
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     return X_scaled, y

# def add_bias_units(X):
#     return np.hstack([np.ones((X.shape[0], 1)), X])

# def xavier_init(in_size, out_size):
#     return np.random.randn(in_size, out_size) * np.sqrt(2. / (in_size + out_size))

# def delete_dir_and_contents(path):
#     if os.path.exists(path):
#         for file in os.listdir(path):
#             file_path = os.path.join(path, file)
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 delete_dir_and_contents(file_path)
#         os.rmdir(path)


# def load_dataset(file_path):
#     data = pd.read_csv(file_path, header=None)
#     y = data.iloc[:, 0].apply(lambda x: 1 if x == 'M' else 0).values  # Correctement extraire les étiquettes
#     X = data.iloc[:, 1:].values  # Utiliser les colonnes restantes pour les caractéristiques
    
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     return X_scaled, y

# def create_dataset_from_path(file_path, usecols, y_col, y_categorical):
#     data = pd.read_csv(file_path, usecols=usecols)
#     y = data.iloc[:, y_col].apply(lambda x: 1 if x == 'M' else 0).values if y_categorical else data.iloc[:, y_col].values
#     X = data.drop(columns=[data.columns[y_col]]).values
    
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     return Dataset(X_scaled, y)

# def add_bias_units(X):
#     return np.hstack([np.ones((X.shape[0], 1)), X])

# def xavier_init(in_size, out_size):
#     return np.random.randn(in_size, out_size) * np.sqrt(2. / (in_size + out_size))

# def delete_dir_and_contents(path):
#     if os.path.exists(path):
#         for file in os.listdir(path):
#             file_path = os.path.join(path, file)
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 delete_dir_and_contents(file_path)
#         os.rmdir(path)

# def calculate_and_display_metrics(model, X, y):
#     predictions = model.feed_forward(X)
#     accuracy = np.mean(np.argmax(predictions, axis=1) == y)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     return accuracy

# def evaluate_binary_classifier(y_true, y_pred):
#     tp = np.sum((y_true == 1) & (y_pred == 1))
#     tn = np.sum((y_true == 0) & (y_pred == 0))
#     fp = np.sum((y_true == 0) & (y_pred == 1))
#     fn = np.sum((y_true == 1) & (y_pred == 0))
    
#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     precision = tp / (tp + fp) if tp + fp != 0 else 0
#     recall = tp / (tp + fn) if tp + fn != 0 else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
#     return accuracy, precision, recall, f1_score

# def is_overfitting(losses):
#     if len(losses) < 2:
#         return False
#     return losses[-1] > min(losses)