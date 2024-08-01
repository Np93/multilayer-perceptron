import os
import numpy as np
import pandas as pd
from modules import Model
from utils import load_dataset

# Utiliser le chemin relatif pour accéder au répertoire 'data' à la racine du projet
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..'))
data_dir = os.path.join(project_root, "data")
test_file_path = os.path.join(data_dir, "test.csv")
model_name = "mymodel"

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

def main():
    # Vérifiez les fichiers disponibles dans le répertoire data
    print("Files in data directory:", os.listdir(data_dir))
    
    model = Model.load(model_name)
    X_test, y_test = load_test_data(test_file_path)
    
    predictions = model.feed_forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    correct_predictions = np.sum(predicted_classes == y_test)
    total_predictions = len(y_test)
    
    for i in range(total_predictions):
        expected = 'M' if y_test[i] == 1 else 'B'
        predicted = 'M' if predicted_classes[i] == 1 else 'B'
        neuron_outputs = predictions[i]
        print(f"Sample {i + 1}: Expected: {expected}, Predicted: {predicted}")
        print(f"Neuron outputs: {neuron_outputs[0]:.4f} (B), {neuron_outputs[1]:.4f} (M)")
    
    print(f"Correct predictions: {correct_predictions} / {total_predictions}")

    # Calcul et affichage des métriques
    tp, fp, tn, fn = evaluate_binary_classifier(model, X_test, y_test)
    print_metrics(tp, fp, tn, fn)

    # Afficher la matrice de confusion
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

if __name__ == "__main__":
    main() 




# import os
# import numpy as np
# import pandas as pd
# from modules import Model
# from utils import load_dataset

# def load_test_data(file_path):
#     X, y = load_dataset(file_path)
#     return X, y

# def main():
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(base_dir, "models", "mymodel")
#     test_file_path = os.path.join(base_dir, "data", "test.csv")

#     X_test, y_test = load_test_data(test_file_path)
#     model = Model.load(model_name="mymodel")

#     predictions = model.feed_forward(X_test)
#     predictions = np.argmax(predictions, axis=1)

#     correct_predictions = np.sum(predictions == y_test)
#     total_predictions = y_test.shape[0]
#     accuracy = correct_predictions / total_predictions

#     for i in range(len(predictions)):
#         print(f"Sample {i + 1}: Expected: {'M' if y_test[i] == 1 else 'B'}, Predicted: {'M' if predictions[i] == 1 else 'B'}")

#     print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

# if __name__ == "__main__":
#     main()