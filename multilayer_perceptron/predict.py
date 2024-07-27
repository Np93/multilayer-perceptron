import numpy as np
import pandas as pd
from multilayer_perceptron.mlp import MLP
from multilayer_perceptron.layers import DenseLayer

def predict_model(features_file, model_weights_files, model_biases_files):
    X = pd.read_csv(features_file).values

    layers = []
    for i, (weight_file, bias_file) in enumerate(zip(model_weights_files, model_biases_files)):
        weights = np.load(weight_file, allow_pickle=True)
        biases = np.load(bias_file, allow_pickle=True)
        activation = 'softmax' if i == len(model_weights_files) - 1 else 'relu'
        layer = DenseLayer(weights.shape[0], weights.shape[1], activation=activation)
        layer.weights = weights
        layer.bias = biases
        layers.append(layer)

    mlp = MLP(layers)
    predictions = mlp.forward(X)
    return np.argmax(predictions, axis=1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Make predictions with the trained MLP model.')
    parser.add_argument('--features_file', type=str, required=True, help='Path to the features CSV file.')
    parser.add_argument('--model_weights_files', nargs='+', required=True, help='Paths to the model weights files.')
    parser.add_argument('--model_biases_files', nargs='+', required=True, help='Paths to the model biases files.')
    args = parser.parse_args()
    predictions = predict_model(args.features_file, args.model_weights_files, args.model_biases_files)
    print(predictions)

    # poetry run python multilayer_perceptron/predict.py --features_file data/valid_features.csv --model_weights_files data/model_layer_0_weights.npy data/model_layer_1_weights.npy data/model_layer_2_weights.npy --model_biases_files data/model_layer_0_biases.npy data/model_layer_1_biases.npy data/model_layer_2_biases.npy