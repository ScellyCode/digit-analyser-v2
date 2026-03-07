from helper_functions import relu
from helper_functions import softmax
from helper_functions import get_models_dir

import numpy as np
import os
import re


class NeuralNetwork:
    """
    A simple multi-layer neural network for classifying handwritten digits.
    Supports arbitrary number and size of hidden layers.
    """
    MODELS_DIR = get_models_dir()

    def __init__(self, load_from_file: str):
        """
        Initialize the network from a saved .npz file.

        Args:
            load_from_file (str): Path to the .npz file where the model data is stored.
        """
        loaded = self.load(load_from_file)
        self.file_name = load_from_file
        self.weights = loaded["weights"]
        self.biases = loaded["biases"]
        self._cache = {}

    def forward_pass(self, forward_x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            forward_x (np.ndarray): Input data (batch).

        Returns:
            np.ndarray: Network outputs (probabilities).
        """
        a = forward_x
        activations = [forward_x]
        pre_activations = []

        for forwardIdx in range(len(self.weights)):
            W = self.weights[forwardIdx]
            b = self.biases[forwardIdx]

            z = np.dot(a, W) + b
            pre_activations.append(z)

            if forwardIdx == len(self.weights) - 1:
                a = softmax(z)
            else:
                a = relu(z)

            activations.append(a)

        self._cache = {
            "activations": activations,
            "pre_activations": pre_activations,
        }

        return a

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        """
        Return the predicted classes for the input data.

        Args:
            x_predict (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.forward_pass(x_predict)
        return np.argmax(probs, axis=1)

    def load(self, filepath: str, directory: str = MODELS_DIR) -> dict:
        """
        Load network weights and biases from a .npz file.

        Args:
            filepath (str): Filename.
            directory (str): Source directory.
            
        Raises:
            ValueError: If the file contains an error (malformed).
        Returns:
            dict: Dictionary with keys 'weights', 'biases', 'dimensions'
        """
        os.makedirs(directory, exist_ok=True)

        if not filepath.endswith(".npz"):
            filepath += ".npz"

        filepath = os.path.join(directory, filepath)
        with np.load(filepath) as data:
            if "dimensions" not in data:
                raise ValueError(f"File {filepath} is missing 'dimensions' metadata")

            dimensions = data["dimensions"].tolist()

            W_keys = sorted([k for k in data.files if re.fullmatch(r"W\d+", k)], key=lambda x: int(x[1:]))
            b_keys = sorted([k for k in data.files if re.fullmatch(r"b\d+", k)], key=lambda x: int(x[1:]))
            
            if len(W_keys) != len(b_keys):
                raise ValueError("Saved file is malformed: unequal number of W and b arrays")

            weights = []
            biases = []
            for wk, bk in zip(W_keys, b_keys):
                weights.append(data[wk])
                biases.append(data[bk])

        print(f"Model loaded from {filepath}")
        return {"weights": weights, "biases": biases, "dimensions": dimensions}

    def get_model_info(self) -> dict:
        filepath = os.path.join(self.MODELS_DIR, self.file_name)
        info = {}
        with np.load(filepath) as data:
            info["parameters"] = int(sum(data[k].size for k in data if k.startswith("W") or k.startswith("b")))
            info["layers"] = data["dimensions"].tolist()
            info["epochs"] = int(data["epochs"][0]) if "epochs" in data else None
            info["learning_rate"] = float(data["learning_rate"][0]) if "learning_rate" in data else None
            info["batch_size"] = int(data["batch_size"][0]) if "batch_size" in data else None
            info["train_acc"] = float(data["train_acc"][0]) if "train_acc" in data else None
            info["test_acc"] = float(data["test_acc"][0]) if "test_acc" in data else None
        return info

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the network.

        Returns:
            int: Number of parameters.
        """
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size

        return total
