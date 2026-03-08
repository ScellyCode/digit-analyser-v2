import os
import sys
import re

import numpy as np


def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, needed for the packaged project to work correctly.
    
    Args:
        relative_path (string): relative path to file
    
    Returns:
        string: absolute path to file
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def relu(z: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation function to the input array.

    Args:
        z (np.ndarray): Input values.

    Returns:
        np.ndarray: Output values after ReLU.
    """
    return np.maximum(0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply the softmax function to the input array.

    Args:
        z (np.ndarray): Input values (2D array).

    Returns:
        np.ndarray: Probability distribution for each row.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_models_dir() -> str:
    """
    Returns the absolute path of the directory containing the models.
    
    Returns:
        str: Absolute path to the Models directory.
    """
    return os.path.normpath(os.path.abspath(resource_path("./models")))


def get_highest_model_filename(directory: str = None) -> str | None:
    """
    Find the filename of the saved model with the highest version number in the given directory.

    Args:
        directory (str): Directory to search for models.

    Returns:
        str | None: Filename of the model with highest version or None if no model found.
    """
    if directory is None:
        directory = get_models_dir()
    
    pattern = re.compile(r"mnist_model_v(\d+)\.npz")
    highest_version = 0
    found = False

    os.makedirs(directory, exist_ok=True)

    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        if match:
            version = int(match.group(1))
            highest_version = max(highest_version, version)
            found = True

    if not found:
        return None

    return f"mnist_model_v{highest_version}.npz"
