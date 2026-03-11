import os
import sys
import re

import numpy as np
from PIL import Image


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


def preprocess_digit(image_vector: list[float]) -> list[float]:
    """
    Processes the image to work better with the MNIST dataset.

    Args:
        image_vector: The image vector to be processed.

    Returns:
        list[float]: Processed image vector to use for prediction.
    """
    img = np.array(image_vector).reshape(28, 28)

    rows = np.any(img > 0.01, axis=1)
    cols = np.any(img > 0.01, axis=0)

    if not rows.any():
        return image_vector

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img[rmin:rmax + 1, cmin:cmax + 1]

    crop_h, crop_w = cropped.shape
    scale = 20.0 / max(crop_h, crop_w)
    new_h, new_w = max(1, int(crop_h * scale)), max(1, int(crop_w * scale))

    pil_img = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(pil_img).astype(np.float32) / 255.0

    result = np.zeros((28, 28), dtype=np.float32)

    total_mass = resized.sum()
    if total_mass > 0:
        cy = int(np.round(np.sum(np.arange(new_h)[:, None] * resized) / total_mass))
        cx = int(np.round(np.sum(np.arange(new_w)[None, :] * resized) / total_mass))
    else:
        cy, cx = new_h // 2, new_w // 2

    start_y = 14 - cy
    start_x = 14 - cx

    sy = max(0, start_y)
    sx = max(0, start_x)
    ey = min(28, start_y + new_h)
    ex = min(28, start_x + new_w)

    crop_sy = sy - start_y
    crop_sx = sx - start_x
    crop_ey = crop_sy + (ey - sy)
    crop_ex = crop_sx + (ex - sx)

    result[sy:ey, sx:ex] = resized[crop_sy:crop_ey, crop_sx:crop_ex]

    return result.flatten().tolist()
