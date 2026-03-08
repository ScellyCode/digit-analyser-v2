import sys
import os
import tempfile

import numpy as np
import webview
from PIL import Image

from neural_network import NeuralNetwork
from helper_functions import resource_path, get_highest_model_filename, get_models_dir


class Api:
    def __init__(self):
        self.current_model = get_highest_model_filename()
        if self.current_model is None:
            models_dir = get_models_dir()
            raise RuntimeError(f"No model files found in models directory: {models_dir}")
        self.nn = NeuralNetwork(self.current_model)

    def __save_debug_image(self, image_data_vector: list[float]) -> None:
        """
        Saves a debug image to disk to visualize what the NN is receiving.
        
        Args:
            image_data_vector (list[float]): A list of floats representing the image data.
        """
        arr = np.array(image_data_vector).reshape(28, 28) * 255
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        debug_path = os.path.join(tempfile.gettempdir(), "debug_image.png")
        img.save(debug_path)
        print(f"Debug image saved to {debug_path}")

    def predict_digit(self, image_data_vector: list[float]) -> list[float]:
        """
        Takes an image vector and gives back a prediction for each digit inform of a list.
        
        Args:
            image_data_vector (list[float]): A list of floats representing the image data.

        Returns:
            list[float]: A list of floats representing the probability of every digit.
        """
        if '--dev' in sys.argv:
            self.__save_debug_image(image_data_vector)
        x = np.array(image_data_vector).reshape(1, -1)
        return self.nn.forward_pass(x).tolist()

    def get_current_model(self) -> str:
        """
        Returns the current model name.
        
        Returns:
            str: The current model name.

        """
        return self.current_model

    def get_models(self) -> list[str]:
        """
        Returns a list of all available models.
        
        Returns:
            list[str]: A list of all available models.

        """
        models_dir = get_models_dir()
        files = [f for f in os.listdir(models_dir) if f.endswith('.npz')]
        files.sort()
        return files

    def set_model(self, model_filename: str) -> None:
        """
        Loads a new model from name.
        
        Args:
            model_filename: name of the model file.
        """
        self.nn = NeuralNetwork(model_filename)
        self.current_model = model_filename

    def get_model_info(self) -> dict:
        """
        Returns the current model information.
        
        Returns:
            dict: A dictionary containing information about the current model.
        """
        return self.nn.get_model_info()


if __name__ == "__main__":
    api = Api()
    if '--dev' in sys.argv:
        print("Running in development mode (Hot Reloading active)...")
        webview.create_window("Digit Analyser", 'http://localhost:5173', js_api=api)
        webview.start(debug=True)
    else:
        print("Running in production mode...")
        html_path = resource_path("frontend/dist/index.html")
        webview.create_window("Digit Analyser", html_path, js_api=api)
        webview.start()
