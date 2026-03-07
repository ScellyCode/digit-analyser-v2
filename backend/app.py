import tempfile

from backend.neural_network import NeuralNetwork
from helper_functions import resource_path
from helper_functions import get_highest_model_filename

import sys
import webview
import os
import numpy as np
from PIL import Image


def save_debug_image(image_data_vector):
    arr = np.array(image_data_vector).reshape(28, 28) * 255
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    temp_dir = tempfile.gettempdir()
    debug_path = os.path.join(temp_dir, "debug_image.png")
    img.save(debug_path)
    print(f"Debug image saved to {debug_path}")


class Api:
    def __init__(self):
        self.current_model = get_highest_model_filename()
        self.nn = NeuralNetwork(self.current_model)

    def predict_digit(self, image_data_vector) -> list:
        if '--dev' in sys.argv:
            save_debug_image(image_data_vector)
        x = np.array(image_data_vector).reshape(1, -1)
        return self.nn.forward_pass(x).tolist()

    def get_current_model(self):
        return self.current_model

    def get_models(self):
        from helper_functions import get_models_dir
        models_dir = get_models_dir()
        files = [f for f in os.listdir(models_dir) if f.endswith('.npz')]
        files.sort()
        return files

    def set_model(self, model_filename):
        self.nn = NeuralNetwork(model_filename)
        self.current_model = model_filename
        return {"status": "ok", "model": model_filename}
    
    def get_model_info(self):
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
