from helper_functions import resource_path

import sys
import webview

if __name__ == "__main__":
    if '--dev' in sys.argv:
        print("Running in development mode (Hot Reloading active)...")
        webview.create_window("Digit Analyser", 'http://localhost:5173')
    else:
        print("Running in production mode...")
        html_path = resource_path("frontend/dist/index.html")
        webview.create_window("Digit Analyser", html_path)

    webview.start()
