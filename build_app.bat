@echo off
echo Building App

echo 1. Building Frontend (React)
echo ...
cd frontend
call npm run build
cd ..

echo 2. Generating Python Documentation (Sphinx)
echo ...
cd docs
call make.bat html
cd ..

echo 3. Package Python App (PyInstaller)
echo ...
pyinstaller --noconsole --onefile --name "DigitAnalyser" --icon "assets/lament.ico" --add-data "frontend/dist;frontend/dist" --add-data "models;models" --add-data "docs/build/html;docs/html" backend/app.py

echo Build Complete!
pause