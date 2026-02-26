@echo off
echo Building App

echo 1. Building Frontend (React)
echo ...
cd frontend
call npm run build
cd ..

echo 2. Package Python App (PyInstaller)
echo ...
pyinstaller --noconsole --onefile --name "DigitAnalyser" --icon "assets/lament.ico" --add-data "frontend/dist;frontend/dist" --add-data "models;models" backend/app.py

echo Build Complete!
pause