@echo off
echo ========================================
echo  ChartGen Suite - Build EXE
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (echo ERROR: Python not found. & pause & exit /b 1)

echo [1/3] Installing build dependencies...
pip install pyinstaller==6.3.0 customtkinter "Pillow==10.4.0" --quiet

echo [2/3] Cleaning old build...
if exist src\build   rmdir /s /q src\build
if exist src\dist    rmdir /s /q src\dist
if exist src\ChartGen.spec del src\ChartGen.spec

echo [3/3] Building ChartGen.exe...
cd src
python -m PyInstaller --onefile --windowed --name "ChartGen" ^
    --collect-all customtkinter ^
    --collect-all PIL ^
    --icon "chartgen.ico" ^
    --add-data "chartgen.py;." ^
    --add-data "chartmodel.py;." ^
    --add-data "scraper.py;." ^
    --add-data "preprocess_dataset.py;." ^
    --add-data "train_chartnet.py;." ^
    app.py

if errorlevel 1 (echo. & echo ERROR: Build failed. & cd .. & pause & exit /b 1)

if not exist "..\release" mkdir "..\release"
move /y dist\ChartGen.exe ..\release\ChartGen.exe
rmdir /s /q build & rmdir /s /q dist & del ChartGen.spec
cd ..

echo.
echo ========================================
echo  SUCCESS! ChartGen.exe is in release/
echo ========================================
pause
