@echo off
echo ========================================
echo  ChartGen - Build EXE
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing build dependencies...
pip install pyinstaller==6.3.0 customtkinter Pillow==10.4.0 --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

:: Clean old build
echo [2/3] Cleaning old build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist ChartGen.spec del ChartGen.spec

:: Build exe
echo [3/3] Building ChartGen.exe...
python -m PyInstaller --onefile --windowed --name "ChartGen" ^
    --collect-all customtkinter ^
    --collect-all PIL ^
    --icon "chartgen.ico" ^
    app.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed. Check output above.
    pause
    exit /b 1
)

:: Move to release folder
if not exist "..\release" mkdir "..\release"
move /y dist\ChartGen.exe ..\release\ChartGen.exe

:: Cleanup
rmdir /s /q build
rmdir /s /q dist
del ChartGen.spec

echo.
echo ========================================
echo  SUCCESS! ChartGen.exe is in release/
echo ========================================
pause
