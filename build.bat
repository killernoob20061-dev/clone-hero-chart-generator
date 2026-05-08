@echo off
echo ========================================
echo  ChartGen Suite - Build EXE (Nuitka)
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (echo ERROR: Python not found. & pause & exit /b 1)

echo [1/3] Installing build dependencies...
pip install nuitka ordered-set zstandard --quiet

echo [2/3] Cleaning old build...
if exist src\dist     rmdir /s /q src\dist
if exist src\app.dist rmdir /s /q src\app.dist
if exist src\app.build rmdir /s /q src\app.build

echo [3/3] Building ChartGen with Nuitka (this takes 5-10 minutes)...
echo       Nuitka may download MinGW64 on first run - this is normal.
echo.
cd src

python -m nuitka --standalone ^
    --windows-console-mode=disable ^
    --enable-plugin=tk-inter ^
    --include-package=customtkinter ^
    --include-package-data=customtkinter ^
    --include-package=PIL ^
    --include-package-data=PIL ^
    --include-data-files=chartgen.py=chartgen.py ^
    --include-data-files=chartmodel.py=chartmodel.py ^
    --include-data-files=scraper.py=scraper.py ^
    --include-data-files=preprocess_dataset.py=preprocess_dataset.py ^
    --include-data-files=train_chartnet.py=train_chartnet.py ^
    --assume-yes-for-downloads ^
    --output-dir=dist ^
    app.py

if errorlevel 1 (echo. & echo ERROR: Build failed. & cd .. & pause & exit /b 1)

echo.
echo Moving to release folder...
if not exist "..\release" mkdir "..\release"
if exist "..\release\ChartGen" rmdir /s /q "..\release\ChartGen"

ren dist\app.dist\app.exe ChartGen.exe
xcopy /e /i /y dist\app.dist ..\release\ChartGen >nul
rmdir /s /q dist
cd ..

echo.
echo ========================================
echo  SUCCESS! Run release\ChartGen\ChartGen.exe
echo ========================================
pause
