@echo off
echo =========================================
echo Setting up Local Anime Voice Cloner
echo =========================================

REM Create virtual environment only if it does not exist
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other dependencies...
python -m pip install -r requirements.txt

echo.
echo =========================================
echo Setup Complete!
echo =========================================

pause