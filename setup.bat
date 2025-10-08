@echo off
REM =================================================
REM Fall Detector YOLOv5 - Auto Setup
REM =================================================

REM 1. SET PROJECT_DIR
SET PROJECT_DIR=%~dp0
SET VENV_DIR=%PROJECT_DIR%\.venv

REM 2. Creating virtual environment
IF NOT EXIST "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM 3. Activating virtual environment
echo Activating virtual environment...
CALL "%VENV_DIR%\Scripts\activate.bat"

REM 4. Installing requirements
echo Installing required packages...
pip install --upgrade pip
pip install -r "%PROJECT_DIR%\requirements.txt"

REM 5. Starting FastAPI backend
echo Starting FastAPI server...
uvicorn main:app --reload

pause
