@echo off
REM =============================================================================
REM Quick Start - Gender Voice Detection Streamlit App
REM =============================================================================

echo ================================================================================
echo   GENDER VOICE DETECTION - STREAMLIT APP SETUP
echo ================================================================================

REM Check Python
echo.
echo [1/4] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo [2/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements_app.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo Dependencies installed successfully!

REM Check if model exists
echo.
echo [3/4] Checking model...
if exist "models\lstm_production.h5" (
    echo Model found: models\lstm_production.h5
) else (
    echo Model not found. Training model...
    
    REM Check if data exists
    if not exist "data\processed\features_latest.npy" (
        echo ERROR: Data not found in data\processed\
        echo Please ensure features_latest.npy and labels_latest.npy exist
        pause
        exit /b 1
    )
    
    REM Train model
    echo Starting training this may take several minutes...
    python train_model.py
    
    if errorlevel 1 (
        echo ERROR: Model training failed!
        pause
        exit /b 1
    )
    echo Model trained successfully!
)

REM Launch Streamlit
echo.
echo [4/4] Launching Streamlit app...
echo ================================================================================
echo.
echo App will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo ================================================================================
echo.

timeout /t 2 /nobreak >nul

streamlit run app_streamlit.py
