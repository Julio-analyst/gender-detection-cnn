# =============================================================================
# Quick Start - Gender Voice Detection Streamlit App
# =============================================================================
# Script ini akan setup dan menjalankan aplikasi secara otomatis
# =============================================================================

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "  GENDER VOICE DETECTION - STREAMLIT APP SETUP  " -ForegroundColor Yellow
Write-Host ("="*80) -ForegroundColor Cyan

# Check Python
Write-Host "`n[1/4] Checking Python..." -ForegroundColor Green
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n[2/4] Installing dependencies..." -ForegroundColor Green
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
pip install -r requirements_app.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    exit 1
}
Write-Host "Dependencies installed successfully!" -ForegroundColor Green

# Check if model exists
Write-Host "`n[3/4] Checking model..." -ForegroundColor Green
if (Test-Path "models\lstm_production.h5") {
    Write-Host "Model found: models\lstm_production.h5" -ForegroundColor Green
} else {
    Write-Host "Model not found. Training model..." -ForegroundColor Yellow
    
    # Check if data exists
    if (-not (Test-Path "data\processed\features_latest.npy")) {
        Write-Host "ERROR: Data not found in data\processed\" -ForegroundColor Red
        Write-Host "Please ensure features_latest.npy and labels_latest.npy exist" -ForegroundColor Yellow
        exit 1
    }
    
    # Train model
    Write-Host "Starting training (this may take several minutes)..." -ForegroundColor Yellow
    python train_model.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Model training failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Model trained successfully!" -ForegroundColor Green
}

# Launch Streamlit
Write-Host "`n[4/4] Launching Streamlit app..." -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "`nApp will open in your browser at: http://localhost:8501" -ForegroundColor Yellow
Write-Host "`nPress Ctrl+C to stop the app" -ForegroundColor Yellow
Write-Host ("="*80) -ForegroundColor Cyan

Start-Sleep -Seconds 2

streamlit run app_streamlit.py
