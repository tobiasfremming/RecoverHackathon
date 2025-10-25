# Quick Start Script for Deep Sets Training
# Windows PowerShell

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Deep Sets Autoencoder - Quick Start" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+." -ForegroundColor Red
    exit 1
}

# Check PyTorch installation
Write-Host "`nChecking PyTorch installation..." -ForegroundColor Yellow
$torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $torchCheck" -ForegroundColor Green
} else {
    Write-Host "✗ PyTorch not found. Installing..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Check other dependencies
Write-Host "`nChecking other dependencies..." -ForegroundColor Yellow
$packages = @("numpy", "pandas", "polars", "matplotlib", "seaborn", "tqdm")
foreach ($package in $packages) {
    $check = python -c "import $package; print('OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $package installed" -ForegroundColor Green
    } else {
        Write-Host "✗ $package not found - installing..." -ForegroundColor Yellow
        pip install $package
    }
}

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
$dirs = @("checkpoints", "logs", "submissions")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ Exists: $dir" -ForegroundColor Green
    }
}

# Check data directory
Write-Host "`nChecking data..." -ForegroundColor Yellow
if (Test-Path "data/hackathon-recover-x-cogito") {
    $csvFiles = Get-ChildItem "data/hackathon-recover-x-cogito" -Filter "*.csv"
    Write-Host "✓ Found $($csvFiles.Count) CSV files in data directory" -ForegroundColor Green
} else {
    Write-Host "✗ Data directory not found." -ForegroundColor Red
    Write-Host "  Please download data from Kaggle and place in data/hackathon-recover-x-cogito/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! Ready to Train" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Offer training options
Write-Host "Choose an option:" -ForegroundColor Yellow
Write-Host "  1. Quick test (10 epochs, fast config)" -ForegroundColor White
Write-Host "  2. Default training (50 epochs)" -ForegroundColor White
Write-Host "  3. Production training (100 epochs, large model)" -ForegroundColor White
Write-Host "  4. Test model architecture only" -ForegroundColor White
Write-Host "  5. Open interactive notebook" -ForegroundColor White
Write-Host "  6. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice (1-6)"

switch ($choice) {
    "1" {
        Write-Host "`nStarting quick test training..." -ForegroundColor Green
        python train.py --config fast --epochs 10 --batch_size 64
    }
    "2" {
        Write-Host "`nStarting default training..." -ForegroundColor Green
        python train.py --config default
    }
    "3" {
        Write-Host "`nStarting production training..." -ForegroundColor Green
        python train.py --config production
    }
    "4" {
        Write-Host "`nTesting model architecture..." -ForegroundColor Green
        python -c "from models.deep_sets_autoencoder import DeepSetsAutoencoder; import torch; model = DeepSetsAutoencoder(); print('✓ Model created successfully!'); print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
    }
    "5" {
        Write-Host "`nOpening notebook..." -ForegroundColor Green
        jupyter notebook experiments/model_training.ipynb
    }
    "6" {
        Write-Host "`nExiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "`nInvalid choice. Exiting..." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nDone!" -ForegroundColor Green
