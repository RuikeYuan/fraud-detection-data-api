# setup_env.ps1 - Create Python venv and install dependencies
# Usage: .\setup_env.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "=== Fraud Detection API - Environment Setup ===" -ForegroundColor Cyan

# Step 1: Find Python
Write-Host ""
Write-Host "[1/3] Checking Python installation..." -ForegroundColor Yellow

$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 9) {
                $python = $candidate
                Write-Host "  Found: $ver  ($candidate)" -ForegroundColor Green
                break
            } else {
                Write-Host "  $ver is too old (need 3.9+), searching..." -ForegroundColor DarkYellow
            }
        }
    } catch { }
}

if (-not $python) {
    Write-Host "  Python not found. Installing Python 3.12 via winget..." -ForegroundColor Yellow
    winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python installation failed. Please install Python 3.9+ manually."
        exit 1
    }
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $python = "python"
    Write-Host "  Python installed successfully." -ForegroundColor Green
}

# Step 2: Create venv
Write-Host ""
Write-Host "[2/3] Creating virtual environment (venv)..." -ForegroundColor Yellow

$venvDir = Join-Path $ProjectDir "venv"
if (Test-Path $venvDir) {
    Write-Host "  venv already exists, skipping creation." -ForegroundColor DarkYellow
} else {
    & $python -m venv venv
    Write-Host "  Virtual environment created at: $venvDir" -ForegroundColor Green
}

# Step 3: Install dependencies
Write-Host ""
Write-Host "[3/3] Installing requirements.txt..." -ForegroundColor Yellow

$pip = Join-Path $venvDir "Scripts\pip.exe"
& $pip install --upgrade pip
& $pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency installation failed. Check errors above."
    exit 1
}

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Cyan
Write-Host "To activate the virtual environment, run:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
