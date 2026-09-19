# Set up Fungus-CV on Windows if needed, then run `fungus doctor`.
# Usage: powershell -ExecutionPolicy Bypass -File .claude\skills\run-on-windows\check.ps1 [-Models]
param([switch]$Models)
# Continue, not Stop: Windows PowerShell 5.1 turns a native command's stderr into a
# terminating error under Stop. Exit codes are checked instead.
$ErrorActionPreference = "Continue"

$root = (git rev-parse --show-toplevel 2>$null)
if (-not $root) { $root = (Get-Location).Path }
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    # Prefer the py launcher; the Microsoft Store "python" stub does not run Python.
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { $exe = "py"; $pre = @("-3") } else { $exe = "python"; $pre = @() }
    & $exe @pre -c "import sys; sys.exit(sys.version_info < (3, 10))"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.10+ is needed: winget install Python.Python.3.12"
        exit 1
    }
    Write-Host "Creating .venv"
    & $exe @pre -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not create .venv"; exit 1 }
}

$missing = $false
& $venvPython -c "import fungus_cv, PySide6, pytest, pytestqt" 2>$null
if ($LASTEXITCODE -ne 0) { $missing = $true }
$extras = "dev,gui"
if ($Models) {
    $extras = "dev,gui,sam"
    & $venvPython -c "import torch, transformers" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missing = $true
        Write-Host ("For an NVIDIA GPU, install the CUDA build of torch first " +
                    "(https://pytorch.org/get-started/locally/); otherwise pip installs the CPU build.")
    }
}
if ($missing) {
    Write-Host "Installing fungus-cv[$extras]"
    & $venvPython -m pip install -q -e ".[$extras]" pytest-qt
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed"; exit 1 }
}

& $venvPython --version
& $venvPython -c "import torch; print('PyTorch', torch.__version__, '- CUDA GPU:', torch.cuda.is_available())" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch not installed (SAM 2 and trained models need: check.ps1 -Models)"
}
& (Join-Path $root ".venv\Scripts\fungus.exe") doctor
exit 0
