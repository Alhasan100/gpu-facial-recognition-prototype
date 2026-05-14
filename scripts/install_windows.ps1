<#
.SYNOPSIS
One-command Windows installer for the GPU facial recognition prototype.

.DESCRIPTION
Creates a project virtual environment, installs Python dependencies, builds
CUDA-enabled dlib, verifies the GPU runtime, and prints the next usage commands.
#>

[CmdletBinding()]
param(
    [string]$PythonLauncher = "py -3.12",
    [string]$VenvPath = ".venv",
    [string]$CudaArchitectures = "native",
    [switch]$SkipDlibCudaBuild
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$CommandName)

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$CommandName' was not found in PATH."
    }
}

function Add-PathPrefix {
    param([string[]]$Paths)

    $existing = $Paths | Where-Object { $_ -and (Test-Path $_) }
    if ($existing.Count -gt 0) {
        $env:PATH = ($existing -join [IO.Path]::PathSeparator) + [IO.Path]::PathSeparator + $env:PATH
    }
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

if ($env:OS -ne "Windows_NT") {
    throw "This installer is for Windows only."
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

Add-PathPrefix @(
    "C:\Program Files\CMake\bin",
    "C:\Program Files (x86)\CMake\bin"
)

if (-not $env:CUDA_PATH) {
    $cudaRoots = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object {
            if ($_.Name -match 'v(\d+(?:\.\d+)*)') {
                [version]$Matches[1]
            } else {
                [version]"0.0"
            }
        } -Descending

    if ($cudaRoots) {
        $env:CUDA_PATH = $cudaRoots[0].FullName
    }
}

if ($env:CUDA_PATH) {
    Add-PathPrefix @((Join-Path $env:CUDA_PATH "bin"))
}

Write-Step "Checking system prerequisites"
Require-Command cmake
Require-Command nvcc

$vcBuildTools = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vcBuildTools)) {
    throw "Visual Studio Build Tools were not found. Install the C++ workload and rerun this installer."
}

Write-Step "Creating virtual environment"
if ([IO.Path]::IsPathRooted($VenvPath)) {
    $venvFullPath = $VenvPath
} else {
    $venvFullPath = Join-Path $projectRoot $VenvPath
}
if (-not (Test-Path $venvFullPath)) {
    $launcherParts = $PythonLauncher -split '\s+'
    $launcher = $launcherParts[0]
    $launcherArgs = @()
    if ($launcherParts.Count -gt 1) {
        $launcherArgs = $launcherParts[1..($launcherParts.Count - 1)]
    }
    Invoke-Checked -Executable $launcher -Arguments ($launcherArgs + @("-m", "venv", $venvFullPath))
} else {
    Write-Host "Virtual environment already exists: $venvFullPath"
}

$python = Join-Path $venvFullPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable was not found in the virtual environment."
}

Write-Step "Installing Python dependencies"
Invoke-Checked -Executable $python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel")
Invoke-Checked -Executable $python -Arguments @("-m", "pip", "install", "-r", "requirements.txt")

if (-not $SkipDlibCudaBuild) {
    Write-Step "Building CUDA-enabled dlib"
    $buildScript = Join-Path $projectRoot "scripts\build_dlib_cuda_windows.ps1"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $buildScript -Python $python -CudaArchitectures $CudaArchitectures
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA dlib build failed."
    }
} else {
    Write-Host "Skipping CUDA dlib build because -SkipDlibCudaBuild was specified."
}

Write-Step "Verifying installation"
$verifyScript = Join-Path $env:TEMP "verify_gpu_face_install.py"
$sitePackages = & $python -c "import site; print([p for p in site.getsitepackages() if p.lower().endswith('site-packages')][0])"
$verifyCode = @"
import os
from pathlib import Path

site_packages = Path(r"$($sitePackages)")
paths = [
    site_packages / "nvidia" / "cudnn" / "bin",
    site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
]
cuda_path = os.environ.get("CUDA_PATH")
if cuda_path:
    paths.append(Path(cuda_path) / "bin")

for path in paths:
    if path.exists():
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(path))

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

import cv2
import dlib
import face_recognition

print("OpenCV:", cv2.__version__)
print("dlib:", dlib.__version__)
print("DLIB_USE_CUDA:", dlib.DLIB_USE_CUDA)
print("CUDA devices:", dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0)

if not dlib.DLIB_USE_CUDA or dlib.cuda.get_num_devices() < 1:
    raise SystemExit(1)
"@

$verifyCode | Set-Content -Encoding UTF8 -Path $verifyScript
Invoke-Checked -Executable $python -Arguments @($verifyScript)

Write-Host ""
Write-Host "Setup is complete. GPU support was verified successfully." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment:"
Write-Host "   .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "2. Optional: verify CUDA/dlib again:"
Write-Host "   python -c `"import dlib; print('DLIB_USE_CUDA:', dlib.DLIB_USE_CUDA); print('CUDA devices:', dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0)`""
Write-Host ""
Write-Host "3. Collect training images. Replace person_name with the label you want to train:"
Write-Host "   python .\01_collect_data.py --name person_name"
Write-Host ""
Write-Host "4. Train face encodings:"
Write-Host "   python .\02_train_gpu.py"
Write-Host ""
Write-Host "5. Start live recognition:"
Write-Host "   python .\03_recognize.py --debug"
Write-Host ""
Write-Host "Controls"
Write-Host "  Data collection: press C to capture, Q or Esc to quit."
Write-Host "  Recognition: press Q or Esc to quit."
