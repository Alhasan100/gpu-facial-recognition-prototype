<#
.SYNOPSIS
Builds and installs dlib with CUDA and cuDNN support on Windows.

.DESCRIPTION
This script is intended for a project virtual environment on Windows. It installs
the NVIDIA cuDNN Python wheel, creates the cuDNN import library required by MSVC,
downloads the dlib source distribution, patches dlib's build script so CMake can
receive CUDA/cuDNN paths from environment variables, and installs the resulting
CUDA-enabled dlib package into the active Python environment.
#>

[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$DlibVersion = "20.0.1",
    [string]$CudaArchitectures = "native"
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

function Get-VcVarsPath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe was not found. Install Visual Studio Build Tools with the C++ workload."
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installPath) {
        throw "Visual Studio C++ build tools were not found."
    }

    $vcvarsPath = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvarsPath)) {
        throw "vcvars64.bat was not found at '$vcvarsPath'."
    }

    return $vcvarsPath
}

function Invoke-VcCommand {
    param(
        [string]$VcVarsPath,
        [string]$Command
    )

    $wrapped = "call `"$VcVarsPath`" >nul && $Command"
    & cmd.exe /d /s /c $wrapped
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command"
    }
}

function Get-PythonValue {
    param([string]$Code)
    (& $Python -c $Code).Trim()
}

function Add-PathPrefix {
    param([string[]]$Paths)

    $existing = $Paths | Where-Object { $_ -and (Test-Path $_) }
    if ($existing.Count -gt 0) {
        $env:PATH = ($existing -join [IO.Path]::PathSeparator) + [IO.Path]::PathSeparator + $env:PATH
    }
}

if ($env:OS -ne "Windows_NT") {
    throw "This script is for Windows only."
}

Add-PathPrefix @(
    "C:\Program Files\CMake\bin",
    "C:\Program Files (x86)\CMake\bin"
)

if (-not $env:CUDA_PATH) {
    $bootstrapCudaRoots = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object {
            if ($_.Name -match 'v(\d+(?:\.\d+)*)') {
                [version]$Matches[1]
            } else {
                [version]"0.0"
            }
        } -Descending

    if ($bootstrapCudaRoots) {
        $env:CUDA_PATH = $bootstrapCudaRoots[0].FullName
    }
}

if ($env:CUDA_PATH) {
    Add-PathPrefix @((Join-Path $env:CUDA_PATH "bin"))
}

Write-Step "Checking required tools"
Require-Command $Python
Require-Command cmake
Require-Command nvcc

$vcvarsPath = Get-VcVarsPath
Write-Host "Visual Studio environment: $vcvarsPath"

$pythonExecutable = Get-PythonValue "import sys; print(sys.executable)"
$pythonPrefix = Get-PythonValue "import sys; print(sys.prefix)"
$sitePackages = Get-PythonValue "import site, sys; paths=[p for p in site.getsitepackages() if p.lower().endswith('site-packages')]; print(paths[0] if paths else '')"
if (-not $sitePackages) {
    $sitePackages = Get-PythonValue "import sysconfig; print(sysconfig.get_paths()['purelib'])"
}

Write-Host "Python executable: $pythonExecutable"
Write-Host "Python prefix: $pythonPrefix"

Write-Step "Installing Python build dependencies and cuDNN wheel"
& $Python -m pip install --upgrade pip "setuptools<81" wheel packaging
if ($LASTEXITCODE -ne 0) { throw "Failed to install Python build dependencies." }

& $Python -m pip install nvidia-cudnn-cu13
if ($LASTEXITCODE -ne 0) { throw "Failed to install nvidia-cudnn-cu13." }

$cudnnRoot = Join-Path $sitePackages "nvidia\cudnn"
$cudnnInclude = Join-Path $cudnnRoot "include"
$cudnnBin = Join-Path $cudnnRoot "bin"
$cu13Bin = Join-Path $sitePackages "nvidia\cu13\bin\x86_64"

if (-not (Test-Path (Join-Path $cudnnInclude "cudnn.h"))) {
    throw "cuDNN headers were not found under '$cudnnInclude'."
}

if (-not (Test-Path (Join-Path $cudnnBin "cudnn64_9.dll"))) {
    throw "cuDNN runtime DLL was not found under '$cudnnBin'."
}

if (-not $env:CUDA_PATH) {
    $cudaRoots = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object {
            if ($_.Name -match 'v(\d+(?:\.\d+)*)') {
                [version]$Matches[1]
            } else {
                [version]"0.0"
            }
        } -Descending
    if (-not $cudaRoots) {
        throw "CUDA_PATH is not set and no CUDA Toolkit directory was found."
    }
    $env:CUDA_PATH = $cudaRoots[0].FullName
}

$cudaBin = Join-Path $env:CUDA_PATH "bin"
Add-PathPrefix @($cudnnBin, $cu13Bin, $cudaBin)

$env:CudaToolkitDir = $env:CUDA_PATH
if ((Split-Path $env:CUDA_PATH -Leaf) -match 'v(\d+)\.(\d+)') {
    $cudaVersionEnvName = "CUDA_PATH_V$($Matches[1])_$($Matches[2])"
    Set-Item -Path "env:$cudaVersionEnvName" -Value $env:CUDA_PATH
}

Write-Host "CUDA_PATH: $env:CUDA_PATH"
Write-Host "cuDNN include: $cudnnInclude"
Write-Host "cuDNN bin: $cudnnBin"

$buildRoot = Join-Path ([IO.Path]::GetTempPath()) "dlib-cuda-build"
$buildId = Get-Date -Format "yyyyMMdd-HHmmss"
$workDir = Join-Path $buildRoot $buildId
$downloadDir = Join-Path $workDir "download"
$importLibDir = Join-Path $workDir "cudnn_import_lib"

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
New-Item -ItemType Directory -Force -Path $importLibDir | Out-Null

Write-Step "Creating cuDNN import library"
$cudnnDll = Join-Path $cudnnBin "cudnn64_9.dll"
$dumpOutput = & cmd.exe /d /s /c "call `"$vcvarsPath`" >nul && dumpbin /exports `"$cudnnDll`""
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect cuDNN exports with dumpbin."
}

$exportNames = foreach ($line in $dumpOutput) {
    if ($line -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(cudnn\S+)') {
        $Matches[1]
    }
}

if (-not $exportNames) {
    throw "No cuDNN exports were discovered."
}

$defPath = Join-Path $importLibDir "cudnn.def"
$libPath = Join-Path $importLibDir "cudnn.lib"
@("LIBRARY cudnn64_9.dll", "EXPORTS") + $exportNames | Set-Content -Encoding ASCII -Path $defPath
Invoke-VcCommand -VcVarsPath $vcvarsPath -Command "lib /def:`"$defPath`" /machine:x64 /out:`"$libPath`""

Write-Step "Downloading dlib source"
& $Python -m pip download --no-binary :all: --no-deps "dlib==$DlibVersion" -d $downloadDir
if ($LASTEXITCODE -ne 0) { throw "Failed to download dlib source." }

$sourceArchive = Get-ChildItem $downloadDir -Filter "dlib-$DlibVersion*.tar.gz" | Select-Object -First 1
if (-not $sourceArchive) {
    throw "Downloaded dlib source archive was not found."
}

tar -xf $sourceArchive.FullName -C $workDir
if ($LASTEXITCODE -ne 0) { throw "Failed to extract dlib source archive." }

$sourceDir = Join-Path $workDir "dlib-$DlibVersion"
$setupPath = Join-Path $sourceDir "setup.py"
if (-not (Test-Path $setupPath)) {
    throw "dlib setup.py was not found."
}

Write-Step "Patching dlib setup.py for CUDA/cuDNN environment variables"
$setupText = Get-Content -Raw -Path $setupPath
$new = '        cmake_env_prefixes = (
            "DLIB_",
            "CUDNN_",
            "CUDAToolkit_",
            "CMAKE_CUDA_ARCHITECTURES",
        )
        for key, value in os.environ.items():
            if key.startswith(cmake_env_prefixes):
                cmake_args_dict[key] = value'

$pattern = '(?ms)^        for key, value in os\.environ\.items\(\):\s*^            if key\.startswith\("DLIB_"\):\s*^                cmake_args_dict\[key\] = value'
if ($setupText -notmatch $pattern) {
    throw "Could not find the expected dlib setup.py block to patch."
}

[regex]::Replace($setupText, $pattern, $new, 1) | Set-Content -Encoding UTF8 -Path $setupPath

Write-Step "Building and installing CUDA-enabled dlib"
$env:DLIB_USE_CUDA = "1"
$env:CUDAToolkit_ROOT = $env:CUDA_PATH
$env:CUDNN_INCLUDE_DIR = $cudnnInclude
$env:CUDNN_LIBRARY = $importLibDir
$env:CMAKE_CUDA_ARCHITECTURES = $CudaArchitectures

& $Python -m pip uninstall -y dlib
if ($LASTEXITCODE -ne 0) { throw "Failed to uninstall the existing dlib package." }

$buildCommand = "cd /d `"$sourceDir`" && `"$pythonExecutable`" setup.py install"
Invoke-VcCommand -VcVarsPath $vcvarsPath -Command $buildCommand

Write-Step "Verifying CUDA-enabled dlib"
$verifyScript = Join-Path $workDir "verify_dlib_cuda.py"
$verifyCode = @"
import os
from pathlib import Path

site_packages = Path(r"$($sitePackages)")
paths = [
    site_packages / "nvidia" / "cudnn" / "bin",
    site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
    Path(r"$($env:CUDA_PATH)") / "bin",
]
for path in paths:
    if path.exists():
        os.add_dll_directory(str(path))

import dlib
print("DLIB_USE_CUDA:", dlib.DLIB_USE_CUDA)
print("CUDA devices:", dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0)
if not dlib.DLIB_USE_CUDA or dlib.cuda.get_num_devices() < 1:
    raise SystemExit(1)
"@

$verifyCode | Set-Content -Encoding UTF8 -Path $verifyScript
& $Python $verifyScript
if ($LASTEXITCODE -ne 0) {
    throw "dlib verification failed. CUDA is still not enabled."
}

Write-Host ""
Write-Host "CUDA-enabled dlib is installed successfully." -ForegroundColor Green
