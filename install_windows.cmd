@echo off
setlocal

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_windows.ps1" %*

if errorlevel 1 (
    echo.
    echo Installation failed. Review the output above.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully.
pause
