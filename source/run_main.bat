@echo off
setlocal
set "PYTHON_EXE=C:\Users\lss\anaconda3\envs\gr_main\python.exe"
cd /d "%~dp0"

if not exist "%PYTHON_EXE%" (
    echo The gr_main Python environment was not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py
if errorlevel 1 (
    echo.
    echo Droplet Analysis System exited with an error.
    pause
)

endlocal

