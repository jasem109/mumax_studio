@echo off
rem mumax Studio launcher for Windows. Double-click to start the GUI.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.9+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
python -c "import numpy, scipy, PIL" 2>nul || python -m pip install --user -r requirements.txt
python -m mumaxstudio gui %*
if errorlevel 1 pause
