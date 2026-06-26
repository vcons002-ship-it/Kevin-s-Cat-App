@echo off
REM Kevin's Cat App - start the app on Windows.
REM Double-click this after you've run setup.bat once.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo.
    echo Setup hasn't been run yet ^(no .\venv found^).
    echo Please run setup.bat first, then double-click start.bat again.
    echo.
    pause
    exit /b 1
)

echo Starting Kevin's Cat App...
echo Open the printed http://... address in a browser on the same WiFi.
echo Close this window ^(or press Ctrl+C^) to stop.
echo.

"venv\Scripts\python.exe" run.py

echo.
echo Kevin's Cat App has stopped.
pause
