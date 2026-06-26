@echo off
REM Kevin's Cat App - Windows setup launcher.
REM Double-click this file, or run it from a Command Prompt in this folder.
REM It just runs setup.ps1 with PowerShell (bypassing the execution policy for
REM this one script, without changing any system settings).

echo Running Kevin's Cat App setup...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
pause
