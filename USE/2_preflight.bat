@echo off
cd /d "%~dp0.."
echo.
echo === PREFLIGHT ===
echo.
python scripts\preflight_train.py
echo.
pause
