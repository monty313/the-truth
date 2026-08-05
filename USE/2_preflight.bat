@echo off
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%CD%\code"
echo.
echo === PREFLIGHT ===
echo.
python scripts\preflight_train.py
echo.
pause
