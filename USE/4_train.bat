@echo off
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%CD%\code"
echo.
echo === GPU TRAIN ===
echo.
python scripts\gpu_train.py
echo.
pause
