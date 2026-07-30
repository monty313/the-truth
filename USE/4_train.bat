@echo off
cd /d "%~dp0.."
echo.
echo === GPU TRAIN ===
echo.
python scripts\gpu_train.py
echo.
pause
