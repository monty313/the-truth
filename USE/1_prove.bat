@echo off
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%CD%\code"
echo.
echo === PROVE IT (champion 3.0 / 3.5) ===
echo.
python scripts\prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
echo.
pause
