@echo off
cd /d "%~dp0.."
echo.
echo === SELF HEAL EPOCH ===
echo.
python scripts\self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
echo.
pause
