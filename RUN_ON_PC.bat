@echo off
REM ============================================================
REM  MOMENTUM ONE - continuous drill trainer (Monty's PC)
REM  Trains NON-STOP toward 2x the goal with 0 breaches.
REM  - resumes the working checkpoint (warm-starts from best_trading)
REM  - saves best_trading ONLY when it gets MORE CONSISTENT
REM  Stop anytime: close this window or press Ctrl+C.
REM  Resume: just double-click this file again.
REM ============================================================
cd /d "%~dp0"
echo Starting continuous training. Progress -> artifacts\drill2x_progress.json
:loop
python scripts\drill_2x.py --minutes 600 --eval-every 8 --ckpt drill_live
echo(
echo Chunk ended. Resuming in 3 seconds (Ctrl+C to stop)...
timeout /t 3 >nul
goto loop
