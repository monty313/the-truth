@echo off
REM Safe tests for the NEW brain only. Does NOT touch PROVEN.
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%CD%\code"
echo.
echo PYTHONPATH uses code\ for packages (FinRL-clean layout).
echo === NEW brain tests (lineage only) ===
echo Folder: lineages\adaptive_rl_brain_7_31_26
echo PROVEN is NOT modified.
echo.
python tests\lineages\adaptive_rl_brain_7_31_26\test_sets.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_confluence.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_structure.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_classify.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_mindless_wall.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_live_indicators.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_pipeline.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_observation.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_rewards.py
if errorlevel 1 goto fail
python tests\lineages\adaptive_rl_brain_7_31_26\test_mtf_and_train.py
if errorlevel 1 goto fail
echo.
echo === ALL NEW-BRAIN TESTS OK ===
echo.
pause
exit /b 0
:fail
echo.
echo === TEST FAILED ===
pause
exit /b 1
