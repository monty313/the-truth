@echo off
cd /d "%~dp0.."
echo.
echo === Visual map tools (VS Code Grok) ===
echo.
echo 1) CodeVisualizer is a VS Code extension.
echo    Press Ctrl+Shift+X, search CodeVisualizer.
echo    Right-click the "code" folder -^> Visualize Codebase Flow
echo.
echo 2) Understand-Anything lives in:
echo    tools\Understand-Anything\
echo    Guide: tools\Understand-Anything\00_HOW_TO_USE_IN_VSCODE.md
echo.
echo 3) Grok chat cannot run /understand.
echo    Use Copilot Chat for that plugin, or CodeVisualizer.
echo.
start "" notepad "tools\Understand-Anything\00_HOW_TO_USE_IN_VSCODE.md"
pause
