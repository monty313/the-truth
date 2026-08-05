@echo off
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%CD%\code"
start "" notepad "GOAL.md"
