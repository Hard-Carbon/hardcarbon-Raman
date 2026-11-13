@echo off
cd /d "%~dp0"
if exist "C:\Python314\python.exe" (
  "C:\Python314\python.exe" -u main.py
) else (
  python -u main.py
)
pause
