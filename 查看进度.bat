@echo off
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
set PYTHONUNBUFFERED=1
"C:\Users\32201\anaconda3\python.exe" monitor.py --watch 60
echo.
echo Stopped. You can close this window, or press any key.
pause