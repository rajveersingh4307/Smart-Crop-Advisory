@echo off
cd /d "%~dp0ml_backend"
python -m uvicorn main:app --reload --port 8000
pause
