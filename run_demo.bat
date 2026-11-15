@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python run_demo.py
pause
