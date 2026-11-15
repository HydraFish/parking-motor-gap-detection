@echo off
title MotoSpace CV Telemetry Dashboard
echo ========================================================
echo  Starting MotoSpace CV Telemetry Dashboard...
echo ========================================================
echo.
echo URL: http://localhost:8000
echo.
start http://localhost:8000
python app.py
pause
