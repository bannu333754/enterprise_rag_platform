@echo off
title Enterprise RAG App
echo Starting RAG Engine and Desktop Application...
taskkill /F /IM python.exe >nul 2>&1
start /b python -m uvicorn day75_unified_production_app:app --host 127.0.0.1 --port 8000
timeout /t 5 /nobreak >nul
start msedge.exe --app=http://127.0.0.1:8000 --window-size=1200,820
exit
