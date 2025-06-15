@echo off
cd /d %~dp0
C:\Users\user\miniconda3\python.exe -m uvicorn server:app --host 0.0.0.0 --port 9001
cmd