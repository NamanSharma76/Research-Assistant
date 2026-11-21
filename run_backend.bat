@echo off
cd /d "%~dp0academic-assistant\backend"
echo ========================================================
echo Starting FastAPI Backend at http://localhost:8000
echo Interactive API docs at http://localhost:8000/docs
echo ========================================================
..\venv\Scripts\uvicorn.exe app.api.main:app --reload --port 8000
pause
