@echo off
cd /d "%~dp0academic-assistant\frontend"
echo ========================================================
echo Starting Streamlit Frontend at http://localhost:8501
echo ========================================================
..\venv\Scripts\streamlit.exe run app.py
pause
