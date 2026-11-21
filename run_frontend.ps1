Set-Location "$PSScriptRoot\academic-assistant\frontend"
Write-Host "Starting Streamlit Frontend at http://localhost:8501..." -ForegroundColor Cyan
& "..\venv\Scripts\streamlit.exe" run app.py
