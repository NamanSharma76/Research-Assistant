Set-Location "$PSScriptRoot\academic-assistant\backend"
Write-Host "Starting FastAPI Backend at http://localhost:8000..." -ForegroundColor Cyan
Write-Host "Interactive API docs at http://localhost:8000/docs" -ForegroundColor Gray
& "..\venv\Scripts\uvicorn.exe" app.api.main:app --reload --port 8000
