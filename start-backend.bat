@echo off
echo Starting AI Workflow Copilot Backend...
echo.

REM Check if GITHUB_TOKEN is set
if "%GITHUB_TOKEN%"=="" (
    echo ERROR: GITHUB_TOKEN environment variable is not set!
    echo.
    echo Please set your GitHub token:
    echo   set GITHUB_TOKEN=your_token_here
    echo.
    echo Or create a .env file in the backend directory.
    pause
    exit /b 1
)

cd backend
echo Installing/updating dependencies...
pip install -r requirements.txt
echo.

echo Starting server on http://localhost:8000
echo API Documentation: http://localhost:8000/api/v1/docs
echo.
python -m uvicorn app.main:app --reload --port 8000
