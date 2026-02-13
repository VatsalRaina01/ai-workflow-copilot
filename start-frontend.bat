@echo off
echo Starting AI Workflow Copilot Frontend...
echo.

cd frontend

if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    echo.
)

echo Starting development server on http://localhost:5173
echo.
npm run dev
