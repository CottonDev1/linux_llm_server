@echo off
REM Start Prefect Dashboard for Pipeline Orchestration
REM This runs the Prefect server on port 4200

echo ============================================
echo Starting Prefect Dashboard...
echo ============================================

cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if Prefect is installed
python -c "import prefect" 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Prefect is not installed!
    echo Run: pip install "prefect>=3.0.0"
    pause
    exit /b 1
)

echo.
echo Prefect Dashboard starting at: http://localhost:4200
echo.
echo Opening browser...
start http://localhost:4200
echo.
echo Press Ctrl+C to stop the server
echo.
echo Prefect Commands:
echo   # Start Prefect server: prefect server start --port 4200
echo   # Start Prefect worker: prefect worker start --pool "local-process"
echo.

REM Check/create profile
prefect profile create LLM_PYTHON 2>nul
prefect profile use LLM_PYTHON 2>nul
prefect config set PREFECT_API_URL=http://localhost:4200/api 2>nul

REM Start Prefect server
prefect server start --host 0.0.0.0 --port 4200

pause
