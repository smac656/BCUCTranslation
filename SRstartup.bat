@echo off
SETLOCAL

REM ------------------------------
REM 1. Create virtual environment if it doesn't exist
REM ------------------------------
IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    py -m venv .venv
) ELSE (
    echo Virtual environment already exists.
)

REM ------------------------------
REM 2. Activate venv
REM ------------------------------
call .venv\Scripts\activate.bat

REM ------------------------------
REM 3. Upgrade pip and install dependencies
REM ------------------------------
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM ------------------------------
REM 4. Launch FastAPI + Realtime server using Uvicorn
REM ------------------------------
echo Starting FastAPI transcription server...
start cmd /k "python -m uvicorn server_v4:app --host 0.0.0.0 --port 8000 --reload"
echo %OPENAI_API_KEY%

REM Optional: wait for server to initialize
timeout /t 2 /nobreak >nul

REM ------------------------------
REM 5. Open operator view in browser
REM ------------------------------
start "" "http://localhost:8000/static/operator.html"

echo Setup complete.
ENDLOCAL
