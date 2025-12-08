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
REM 4. Launch server and recorder in new windows
REM ------------------------------
echo Starting server...
start cmd /k "python server_v4.py"

REM Optional: wait 2 seconds to let server start
timeout /t 2 /nobreak >nul

start "" "http://localhost:8000/static/operator.html"

echo Setup complete.
ENDLOCAL
