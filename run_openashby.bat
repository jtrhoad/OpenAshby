@echo off
REM Launches OpenAshby. Save this file inside the OpenAshby folder
REM (next to Main.py) and double-click it.

REM %~dp0 expands to the folder this .bat file lives in, so the script
REM works no matter where the OpenAshby folder is on your machine.
cd /d "%~dp0"

REM Activate the virtual environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo Could not activate .venv. Make sure you've created it with:
    echo     python -m venv .venv
    echo and installed dependencies with:
    echo     pip install pandas numpy matplotlib scipy
    echo.
    pause
    exit /b 1
)

REM Run the program
python Main.py

REM Keep the window open if Python errored out, so you can read the traceback
if errorlevel 1 (
    echo.
    echo Main.py exited with an error.
    pause
)
