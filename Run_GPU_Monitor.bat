@echo off
ECHO ======================================================
ECHO  GPU PERFORMANCE MONITOR
ECHO ======================================================
ECHO  This will monitor your GPU usage and VRAM in real-time
ECHO  Press Ctrl+C to stop monitoring
ECHO ======================================================

ECHO.
ECHO ======================================================
ECHO  ACTIVATING VIRTUAL ENVIRONMENT ^& STARTING MONITOR
ECHO ======================================================
IF EXIST .\.venv\Scripts\activate.bat (
    CALL .\.venv\Scripts\activate.bat
    ECHO Virtual environment activated successfully
) ELSE (
    ECHO Warning: Virtual environment not found, using system Python
)

ECHO.
ECHO Starting GPU Monitor...
python GPU_Monitor.py
IF ERRORLEVEL 1 (
    ECHO.
    ECHO ERROR: Python script failed with error code %ERRORLEVEL%
    ECHO Trying to run with detailed error output...
    python -u GPU_Monitor.py
)

ECHO.
ECHO ======================================================
ECHO  GPU MONITORING FINISHED
ECHO ======================================================
PAUSE
