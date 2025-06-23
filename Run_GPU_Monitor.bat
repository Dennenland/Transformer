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
CALL .\.venv\Scripts\activate.bat
python GPU_Monitor.py

ECHO.
ECHO ======================================================
ECHO  GPU MONITORING FINISHED
ECHO ======================================================
PAUSE
