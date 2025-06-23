@echo off
ECHO ======================================================
ECHO  GPU Driver and CUDA Toolkit Setup
ECHO  This script will check for and install NVIDIA drivers
ECHO  and the CUDA Toolkit using Chocolatey.
ECHO  It MUST be run as an Administrator.
ECHO ======================================================

REM --- Step 1: Check if NVIDIA drivers are already installed ---
ECHO Checking for existing NVIDIA drivers...
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO [SUCCESS] NVIDIA drivers are already installed.
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    GOTO :EOF
)

ECHO [INFO] NVIDIA drivers not found. Proceeding with installation.

REM --- Step 2: Install Chocolatey ---
powershell -NoProfile -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

REM --- Step 3: Install CUDA Toolkit ---
ECHO Installing CUDA Toolkit (includes the driver)...
ECHO This will target CUDA version 12.1, which is compatible with modern PyTorch.
choco install cuda --version=12.1.1 -y

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Failed to install the CUDA Toolkit.
    ECHO Please check the log for details.
    GOTO :EOF
)

ECHO.
ECHO ======================================================
ECHO  GPU & CUDA SETUP COMPLETE!
ECHO ======================================================
ECHO A system restart is highly recommended to ensure drivers are loaded correctly.
ECHO After restarting, you can proceed with the main project setup.
ECHO ======================================================
ECHO.
PAUSE

:EOF
