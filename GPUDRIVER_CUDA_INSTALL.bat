@echo off

REM =====================================================================
REM  Step 0: Check for Administrator Privileges and Self-Elevate
REM =====================================================================
ECHO Checking for administrator privileges...
net session >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Administrator privileges are required.
    ECHO Attempting to re-launch this script as an administrator...
    ECHO Please approve the UAC prompt.
    powershell -Command "Start-Process -FilePath '%~s0' -Verb RunAs"
    GOTO :EOF
)
ECHO [SUCCESS] Administrator privileges confirmed.


ECHO ======================================================
ECHO  GPU Driver and CUDA Toolkit Setup
ECHO  This script will check for and install NVIDIA drivers
ECHO  and the CUDA Toolkit using Chocolatey.
ECHO ======================================================
ECHO.
PAUSE

REM --- Step 1: Check if NVIDIA drivers are already installed ---
ECHO [STEP 1] Checking for existing NVIDIA drivers...
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO [SUCCESS] NVIDIA drivers are already installed.
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    ECHO No further action is needed from this script.
    GOTO :End
)
ECHO [INFO] NVIDIA drivers not found. Proceeding with installation.
ECHO.

REM --- Step 2: Install Chocolatey ---
ECHO [STEP 2] Attempting to install Chocolatey package manager...
ECHO This requires an internet connection and administrator privileges.
powershell -NoProfile -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

REM --- Check for installation failure ---
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [FATAL ERROR] The Chocolatey installation command failed.
    ECHO The red text above from PowerShell contains the specific error details.
    ECHO This is often due to network issues or not running the script as an Administrator.
    ECHO Please review the errors and try again.
    GOTO :End
)
ECHO [SUCCESS] Chocolatey installation command completed.
SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
ECHO.

REM --- Step 3: Install CUDA Toolkit ---
ECHO [STEP 3] Installing CUDA Toolkit (includes the driver)...
ECHO This will install the latest stable CUDA Toolkit version from Chocolatey.
ECHO This step can take a significant amount of time. Please be patient.
choco install cuda -y

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Failed to install the CUDA Toolkit.
    ECHO Please check the Chocolatey log for details: C:\ProgramData\chocolatey\logs\chocolatey.log
    GOTO :End
)
ECHO [SUCCESS] CUDA Toolkit has been installed.

ECHO.
ECHO ======================================================
ECHO  GPU & CUDA SETUP COMPLETE!
ECHO ======================================================
ECHO A system restart is highly recommended to ensure drivers are loaded correctly.
ECHO After restarting, you can proceed with the main project setup.
ECHO ======================================================

:End
ECHO.
PAUSE
