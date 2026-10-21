@echo off
REM Starts the Smart Sugarcane AI backend and serves it to the whole network.
REM Keep this window open while using the app.
cd /d "%~dp0backend"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at backend\venv
    echo.
    echo Create it first:
    echo     cd backend
    echo     python -m venv venv
    echo     venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo Creating backend\.env from .env.example ...
    copy ".env.example" ".env" >nul
)

REM This machine's address on the local network, so the banner can print a URL
REM that other devices can actually open.
set LANIP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    if not defined LANIP set LANIP=%%a
)
if defined LANIP set LANIP=%LANIP: =%

echo ============================================================
echo   Smart Sugarcane AI - BACKEND
echo.
echo   On this computer : http://localhost:8000
if defined LANIP echo   On other devices : http://%LANIP%:8000
echo   API docs         : http://localhost:8000/docs
echo.
echo   For a phone or another laptop to reach this:
echo     1. It must be on the SAME Wi-Fi as this computer.
echo     2. Windows Firewall must allow port 8000. Run once in
echo        an ADMIN PowerShell:
echo          New-NetFirewallRule -DisplayName "Smart Sugarcane AI"
echo            -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
echo.
echo   The address changes when you join a different network, so
echo   re-read the line above after switching Wi-Fi.
echo.
echo   Wait for "Application startup complete." before using
echo   the web app. Press Ctrl+C to stop.
echo ============================================================
echo.

REM --host 0.0.0.0 listens on every network interface. The previous default
REM bound to loopback only, which is why "localhost" worked here and no other
REM device could connect - every machine resolves localhost to itself.
venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

echo.
echo Backend stopped.
pause
