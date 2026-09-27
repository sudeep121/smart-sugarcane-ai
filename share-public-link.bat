@echo off
REM Publishes your locally-running app to a temporary public HTTPS URL.
REM
REM Requires run-backend.bat to be running first, with the frontend built
REM (cd frontend ^&^& npm run build) so the backend serves the website too.
REM
REM Keep this window open - closing it kills the public link.
cd /d "%~dp0"

echo ============================================================
echo   Smart Sugarcane AI - PUBLIC LINK
echo ============================================================
echo.
echo Checking the app is running locally...

curl -s -o nul -m 5 http://127.0.0.1:8000/api/system/health
if errorlevel 1 (
    echo.
    echo [ERROR] The app is not running on port 8000.
    echo         Start run-backend.bat first, wait for
    echo         "Application startup complete.", then run this again.
    echo.
    pause
    exit /b 1
)

echo   OK - app is running.
echo.
echo Your visitors will be asked for a one-time password on the first
echo visit. That password is your public IP address:
echo.
curl -s -m 10 https://loca.lt/mytunnelpassword
echo.
echo.
echo Opening the tunnel. The URL appears below in a few seconds.
echo Keep this window OPEN. Close it to take the site offline.
echo ============================================================
echo.

REM A fixed subdomain keeps the URL the same every time you run this,
REM so you can put it in a report. If it is already taken, localtunnel
REM falls back to a random name and prints that instead.
npx --yes localtunnel --port 8000 --subdomain smart-sugarcane-ai

echo.
echo Public link closed.
pause
