@echo off
rem Chay chatbot web NeoRAG (app/server.py). Bam dup file nay, hoac:
rem   run_server.bat [cong]          chi may nay mo duoc (127.0.0.1)
rem   run_server.bat lan [cong]      cho may khac cung mang LAN truy cap (0.0.0.0)
setlocal
cd /d "%~dp0"

set "HOST=127.0.0.1"
if /i "%~1"=="lan" (
    set "HOST=0.0.0.0"
    shift
)
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if not exist ".env" (
    echo [canh bao] Chua co file .env: chi tra loi duoc cac cau da co trong cache.
    echo            Tao bang: copy .env.example .env  roi dien OPENAI_API_KEY.
)

if "%HOST%"=="0.0.0.0" (
    echo Che do LAN: may khac trong cung mang mo mot trong cac dia chi sau:
    powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254*' } | ForEach-Object { '  http://' + $_.IPAddress + ':%PORT%   (' + $_.InterfaceAlias + ')' }"
    echo Neu may khac khong vao duoc: cho phep Python qua Windows Firewall, hoac doi mang Wi-Fi sang Private.
)

echo Dang mo http://localhost:%PORT%  (Ctrl+C de dung)
rem Doi khoang 2 giay cho server mo cong roi moi mo trinh duyet (NO_BROWSER=1 de bo qua)
if not defined NO_BROWSER start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & start "" http://localhost:%PORT%"

"%PY%" app\server.py --host %HOST% --port %PORT%
if errorlevel 1 pause
endlocal
