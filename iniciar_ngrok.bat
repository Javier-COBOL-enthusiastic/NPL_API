@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro .venv. Sigue la instalacion del README.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u "scripts\iniciar_ngrok.py"
set "RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RESULT%
