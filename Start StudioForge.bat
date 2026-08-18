@echo off
setlocal EnableExtensions
title StudioForge
cd /d "%~dp0"

REM Optional and gitignored: your own machine-specific overrides. Create
REM local-env.bat next to this file and set SF_DATA_DIR (or SF_CONFIG,
REM SF_TEST_MODELS_DIR, ...) in it to point this checkout at an existing
REM install without touching a tracked file.
if exist "%~dp0local-env.bat" call "%~dp0local-env.bat"

REM Config, registry, engines and logs live in the data dir. Default:
REM <repo>\data, which .gitignore keeps out of the repository. One data
REM dir serves ONE running instance (see README, "Data directory").
if not defined SF_DATA_DIR set "SF_DATA_DIR=%~dp0data"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo.
  echo   The virtual environment is missing:
  echo     %PY%
  echo.
  echo   Run "Update StudioForge.bat" once to create it, then try again.
  echo.
  pause
  exit /b 1
)

echo.
echo   StudioForge
echo   -----------
echo   data dir : %SF_DATA_DIR%
echo   API      : http://127.0.0.1:1234/v1     (point OpenClaw here)
echo   GUI      : http://127.0.0.1:8080        (opens automatically)
echo.
echo   Close this window or press Ctrl+C to stop the server.
echo.

REM --open waits for the control panel to answer before launching the browser,
REM so you never land on a connection-refused page during a slow first start.
"%PY%" -m studioforge serve --open %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo   StudioForge exited with code %RC%.
  echo   If the port is already in use, LM Studio may be running on 1234 -
  echo   quit it, or change server.port in config.yaml.
  echo.
  pause
)
exit /b %RC%
