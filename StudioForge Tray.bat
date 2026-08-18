@echo off
setlocal EnableExtensions
title StudioForge - system tray
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

REM pythonw.exe has no console, so the tray does not leave a black window
REM sitting behind the icon for as long as the app runs. python.exe is the
REM fallback for a venv built without the windowless launcher.
set "PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo.
  echo   The virtual environment is missing:
  echo     %~dp0.venv\Scripts\
  echo.
  echo   Run "Update StudioForge.bat" once to create it, then try again.
  echo.
  pause
  exit /b 1
)

REM start /b so this console closes immediately instead of waiting for the
REM tray to quit; the tray itself keeps running in the notification area.
start "" /b "%PY%" -m studioforge tray %*
exit /b 0
