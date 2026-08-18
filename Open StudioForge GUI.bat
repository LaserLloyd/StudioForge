@echo off
setlocal EnableExtensions
title StudioForge - open control panel
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
  echo Virtual environment missing - run "Update StudioForge.bat" first.
  pause
  exit /b 1
)

REM --wait polls the GUI first, so if the server is not running we say that
REM plainly instead of opening a browser tab at a dead port.
"%PY%" -m studioforge gui --wait
REM Capture immediately: `pause` below succeeds and resets ERRORLEVEL to 0, so
REM reading it after the block reported success on every failure.
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo   The control panel did not answer.
  echo   Start the server first with "Start StudioForge.bat".
  echo.
  pause
)
exit /b %RC%
