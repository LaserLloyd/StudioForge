@echo off
setlocal EnableExtensions
title StudioForge - open control panel

REM These launchers live in <repo>\launchers\; everything they need is one
REM level up. Resolve that once, as an absolute path, and work from there.
for %%I in ("%~dp0..") do set "REPO=%%~fI"
cd /d "%REPO%"

REM Optional and gitignored: your own machine-specific overrides. Create
REM local-env.bat in the repo root (a template is in launchers\) and set
REM SF_DATA_DIR in it to keep your data outside the checkout -- for example
REM C:\Users\<you>\StudioForge-data -- without touching a tracked file.
if exist "%REPO%\local-env.bat" call "%REPO%\local-env.bat"
if exist "%~dp0local-env.bat" call "%~dp0local-env.bat"

REM Config, registry, engines and logs live in the data dir. Default:
REM <repo>\data, which .gitignore keeps out of the repository. One data
REM dir serves ONE running instance (see README, "Data directory").
if not defined SF_DATA_DIR set "SF_DATA_DIR=%REPO%\data"

set "PY=%REPO%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo.
  echo   The virtual environment is missing:
  echo     %PY%
  echo.
  echo   Run "launchers\Update StudioForge.bat" once to create it, then try again.
  echo.
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
  echo   Start the server first with "launchers\Start StudioForge.bat".
  echo.
  pause
)
exit /b %RC%
