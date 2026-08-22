@echo off
setlocal EnableExtensions
title StudioForge - system tray

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

REM pythonw.exe has no console, so the tray does not leave a black window
REM sitting behind the icon for as long as the app runs. python.exe is the
REM fallback for a venv built without the windowless launcher.
set "PY=%REPO%\.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=%REPO%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo.
  echo   The virtual environment is missing:
  echo     %REPO%\.venv\Scripts\
  echo.
  echo   Run "launchers\Update StudioForge.bat" once to create it, then try again.
  echo.
  pause
  exit /b 1
)

REM start /b so this console closes immediately instead of waiting for the
REM tray to quit; the tray itself keeps running in the notification area. If
REM a server is already running from this data dir, the tray attaches to it.
start "" /b "%PY%" -m studioforge tray %*
exit /b 0
