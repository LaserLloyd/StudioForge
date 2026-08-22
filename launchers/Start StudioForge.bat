@echo off
setlocal EnableExtensions
title StudioForge

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

echo.
echo   StudioForge
echo   -----------
echo   repo     : %REPO%
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
