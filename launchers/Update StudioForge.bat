@echo off
setlocal EnableExtensions EnableDelayedExpansion
title StudioForge - update

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

echo.
echo   StudioForge update
echo   ==================
echo   repo     : %REPO%
echo   data dir : %SF_DATA_DIR%
echo.

REM ---------------------------------------------------------------- 1. code
where git >nul 2>&1
if "%ERRORLEVEL%"=="0" (
  git rev-parse --is-inside-work-tree >nul 2>&1
  if "!ERRORLEVEL!"=="0" (
    git remote >nul 2>&1
    for /f %%R in ('git remote') do set "HASREMOTE=1"
    if defined HASREMOTE (
      echo [1/4] Pulling latest code...
      git pull --ff-only
    ) else (
      echo [1/4] No git remote configured - skipping code update.
    )
  ) else (
    echo [1/4] Not a git checkout - skipping code update.
  )
) else (
  echo [1/4] git not found - skipping code update.
)

REM -------------------------------------------------------- 2. environment
echo.
echo [2/4] Syncing the Python environment...
where uv >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo   ERROR: 'uv' is not installed or not on PATH.
  echo   Install it from https://docs.astral.sh/uv/ then re-run this script.
  pause
  exit /b 1
)
if not exist "%PY%" (
  echo   Creating the virtual environment...
  uv venv --python 3.12 .venv || goto :failed
)
uv pip install --python "%PY%" -e ".[dev]" || goto :failed

REM ------------------------------------------------------------- 3. engine
echo.
echo [3/4] Checking for a newer llama.cpp engine...
REM --update installs the newest release, smoke-tests it, and only then makes
REM it the default. A build that fails its micro-load is never pinned.
"%PY%" -m studioforge engine --update
if not "%ERRORLEVEL%"=="0" (
  echo   Engine update did not complete - the previous engine is still active.
)

REM ------------------------------------------------------------- 4. verify
echo.
echo [4/4] Verifying...
"%PY%" -m studioforge scan
"%PY%" -m studioforge engine --smoke-test

echo.
echo   Update complete. Start the server with "launchers\Start StudioForge.bat".
echo.
pause
exit /b 0

:failed
echo.
echo   Update FAILED. Your previous install is untouched.
echo.
pause
exit /b 1
