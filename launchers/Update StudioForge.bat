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

REM This step rewrites engines\ INSIDE THE LIVE DATA DIR, and a running server
REM holds llama-server.exe and its DLLs open through every loaded model. Ask
REM the gateway whether it is up before touching anything (D49-14): a reinstall
REM under a live server is refused by the installer anyway, and a smoke test
REM would take a GPU out from under whatever is using it. Two seconds, no
REM dependencies beyond the venv Python that step 2 just built. The DEFAULT
REM port only: a server moved off 1234 by config.yaml is not detected here, so
REM stop it yourself before running this.
set "SF_SERVER_LIVE="
"%PY%" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:1234/health',timeout=2)" >nul 2>&1
if "%ERRORLEVEL%"=="0" set "SF_SERVER_LIVE=1"

if defined SF_SERVER_LIVE (
  echo   SKIPPED: a StudioForge server is answering on http://127.0.0.1:1234.
  echo   Updating the engine under a running server would overwrite binaries its
  echo   loaded models are executing. Use the running server instead:
  echo     - GUI: Setup tab ^(or Server tab^) - llama.cpp engine - Check / Install,
  echo       then Activate; then Dashboard - Restart engines.
  echo     - REST: POST /api/engine/install then POST /api/engine/activate.
  echo   Or stop the server ^("Stop StudioForge.bat"^) and re-run this script.
) else (
  REM --update installs the newest release, smoke-tests the INSTALLED-but-not-
  REM yet-active build, and activates and pins it only if that test passes
  REM (D49-4). A build that fails is left on disk, unused, and the previous
  REM engine stays active - which is what the message says, now truthfully.
  "%PY%" -m studioforge engine --update
  if not "!ERRORLEVEL!"=="0" (
    echo   Engine update did not complete - the previous engine is still active.
  )
)

REM ------------------------------------------------------------- 4. verify
echo.
echo [4/4] Verifying...
"%PY%" -m studioforge scan
if defined SF_SERVER_LIVE (
  echo   Skipping the engine smoke test: it micro-loads a model on the GPU, and
  echo   a server is running. Use the GUI's "Smoke test" button instead.
) else (
  "%PY%" -m studioforge engine --smoke-test
)

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
