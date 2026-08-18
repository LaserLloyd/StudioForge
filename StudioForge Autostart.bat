@echo off
setlocal EnableExtensions
title StudioForge - start at login
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

echo.
echo   StudioForge - start at login
echo   ============================
echo.
"%PY%" -m studioforge autostart status
echo.
echo   [1] Enable  - start StudioForge automatically when you log in
echo   [2] Enable and open the control panel each time
echo   [3] Disable - stop starting automatically
echo   [4] Cancel
echo.
set /p "CHOICE=Choose 1-4: "

if "%CHOICE%"=="1" goto :enable
if "%CHOICE%"=="2" goto :enable_open
if "%CHOICE%"=="3" goto :disable
goto :done

:enable
"%PY%" -m studioforge autostart enable
goto :done

:enable_open
"%PY%" -m studioforge autostart enable --open
goto :done

:disable
"%PY%" -m studioforge autostart disable
goto :done

:done
echo.
REM Enabling writes a small hidden-launch script into your Startup folder; it
REM needs no administrator rights and option 3 removes it again.
"%PY%" -m studioforge autostart status
echo.
pause
exit /b 0
