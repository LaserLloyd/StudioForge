@echo off
setlocal EnableExtensions
title StudioForge - start at login

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
echo   StudioForge - start at login
echo   ============================
echo.
"%PY%" -m studioforge autostart status
echo.
echo   [1] Enable  - start the system tray at login (it brings the server up,
echo                 restarts it if it crashes, and sits in the notification area)
echo   [2] Enable  - server only, no tray
echo   [3] Enable  - server only, and open the control panel each time
echo   [4] Disable - stop starting automatically
echo   [5] Cancel
echo.
set /p "CHOICE=Choose 1-5: "

if "%CHOICE%"=="1" goto :enable_tray
if "%CHOICE%"=="2" goto :enable
if "%CHOICE%"=="3" goto :enable_open
if "%CHOICE%"=="4" goto :disable
goto :done

:enable_tray
"%PY%" -m studioforge autostart enable --tray
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
REM needs no administrator rights and option 4 removes it again.
"%PY%" -m studioforge autostart status
echo.
pause
exit /b 0
