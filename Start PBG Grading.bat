@echo off
rem PBG Grading - station launcher (Windows). Double-click to start.
rem Starts the app in the local environment; the browser opens by itself.
rem
rem Per-station overrides (remove "rem " to enable):
rem set "PBG_HOST=0.0.0.0"   - also serve the app to other machines on the LAN
rem set "PBG_PORT=8050"      - change the port
rem set "PBG_NO_BROWSER=1"   - don't auto-open the browser
setlocal
set "HERE=%~dp0"

if not exist "%HERE%.micromamba\micromamba.exe" goto :nosetup
if not exist "%HERE%envs\grading" goto :nosetup

rem Activate the env by PATH and run python.exe DIRECTLY (not via
rem `micromamba run`). This makes the app a normal child process so the PyQt6
rem calibration dialogs it spawns open in the foreground, and it puts the
rem env's library dirs on PATH for correct DLL resolution.
set "ENV=%HERE%envs\grading"
set "CONDA_PREFIX=%ENV%"
set "PATH=%ENV%;%ENV%\Library\mingw-w64\bin;%ENV%\Library\usr\bin;%ENV%\Library\bin;%ENV%\Scripts;%ENV%\bin;%PATH%"
"%ENV%\python.exe" "%HERE%PBG_grading.py"
echo.
echo PBG Grading stopped.
pause
exit /b

:nosetup
echo Environment not found - run setup_windows.bat first.
pause
exit /b 1
