@echo off
setlocal
rem ===========================================================================
rem PBG Grading - one-time station setup (Windows). Double-click me.
rem
rem Pure batch (cmd) on purpose: no PowerShell *script file* is run, so this
rem works even on domain/managed machines where Group Policy forces an
rem AllSigned PowerShell execution policy (which blocks unsigned .ps1 files
rem and overrides -ExecutionPolicy Bypass). It:
rem   1. downloads micromamba (a single small .exe; no admin rights, doesn't
rem      touch any system Python),
rem   2. builds the pinned conda environment from environment.yml into
rem      .\envs\grading inside this folder,
rem   3. tries to put a "PBG Grading" shortcut on the Desktop (optional).
rem Re-running is safe: it rebuilds the environment from scratch.
rem ===========================================================================
cd /d "%~dp0"

set "HERE=%~dp0"
set "MM=%HERE%.micromamba\micromamba.exe"
set "ENVPREFIX=%HERE%envs\grading"
set "MAMBA_ROOT_PREFIX=%HERE%.micromamba"

if not exist "%HERE%environment.yml" (
  echo ERROR: environment.yml not found next to this script.
  goto :fail
)

rem Heads-up when the install lives inside a synced folder: sync clients hold
rem file locks that can break env deletes/rebuilds. Stations run best from a
rem plain local folder (e.g. C:\PBG Grading).
echo "%HERE%" | findstr /i "Dropbox OneDrive" >nul && (
  echo NOTE: this folder appears to be inside Dropbox/OneDrive. If setup
  echo fails with "file in use" errors, pause syncing and run setup again.
  echo Consider installing stations outside synced folders.
  echo.
)

rem --- 0. Keep Dropbox away from the environment ----------------------------
rem If this folder lives inside Dropbox, syncing offloads/deletes binaries out
rem of the env and breaks the app. Set Dropbox's official ignore marker on the
rem env folders (inline powershell command, not a script file - not blocked by
rem AllSigned policy; best-effort either way).
if not exist "%HERE%envs" mkdir "%HERE%envs"
if not exist "%HERE%.micromamba" mkdir "%HERE%.micromamba"
powershell -NoProfile -Command "Set-Content -Path '%HERE%envs' -Stream com.dropbox.ignored -Value 1; Set-Content -Path '%HERE%.micromamba' -Stream com.dropbox.ignored -Value 1" >nul 2>nul

rem --- 1. Download micromamba (single .exe, no admin) -----------------------
if exist "%MM%" goto :have_mm
echo ==^> Downloading micromamba ^(win-64^)...
if not exist "%HERE%.micromamba" mkdir "%HERE%.micromamba"
where curl >nul 2>nul
if errorlevel 1 (
  echo ERROR: curl.exe not found. Requires Windows 10 1803+ or Windows 11.
  goto :fail
)
curl -L --retry 3 -o "%MM%" "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"
if errorlevel 1 (
  echo ERROR: download failed - check network / proxy.
  goto :fail
)
:have_mm
if not exist "%MM%" ( echo ERROR: micromamba.exe missing after download. & goto :fail )
echo ==^> micromamba ready.

rem --- 2. (Re)build the environment -----------------------------------------
if exist "%ENVPREFIX%" (
  echo ==^> Existing environment found - rebuilding it.
  rmdir /s /q "%ENVPREFIX%"
)
rem Locked files (app still running, OneDrive/Dropbox sync) can survive the
rem delete; retry once, then stop with a clear message instead of letting
rem micromamba abort with "Non-conda folder exists at prefix".
if exist "%ENVPREFIX%" (
  timeout /t 2 /nobreak >nul
  rmdir /s /q "%ENVPREFIX%" 2>nul
)
if exist "%ENVPREFIX%" (
  echo ERROR: could not fully remove the old environment at:
  echo   "%ENVPREFIX%"
  echo Close the PBG Grading app ^(check Task Manager for python.exe^),
  echo pause OneDrive/Dropbox syncing, delete the envs folder by hand,
  echo then run this setup again.
  goto :fail
)
echo ==^> Building environment from environment.yml ^(several minutes on first run^)...
"%MM%" create -y -p "%ENVPREFIX%" -f "%HERE%environment.yml"
if errorlevel 1 ( echo ERROR: environment build failed. & goto :fail )

rem --- 3. Verify imports ----------------------------------------------------
echo ==^> Verifying imports...
"%ENVPREFIX%\python.exe" -c "import dash, plotly, numpy, pandas, scipy, cv2, yaml, dotenv, requests, dropbox, depthai, PyQt6.QtCore; import sys; print('  python', sys.version.split()[0]); print('  depthai', depthai.__version__); print('  opencv', cv2.__version__)"
if errorlevel 1 ( echo ERROR: import verification failed. & goto :fail )

rem --- 4. Desktop shortcut (best-effort) ------------------------------------
rem Inline `powershell -Command` runs as an interactive command, NOT a script
rem file, so AllSigned policy does not block it. Non-fatal either way.
echo ==^> Creating Desktop shortcut ^(optional^)...
powershell -NoProfile -Command "$ws=New-Object -ComObject WScript.Shell; $l=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\PBG Grading.lnk'); $l.TargetPath='%HERE%Start PBG Grading.bat'; $l.WorkingDirectory='%HERE%'; $l.Description='PBG Grading station'; $l.Save()" >nul 2>nul
if errorlevel 1 echo    ^(Shortcut step skipped - just launch "Start PBG Grading.bat" in this folder.^)

echo.
echo Setup complete. Launch with the "PBG Grading" Desktop shortcut,
echo or "Start PBG Grading.bat" in this folder.
echo.
pause
exit /b 0

:fail
echo.
echo Setup did not complete. See the message above.
echo.
pause
exit /b 1
