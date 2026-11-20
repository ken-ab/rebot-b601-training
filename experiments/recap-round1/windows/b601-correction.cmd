@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
if "%~1"=="" goto collection
if /i "%~1"=="--evaluation" goto evaluation
if /i "%~1"=="--describe" if "%~2"=="" goto describe
set "REBOT_PYTHON=%~dp0..\.venv-lerobot\Scripts\python.exe"
if not exist "%REBOT_PYTHON%" (
    echo Project Python not found: "%REBOT_PYTHON%" 1>&2
    exit /b 1
)
"%REBOT_PYTHON%" -X utf8 -B -u "%~dp0b601_correction.py" --execute --port 8766 %*
exit /b %ERRORLEVEL%

:collection
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0b601_start_live.ps1"
exit /b %ERRORLEVEL%

:evaluation
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0b601_start_live.ps1" -Evaluation
exit /b %ERRORLEVEL%

:describe
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0b601_start_live.ps1" -OfflineCheck
exit /b %ERRORLEVEL%
