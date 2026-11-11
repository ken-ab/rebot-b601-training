@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "REBOT_PYTHON=%~dp0..\.venv-lerobot\Scripts\python.exe"
if not exist "%REBOT_PYTHON%" (
    echo Project Python not found: "%REBOT_PYTHON%" 1>&2
    exit /b 1
)
"%REBOT_PYTHON%" -X utf8 -B -u "%~dp0b601_policy_episode.py" --execute --port 8766 %*
exit /b %ERRORLEVEL%
