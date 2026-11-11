@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title B601 RTC Pick and Place
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0b601_start_live.ps1" %*
set "B601_EXIT_CODE=%ERRORLEVEL%"
if "%~1"=="" (
    echo.
    echo Run finished. Press any key to close this window.
    pause >nul
)
exit /b %B601_EXIT_CODE%
