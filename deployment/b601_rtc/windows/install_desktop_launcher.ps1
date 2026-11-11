param([Parameter(Mandatory=$true)][string]$ProjectRoot)
$ErrorActionPreference = 'Stop'
$entry = Join-Path ([IO.Path]::GetFullPath($ProjectRoot)) 'b601_start_live.ps1'
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) { throw "Missing entry: $entry" }
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($entry))
$template = @'
@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title B601 RTC Pick and Place
rem Keep CMD source ASCII; PowerShell decodes the Unicode project path.
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -Command "& { param([switch]$CheckOnly,[double]$Seconds,[string]$SshHost,[int]$SshPort,[string]$SshUser) $entry=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('__ENTRY__')); if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) { throw ('B601 launcher not found: ' + $entry) }; & $entry @PSBoundParameters; exit $LASTEXITCODE }" %*
set "B601_EXIT_CODE=%ERRORLEVEL%"
if "%~1"=="" (
    echo.
    echo Run finished. Press any key to close this window.
    pause >nul
)
exit /b %B601_EXIT_CODE%
'@
$desktop = [Environment]::GetFolderPath('Desktop')
$target = Join-Path $desktop 'B601真机抓放.cmd'
if (Test-Path -LiteralPath $target) {
    $backup = Join-Path $ProjectRoot 'backups\desktop-launcher'
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    Copy-Item -LiteralPath $target -Destination (Join-Path $backup ((Get-Date -Format 'yyyyMMdd-HHmmss') + '.cmd'))
}
$text = $template.Replace('__ENTRY__', $encoded).Replace("`r`n", "`n").Replace("`n", "`r`n")
[IO.File]::WriteAllText($target, $text + "`r`n", [Text.Encoding]::ASCII)
Write-Host $target
