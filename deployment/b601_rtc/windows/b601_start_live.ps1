param(
    [switch]$CheckOnly,
    [double]$Seconds = 180,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9.:-]*$')]
    [string]$SshHost = '223.109.239.32',
    [ValidateRange(1, 65535)]
    [int]$SshPort = 15128,
    [ValidatePattern('^[A-Za-z0-9_][A-Za-z0-9_.-]*$')]
    [string]$SshUser = 'root'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$projectRoot = $PSScriptRoot
$python = [IO.Path]::GetFullPath((Join-Path $projectRoot '..\.venv-lerobot\Scripts\python.exe'))
$localPort = 8766

function Test-InferenceService {
    $request = [Net.HttpWebRequest]::Create("http://127.0.0.1:$localPort/healthz")
    $request.Proxy = $null
    $request.Timeout = 5000
    $response = $null
    try {
        $response = $request.GetResponse()
        return [int]$response.StatusCode -eq 200
    } catch {
        return $false
    } finally {
        if ($null -ne $response) { $response.Close() }
    }
}

try {
    if ([double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds) -or $Seconds -le 0) {
        throw 'Seconds must be finite and greater than zero.'
    }
    $required = @(
        $python,
        (Join-Path $projectRoot 'b601_restore_start.py'),
        (Join-Path $projectRoot 'b601_run_with_frames.py'),
        (Join-Path $projectRoot 'b601_policy_episode.py'),
        (Join-Path $projectRoot 'b601_rtc_ready.py'),
        (Join-Path $projectRoot 'b601_action_timeline.py'),
        (Join-Path $projectRoot 'b601_task_completion.py'),
        (Join-Path $projectRoot 'b601-task-reference.jpg'),
        (Join-Path $projectRoot 'b601-start-pose.json'),
        (Join-Path $projectRoot 'record-smoke.yaml')
    )
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required project file is missing: $path"
        }
    }
    Set-Location -LiteralPath $projectRoot
    $ready = Test-InferenceService

    if ($CheckOnly) {
        Write-Host 'Check-only mode: no cameras or motor connection will be opened.'
        Write-Host "Project: $projectRoot"
        Write-Host "Python: $python"
        Write-Host "Episode duration: $Seconds seconds"
        Write-Host "SSH: ${SshUser}@${SshHost}:$SshPort"
        & $python -X utf8 -B (Join-Path $projectRoot 'b601_policy_episode.py') --describe
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if (-not $ready) {
            Write-Host "Inference service 127.0.0.1:$localPort is offline. A normal launch will open SSH."
            exit 1
        }
        & $python -X utf8 -B (Join-Path $projectRoot 'b601_rtc_ready.py') --port $localPort
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "Project files and inference service 127.0.0.1:$localPort are ready."
        exit 0
    }

    if (-not $ready) {
        $ssh = Get-Command ssh.exe -CommandType Application -ErrorAction Stop
        $listeners = @([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
            Where-Object { $_.Port -eq $localPort })
        $sshArguments = @('-T', '-p', "$SshPort", '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3')
        if ($listeners.Count -eq 0) {
            $sshArguments += @('-L', "127.0.0.1:${localPort}:127.0.0.1:8001")
        }
        # The remote launcher already exists on the rented server's system disk.
        # Keep the SSH session alive after starting the two inference services.
        $remoteCommand = 'set -e; if ! curl -fsS --max-time 5 http://127.0.0.1:8002/healthz >/dev/null || ! curl -fsS --max-time 5 http://127.0.0.1:8001/healthz >/dev/null; then bash /root/b601-deploy/start_inference.sh; fi; echo B601-RTC-inference-tunnel-ready; exec sleep infinity'
        $sshArguments += @("${SshUser}@${SshHost}", ('"' + $remoteCommand + '"'))
        Write-Host '正在打开 SSH 窗口；如提示密码，请在该窗口输入当前服务器密码，并保持窗口打开。'
        $tunnel = Start-Process -FilePath $ssh.Source -ArgumentList $sshArguments -PassThru -WindowStyle Normal
        while (-not (Test-InferenceService)) {
            if ($tunnel.HasExited) {
                throw 'SSH 在模型服务就绪前退出，请检查服务器地址、密码及云实例开机状态。'
            }
            Start-Sleep -Milliseconds 500
        }
    }

    $duration = $Seconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    & $python -X utf8 -B (Join-Path $projectRoot 'b601_rtc_ready.py') --port $localPort
    if ($LASTEXITCODE -ne 0) { throw 'RTC 服务未就绪，请查看推理服务器日志。' }
    Write-Host "即将恢复保存的起始姿态，随后执行 RTC 真机抓放；检测到成功即回零，最长 $duration 秒。"
    Write-Host '请先接好 PCAN 和两路相机、关闭预览窗口；从臂处于低位停放姿态，夹爪空、纸巾在右侧，你在切电开关旁。'
    Write-Host 'Q：回零退出；Esc：立即失能。请等停止清理完成后再关闭此窗口。'
    & $python -X utf8 -B -u (Join-Path $projectRoot 'b601_restore_start.py') --execute
    if ($LASTEXITCODE -ne 0) {
        throw '起点恢复未完成，本轮抓放未启动；详细原因见 inference-evals 内的日志。'
    }
    & $python -X utf8 -B -u (Join-Path $projectRoot 'b601_run_with_frames.py') --execute --port $localPort --seconds $duration
    $result = $LASTEXITCODE
    Write-Host "本轮运行结束，退出码：$result。动作日志和图像：$(Join-Path $projectRoot 'inference-evals')"
    exit $result
} catch {
    Write-Host ("启动中止：" + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
