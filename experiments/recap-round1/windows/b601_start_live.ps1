param(
    [switch]$OfflineCheck,
    [switch]$CheckOnly,
    [switch]$ModelOnly,
    [switch]$Evaluation,
    [double]$Seconds,
    [ValidateSet('aligned', 'rtc')][string]$Mode,
    [double]$GripperDeltaDeg,
    [string]$ConfigPath,
    [string]$TrialLabel = '',
    [string]$SshHost,
    [int]$SshPort,
    [string]$SshUser
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
    } catch { return $false }
    finally { if ($null -ne $response) { $response.Close() } }
}

try {
    if (-not $ConfigPath) { $ConfigPath = Join-Path $projectRoot 'b601_execution.json' }
    $ConfigPath = [IO.Path]::GetFullPath($ConfigPath)
    $runner = Join-Path $projectRoot $(if ($ModelOnly) { 'b601_run_with_frames.py' } else { 'b601_correction.py' })
    $diagnostic = Join-Path $projectRoot $(if ($ModelOnly) { 'b601_policy_episode.py' } else { 'b601_correction.py' })
    $required = @($python, $ConfigPath, $runner)
    foreach ($name in @('b601_restore_start.py', 'b601_run_with_frames.py',
        'b601_policy_episode.py', 'b601_rtc_ready.py', 'b601_action_timeline.py',
        'b601_execution_config.py', 'b601_gripper_control.py', 'b601_stop.py',
        'b601_task_completion.py', 'b601-task-reference.jpg', 'b601-start-pose.json',
        'record-smoke.yaml')) { $required += Join-Path $projectRoot $name }
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file is missing: $path" }
    }
    Set-Location -LiteralPath $projectRoot
    $settings = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $PSBoundParameters.ContainsKey('Seconds')) { $Seconds = $settings.seconds }
    if (-not $Mode) { $Mode = $settings.policy_mode }
    if (-not $PSBoundParameters.ContainsKey('GripperDeltaDeg')) {
        $GripperDeltaDeg = $settings.gripper_delta_motor_deg_per_step
    }
    if (-not $SshHost) { $SshHost = $settings.inference_ssh.host }
    if (-not $PSBoundParameters.ContainsKey('SshPort')) { $SshPort = $settings.inference_ssh.port }
    if (-not $SshUser) { $SshUser = $settings.inference_ssh.user }
    if ([double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds) -or $Seconds -le 0) {
        throw 'Seconds must be finite and greater than zero.'
    }
    $duration = $Seconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    $gripperDelta = $GripperDeltaDeg.ToString([Globalization.CultureInfo]::InvariantCulture)
    $commonArgs = @('--config', $ConfigPath, '--policy-mode', $Mode,
                    '--gripper-delta-deg', $gripperDelta)
    if (-not $ModelOnly) {
        $batchRoot = Join-Path $projectRoot 'corrections\recap-100-v1'
        if ($Evaluation) {
            $outbox = if (Test-Path -LiteralPath 'D:\') { 'D:\B601-evaluation-outbox' } else { Join-Path ([IO.Path]::GetTempPath()) 'B601-evaluation-outbox' }
            $batchRoot = Join-Path $outbox 'recap-round1-step-47000'
            $TrialLabel = 'recap-round1-step-47000'
            $helper = Join-Path $env:LOCALAPPDATA 'Temp\b601-recap-training-run\access.ps1'
            if (-not $env:B601_EVAL_SSH_HELPER -and $SshHost -eq 'scyun.cloud' -and $SshPort -eq 30046 -and (Test-Path -LiteralPath $helper -PathType Leaf)) {
                $env:B601_EVAL_SSH_HELPER = $helper
            }
        }
        $commonArgs += @('--hand-guide', '--output', $batchRoot)
    }
    Write-Host "Project: $projectRoot"
    Write-Host "Mode: $Mode; duration: $duration s; J7 delta: $gripperDelta motor-deg/step"

    # This branch returns before network probing, cameras, CAN or motor enable.
    if ($OfflineCheck -or $CheckOnly) {
        & $python -X utf8 -B $diagnostic --describe --seconds $duration @commonArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if ($OfflineCheck) {
            Write-Host 'Local configuration and imports passed. No inference service or robot was opened.'
            exit 0
        }
    }
    $ready = Test-InferenceService
    if ($CheckOnly) {
        if (-not $ready) { throw "Inference service 127.0.0.1:$localPort is offline." }
        & $python -X utf8 -B (Join-Path $projectRoot 'b601_rtc_ready.py') --port $localPort --config $ConfigPath --policy-mode $Mode
        exit $LASTEXITCODE
    }
    if (-not $ready) {
        if (-not $SshHost) {
            throw '尚未配置推理卡。请在 b601_execution.json 的 inference_ssh 中填新推理卡地址；QDS 只用于训练。此时尚未连接或使能机械臂。'
        }
        if ($SshHost -notmatch '^[A-Za-z0-9][A-Za-z0-9.:-]*$' -or
            $SshUser -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]*$' -or
            $SshPort -lt 1 -or $SshPort -gt 65535) { throw 'Invalid inference SSH address, user or port.' }
        $ssh = Get-Command ssh.exe -CommandType Application -ErrorAction Stop
        $listeners = @([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
            Where-Object { $_.Port -eq $localPort })
        $sshArguments = @('-T', '-p', "$SshPort", '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3')
        if ($listeners.Count -eq 0) { $sshArguments += @('-L', "127.0.0.1:${localPort}:127.0.0.1:8001") }
        $remoteCommand = 'set -e; if ! curl -fsS --max-time 5 http://127.0.0.1:8002/healthz >/dev/null || ! curl -fsS --max-time 5 http://127.0.0.1:8001/healthz >/dev/null; then bash /root/b601-deploy/start_inference.sh; fi; echo B601-inference-tunnel-ready; exec sleep infinity'
        $sshArguments += @("${SshUser}@${SshHost}", ('"' + $remoteCommand + '"'))
        Write-Host '正在打开交互式 SSH 窗口；如提示密码，请输入推理卡密码，并保持窗口打开。'
        $tunnel = Start-Process -FilePath $ssh.Source -ArgumentList $sshArguments -PassThru -WindowStyle Normal
        while (-not (Test-InferenceService)) {
            if ($tunnel.HasExited) { throw 'SSH 在模型服务就绪前退出，请检查推理卡地址及实例状态。' }
            Start-Sleep -Milliseconds 500
        }
    }

    & $python -X utf8 -B (Join-Path $projectRoot 'b601_rtc_ready.py') --port $localPort --config $ConfigPath --policy-mode $Mode
    if ($LASTEXITCODE -ne 0) { throw '模型服务或协议未就绪，请查看推理卡日志。' }
    Write-Host "即将恢复起始姿态，随后执行 $Mode 真机抓放；检测到成功即回零，模型运行限时 $duration 秒。"
    Write-Host '请接好 PCAN 和两路相机并关闭预览；从臂处于低位、夹爪空、纸巾在右侧，你在切电开关旁。'
    if ($ModelOnly) {
        Write-Host 'H 标记人工辅助。'
    } else {
        Write-Host 'H 切入手拖。看到「可手拖」后可拖黑臂、从夹指外侧捏合或撑开。G 保持夹爪开度；再按 G 可调。松手离开后按 H 交回模型。'
    }
    Write-Host '目视确认放好、夹爪撤离后按 S 完成并回零。Q 退出回零，Esc 立即失能。'
    $episodeArgs = @('--execute', '--port', "$localPort", '--seconds', $duration) + $commonArgs
    if ($TrialLabel) {
        $labelFlag = if ($ModelOnly) { '--trial-label' } else { '--policy-label' }
        $episodeArgs += @($labelFlag, $TrialLabel)
    }
    if ($ModelOnly) {
        & $python -X utf8 -B -u (Join-Path $projectRoot 'b601_restore_start.py') --execute --config $ConfigPath --gripper-delta-deg $gripperDelta
        if ($LASTEXITCODE -ne 0) { throw '起点恢复未完成，详情见 inference-evals；本条抓放未启动。' }
        & $python -X utf8 -B -u $runner @episodeArgs
        exit $LASTEXITCODE
    }
    if ($Evaluation) { $episodeArgs += '--evaluation' }
    & $python -X utf8 -B -u (Join-Path $projectRoot 'b601_recap_batch.py') @episodeArgs
    exit $LASTEXITCODE
} catch {
    Write-Host ("启动中止：" + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
