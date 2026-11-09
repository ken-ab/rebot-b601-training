"""Durable acceptance -> 12h+ training -> full offline evaluation pipeline.

Unexpected failures stop the pipeline for the primary/diagnostic agents to
inspect. There are no silent hyperparameter changes or unbounded retries.
"""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/yujiegao/桌面/zzk/rebot-b601-training')
KIT = REPO / 'openpi_b601'
LAUNCH = '/home/yujiegao/桌面/zzk/B601_SERVER_ASSESSMENT_20261109/run_on_server.sh'
PYTHON = str(KIT / '.openpi/.venv/bin/python')
BASE = Path('/data2/yujiegao/b601-pi05/models/pi05_base/params')
RUNS = KIT / 'runs/pi05_b601_lora'


def status(stage, **extra):
    row = {'at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'stage': stage, **extra}
    temporary = ROOT / 'PIPELINE_STATUS.tmp'
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(ROOT / 'PIPELINE_STATUS.json')
    (ROOT / 'STATUS_CN.md').write_text('B601 π₀.₅ JAX LoRA 持续任务\n\n'
        f'更新时间：{row["at"]}\n\n当前阶段：`{stage}`\n\n'
        '正式长跑目标：至少 12.25 小时活动训练与评估墙钟；环境下载和初始编译不计入。\n\n'
        '精确阶段/错误见 PIPELINE_STATUS.json，退出码见 receipts/，原始输出见 logs/。'
        '正式训练后的逐步指标和计时位于仓库 runs/pi05_b601_lora/b601-final60-v1_monitor/。\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def wait_receipt(label):
    receipt = ROOT / 'receipts' / f'{label}.json'
    while True:
        if receipt.exists():
            try:
                result = json.loads(receipt.read_text())
            except json.JSONDecodeError:
                time.sleep(2)
                continue
            if result['status'] != 'running':
                if result['status'] != 'passed':
                    raise RuntimeError(f'Required stage failed: {label}; see {result["log"]}')
                return result
        time.sleep(5)


def run_stage(label, command, env=None):
    receipt = ROOT / 'receipts' / f'{label}.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status') == 'passed':
        return json.loads(receipt.read_text())
    status(label, command=command)
    subprocess.run([sys.executable, str(ROOT / 'run_stage.py'), label, *command], check=True, env=env)
    return wait_receipt(label)


def check_smoke(label, experiment):
    receipt = wait_receipt(label)
    text = Path(receipt['log']).read_text()
    rows = re.findall(r'Step\s+(\d+):[^\n]*?\bloss=([^,\s]+)', text)
    # NaN, Inf and missing/duplicate steps are all rejected before the next stage.
    import math
    assert len(rows) == 20 and sorted(int(step) for step, _ in rows) == list(range(20)), rows
    losses = [float(loss) for _, loss in rows]
    assert all(math.isfinite(value) for value in losses)
    gradients = re.findall(r'Step\s+(\d+):[^\n]*?\bgrad_norm=([^,\s]+)', text)
    assert len(gradients) == 20 and all(math.isfinite(float(value)) for _, value in gradients)
    result = {'experiment': experiment, 'steps': 20, 'losses': losses, 'first_loss': losses[0],
              'last_loss': losses[-1], 'checkpoint': str(RUNS / experiment / '19')}
    (ROOT / 'receipts' / f'{label}-acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    verify_env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                      OPENPI_DATA_HOME='/data2/yujiegao/b601-pi05/cache/openpi')
    run_stage(label + '-readback', [PYTHON, str(ROOT / 'verify_checkpoint.py'), result['checkpoint'],
              '--output', str(ROOT / 'receipts' / f'{label}-orbax-readback.json')], verify_env)
    return result


def main():
    status('waiting_for_environment_and_verified_base')
    wait_receipt('prepare-v2')
    audit_env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                    OPENPI_DATA_HOME='/data2/yujiegao/b601-pi05/cache/openpi')
    run_stage('full-data-audit-v1', [PYTHON, str(ROOT / 'audit_prepared_data.py')], audit_env)
    gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-a2df82e1-ad48-ea47-fa69-5ee02ec32b0e,GPU-8d91782a-1f41-2aa0-ef48-585ded17365d',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false')
    run_stage('jax-gpu-collective-v1', [PYTHON, str(ROOT / 'check_jax_gpu.py')], gpu_env)
    wait_receipt('base-download-v2')
    assert (BASE.parent / 'VERIFIED_DOWNLOAD.json').is_file()
    for batch in (2, 4):
        label = f'smoke-b{batch}-v1'
        experiment = f'b601-final60-smoke-fsdp2-b{batch}'
        run_stage(label, ['bash', LAUNCH, 'smoke', '--batch-size', str(batch), '--exp-name', experiment,
                          '--base-params', str(BASE)])
        check_smoke(label, experiment)

    probe_name = 'b601-final60-hook-probe'
    probe = ['bash', LAUNCH, 'train', '--batch-size', '4', '--workers', '0', '--steps', '21',
        '--lr-decay-steps', '10000', '--save-interval', '10', '--long-run-hours', '0.001',
        '--exp-name', probe_name, '--base-params', str(BASE)]
    run_stage('hook-probe-v1', probe)
    assert (RUNS / (probe_name + '_monitor') / 'LONGRUN_COMPLETE.json').exists()
    run_stage('hook-resume-v1', probe + ['--resume'])

    experiment = 'b601-final60-v1'
    long_monitor = RUNS / (experiment + '_monitor')
    command = ['bash', LAUNCH, 'train', '--exp-name', experiment, '--base-params', str(BASE),
        '--batch-size', '4', '--workers', '2', '--steps', '200000', '--lr-decay-steps', '10000',
        '--save-interval', '1000', '--long-run-hours', '12.25']
    run_stage('formal-longrun-v1', command)
    completion = json.loads((long_monitor / 'LONGRUN_COMPLETE.json').read_text())
    assert completion['active_seconds'] >= 12.25 * 3600
    best = Path(json.loads((long_monitor / 'BEST.json').read_text())['preserved_checkpoint'])
    last = Path(completion['checkpoint'])
    evaluate_env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-a2df82e1-ad48-ea47-fa69-5ee02ec32b0e',
        OPENPI_DATA_HOME='/data2/yujiegao/b601-pi05/cache/openpi',
        HF_HOME='/data2/yujiegao/b601-pi05/cache/huggingface', XLA_PYTHON_CLIENT_MEM_FRACTION='0.9',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    for name, checkpoint in [('base', BASE.parent), ('best', best), ('last', last)]:
        output = ROOT / 'evaluation' / name
        output.parent.mkdir(parents=True, exist_ok=True)
        args = [PYTHON, str(KIT / 'evaluate_b601.py'), '--checkpoint', str(checkpoint),
                '--output', str(output), '--normalization-checkpoint', str(best)]
        if name == 'base':
            args.append('--base-model')
        run_stage(f'full-eval-{name}-v1', args, evaluate_env)
    export_env = dict(evaluate_env, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    run_stage('deployment-export-v1', [PYTHON, str(ROOT / 'export_delivery.py')], export_env)
    run_stage('deployment-inference-readback-v1', [PYTHON, str(ROOT / 'B601_PI05_DEPLOYMENT/self_test.py')], evaluate_env)
    status('complete', completion=completion, best_checkpoint=str(best), last_checkpoint=str(last),
           delivery=json.loads((ROOT / 'DELIVERY_RECEIPT.json').read_text()))
    subprocess.run(['systemctl', '--user', 'stop', 'b601-gpu-sampler-20261109'], check=False)


try:
    main()
except Exception as exc:
    status('failed_needs_agent_analysis', error=repr(exc))
    raise
