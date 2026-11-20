"""Real ACT20step acceptance/readback before a separate fixed training arm."""
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LOGROOT = ROOT.parent / 'B601_LONGRUN_20261109'
PYTHON = '/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/.venv/bin/python'
STORE = Path('/data2/yujiegao/b601-act-final60/runs')


def status(stage, **extra):
    row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage, **extra}
    temporary = ROOT / 'PIPELINE_STATUS.tmp'
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(ROOT / 'PIPELINE_STATUS.json')
    print(json.dumps(row), flush=True)


def receipt(label):
    file = LOGROOT / 'receipts' / (label+'.json')
    while True:
        if file.exists():
            row = json.loads(file.read_text())
            if row['status'] != 'running':
                if row['status'] != 'passed':
                    raise RuntimeError(f'Stage{label} failed: {row["log"]}')
                return row
        time.sleep(2)


def run(label, command, env=None):
    subprocess.run([sys.executable, str(LOGROOT/'run_stage.py'), label, *command], check=True, env=env)
    return receipt(label)


try:
    status('waiting_real_smoke_v2')
    smoke = receipt('act-smoke-v2')
    records = []
    for line in Path(smoke['log']).read_text().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if 'loss' in row and 'step' in row:
            records.append(row)
    assert [row['step'] for row in records] == list(range(1, 21))
    assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm_before_clipping']) for row in records)
    checkpoint = STORE/'act-b601-smoke-v2/step_000020'
    metrics = json.loads((checkpoint/'evaluation/METRICS.json').read_text())
    assert metrics['target_free_zero_latent'] and metrics['evaluated_action_rows'] == 5118
    acceptance = {'status': 'passed', 'steps': 20, 'losses': [row['loss'] for row in records],
        'grad_norms': [row['grad_norm_before_clipping'] for row in records],
        'checkpoint': str(checkpoint), 'validation_is_deployment_prediction': True,
        'scope': 'Engineeringacceptance only; formalarmnotyettrained'}
    (ROOT/'SMOKE_ACCEPTANCE.json').write_text(json.dumps(acceptance, indent=2)+'\n')
    status('cpu_smoke_readback')
    run('act-smoke-readback-v1', [PYTHON, str(ROOT/'verify_checkpoint.py'), '--checkpoint', str(checkpoint),
        '--output', str(ROOT/'SMOKE_READBACK.json')], dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    status('formal_act_training_v1', expected_steps=20000, global_batch_size=8, physical_gpu=1)
    run('act-formal-training-v1', [PYTHON, str(ROOT/'train_act.py'), '--name', 'act-b601-final60-v1',
                                '--steps', '20000', '--batch-size', '8'])
    complete = json.loads((STORE/'act-b601-final60-v1/COMPLETE.json').read_text())
    status('training_complete_needs_deployment_export', completion=complete,
           best=json.loads((STORE/'act-b601-final60-v1/BEST.json').read_text()))
except Exception as exc:
    status('failed_needs_agent_analysis', error=repr(exc))
    raise
