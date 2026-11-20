"""Independent GPU1 inference acceptance; does not modify the formal trainer."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601')
MONITOR = Path('/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1_monitor')
BEST = Path(json.loads((MONITOR / 'BEST.json').read_text())['preserved_checkpoint'])
BASE = Path('/data2/yujiegao/b601-pi05/models/pi05_base')
PYTHON = str(KIT / '.openpi/.venv/bin/python')
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-3630f58a-b268-bb35-863e-7565f2e026b9',
    OPENPI_DATA_HOME='/data2/yujiegao/b601-pi05/cache/openpi',
    HF_HOME='/data2/yujiegao/b601-pi05/cache/huggingface',
    XLA_PYTHON_CLIENT_MEM_FRACTION='0.9', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1')
output_root = ROOT / 'early_evaluation'
output_root.mkdir(exist_ok=True)
(output_root / 'PROTOCOL.json').write_text(json.dumps({'selected_checkpoint': str(BEST),
    'physical_gpu': 1, 'purpose': 'early deployment and actual-action validation; not another training arm',
    'qualification': 'validation also used for model selection; independent evaluation time is not added to formal active time'}, indent=2) + '\n')

def run(label, checkpoint, output, *, base=False, maximum=None):
    command = [PYTHON, str(KIT / 'evaluate_b601.py'), '--checkpoint', str(checkpoint),
               '--normalization-checkpoint', str(BEST), '--output', str(output)]
    if base:
        command.append('--base-model')
    if maximum is not None:
        command.extend(['--max-chunks', str(maximum)])
    subprocess.run([sys.executable, str(ROOT / 'run_stage.py'), label, *command], env=environment, check=True)
    arrays = np.load(output / 'predictions.npz')
    assert arrays['prediction'].shape[-2:] == (25, 7) and np.isfinite(arrays['prediction']).all()
    if maximum is None:
        assert arrays['prediction'].shape == (211, 25, 7)
        assert arrays['valid_mask'].shape == (211, 25) and arrays['valid_mask'].sum() == 5118
        assert arrays['episode_source_frame'].shape == (211, 3)
    return json.loads((output / 'METRICS.json').read_text())

run('early-inference-probe-v1', BEST, output_root / 'probe_best5000', maximum=2)
best = run('early-eval-best5000-v1', BEST, output_root / 'best5000')
base = run('early-eval-base-v1', BASE, output_root / 'base', base=True)
summary = {'best_checkpoint': str(BEST), 'base': base, 'best5000': best,
           'qualification': 'offline demonstration agreement, not physical robot success'}
(output_root / 'COMPARISON.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'base_mean_joint_mae': base['mean_joint_1to6_mae'], 'best_mean_joint_mae': best['mean_joint_1to6_mae'],
                  'base_gripper_mae': base['gripper_mae'], 'best_gripper_mae': best['gripper_mae']}), flush=True)
