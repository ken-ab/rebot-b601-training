"""Early portable-layout readback only; does not claim final 12h completion."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--version', choices=('v1', 'v2'), default='v1')
args = parser.parse_args()
KIT = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601')
CHECKPOINT = Path('/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1/10000')
BUNDLE = Path('/data2/yujiegao/b601-pi05/validation/deployment_rehearsal10000_' + args.version)
BUNDLE.mkdir(parents=True, exist_ok=False)
for name in ('params', 'assets'):
    shutil.copytree(CHECKPOINT / name, BUNDLE / 'checkpoint' / name, copy_function=os.link)
(BUNDLE / 'code').mkdir()
for name in ('b601_data.py', 'train_pi05.py', 'b601_policy.py', 'serve_b601.py', 'evaluate_b601.py'):
    shutil.copy2(KIT / name, BUNDLE / 'code' / name)
cache = BUNDLE / 'cache/openpi/big_vision'
cache.mkdir(parents=True)
shutil.copy2('/data2/yujiegao/b601-pi05/cache/openpi/big_vision/paligemma_tokenizer.model', cache)
evaluation = BUNDLE / 'evaluation/candidates/checkpoint10000'
reference = ROOT / ('evaluation' if args.version == 'v2' else 'early_evaluation') / 'checkpoint10000'
shutil.copytree(reference, evaluation)
os.environ['HF_LEROBOT_HOME'] = str(KIT / 'prepared/datasets')
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
sample = LeRobotDataset('rebot-b601-val', video_backend='pyav')[0]
assert sample['task'] == '拿起蓝色餐巾纸，把它放进黑色框里'
golden = np.load(evaluation / 'predictions.npz')['prediction'][0]
noise = np.random.default_rng(20261109).standard_normal((25, 32), dtype=np.float32)
np.savez_compressed(BUNDLE / 'example_observation.npz', overhead=np.asarray(sample['observation.images.overhead']),
    wrist=np.asarray(sample['observation.images.wrist']), state=np.asarray(sample['observation.state']),
    noise=noise, expected_actions=golden)
shutil.copy2(ROOT / 'deployment_self_test.py', BUNDLE / 'self_test.py')
selection = {'delivered_models': {'primary': {'label': 'checkpoint10000', 'optimizer_step': 10001,
    'relative_checkpoint': 'checkpoint',
    'relative_predictions': 'evaluation/candidates/checkpoint10000/predictions.npz'}},
    'qualification': 'Interim portable-layout rehearsal; final training is still running'}
(BUNDLE / 'DEPLOYMENT_SELECTION.json').write_text(json.dumps(selection, indent=2) + '\n')
manifest = {}
for path in BUNDLE.rglob('*'):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
        manifest[str(path.relative_to(BUNDLE))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
(BUNDLE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Rehearsal bundle ready; code directory intentionally has no prepared/ dataset or repository checkout', flush=True)
subprocess.run([sys.executable, str(BUNDLE / 'self_test.py')], check=True)
