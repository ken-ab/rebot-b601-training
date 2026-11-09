"""Compare fixed-noise inference across identical native/portable inputs."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--layout', choices=['native', 'portable'], required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
logging.basicConfig(level=logging.INFO)
kit = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601')
bundle = Path('/data2/yujiegao/b601-pi05/validation/deployment_rehearsal10000_v1')
sys.path.insert(0, str(kit if args.layout == 'native' else bundle / 'code'))
os.environ['OPENPI_DATA_HOME'] = str(bundle / 'cache/openpi')
from b601_policy import load_policy
import jax

checkpoint = (Path('/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1/10000')
              if args.layout == 'native' else bundle / 'checkpoint')
policy = load_policy(checkpoint)
x = np.load(bundle / 'example_observation.npz')
observation = {'observation.images.overhead': x['overhead'], 'observation.images.wrist': x['wrist'],
               'observation.state': x['state'], 'prompt': '拿起蓝色餐巾纸，把它放进黑色框里'}
transformed = policy._input_transform(jax.tree.map(lambda v: v, observation))
leaves, _ = jax.tree_util.tree_flatten_with_path(transformed)
fingerprints, arrays = {}, {}
for key, value in leaves:
    key = jax.tree_util.keystr(key)
    value = np.asarray(value)
    arrays[key] = value
    fingerprints[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
np.savez_compressed(args.output / 'transformed_inputs.npz', **arrays)
predictions, seconds = [], []
for iteration in range(3):
    started = time.monotonic()
    actions = np.asarray(policy.infer(observation, noise=x['noise'])['actions'])
    assert actions.shape == (25, 7) and np.isfinite(actions).all()
    seconds.append(time.monotonic() - started)
    predictions.append(actions)
np.savez_compressed(args.output / 'predictions.npz', prediction=np.stack(predictions), expected=x['expected_actions'])
report = {'layout': args.layout, 'checkpoint': str(checkpoint), 'inputs': fingerprints,
    'repeat_max_abs_difference': [float(np.max(np.abs(p-predictions[0]))) for p in predictions],
    'golden_max_abs_difference': [float(np.max(np.abs(p-x['expected_actions']))) for p in predictions],
    'seconds': seconds, 'device': str(jax.devices()),
    'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_ENABLE_X64','XLA_PYTHON_CLIENT_MEM_FRACTION')}}
(args.output / 'REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'inputs'}), flush=True)
