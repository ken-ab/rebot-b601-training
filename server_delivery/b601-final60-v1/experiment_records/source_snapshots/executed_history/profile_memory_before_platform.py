"""Measure one real policy's residency with JAX preallocation disabled."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
bundle = Path('/data2/yujiegao/b601-pi05/validation/deployment_rehearsal10000_v2')
os.environ['OPENPI_DATA_HOME'] = str(bundle / 'cache/openpi')
sys.path.insert(0, str(bundle / 'code'))
from b601_policy import load_policy
import jax

def residency():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    return [int(row.split(',')[1].strip()) for row in rows if int(row.split(',')[0].strip()) == os.getpid()]

policy = load_policy(bundle / 'checkpoint')
sample = np.load(bundle / 'example_observation.npz')
observation = {'observation.images.overhead': sample['overhead'], 'observation.images.wrist': sample['wrist'],
    'observation.state': sample['state'], 'prompt': '拿起蓝色餐巾纸，把它放进黑色框里'}
memory, times = [residency()], []
for _ in range(10):
    start = time.monotonic()
    actions = np.asarray(policy.infer(observation, noise=sample['noise'])['actions'])
    times.append(time.monotonic()-start)
    np.testing.assert_allclose(actions, sample['expected_actions'], atol=1e-4, rtol=1e-5)
    memory.append(residency())
stats = jax.devices()[0].memory_stats()
report = {'status': 'passed', 'checkpoint': str(bundle / 'checkpoint'), 'device': str(jax.devices()[0]),
    'preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
    'memory_fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
    'jax_memory_stats_bytes': {k: int(v) for k,v in stats.items()}, 'process_residency_mib': memory,
    'max_observed_process_residency_mib': max(v for row in memory for v in row),
    'host_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    'seconds_per_call': times, 'user_total_vram_mib': 8188,
    'scope': 'LinuxA4500 measured process residency/pool stats; not RTX4070Laptop/WSL verification or exact minimumVRAM'}
(args.output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
