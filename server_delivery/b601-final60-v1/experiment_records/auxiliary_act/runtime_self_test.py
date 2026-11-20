"""Portable ACT package integrity/real-observation inference check."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from act_runtime import B601ACT
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for relative, expected in manifest.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalidmanifestpath:{relative}')
        if path.stat().st_size != expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError(f'Integritycheckfailed:{relative}')
    torch.set_num_threads(min(8, torch.get_num_threads()))
    policy = B601ACT(root, args.device)
    sample = np.load(root/'example_observation.npz')
    observation = {'observation.images.overhead': sample['overhead'],
        'observation.images.wrist': sample['wrist'], 'observation.state': sample['state']}
    seconds = []
    for _ in range(11):
        started = time.monotonic()
        actions = policy.infer(observation)['actions']
        seconds.append(time.monotonic()-started)
        np.testing.assert_allclose(actions, sample['expected_actions'], atol=1e-3, rtol=1e-4)
    print(json.dumps({'status': 'passed', 'system': platform.system(), 'device': str(policy.device),
        'shape': list(actions.shape), 'all_finite': bool(np.isfinite(actions).all()),
        'manifest_files_verified': len(manifest), 'golden_atol': 1e-3, 'golden_rtol': 1e-4,
        'max_abs_difference_from_training_gpu': float(np.abs(actions-sample['expected_actions']).max()),
        'first_call_seconds': seconds[0], 'warm_median_seconds': float(np.median(seconds[1:])),
        'warm_p95_seconds': float(np.percentile(seconds[1:],95)),
        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/1048576 if policy.device.type=='cuda' else None,
        'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved()/1048576 if policy.device.type=='cuda' else None,
        'torch_version': torch.__version__, 'metadata': policy.metadata,
        'robot_commands_sent': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
