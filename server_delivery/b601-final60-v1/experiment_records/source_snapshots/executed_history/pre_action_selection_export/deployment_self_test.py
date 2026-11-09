"""Verify the delivered files and a real B601 observation's golden actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def verify_manifest(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or invalid manifest file: {relative}')
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'Delivered file size differs: {relative}')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Delivered file SHA256 differs: {relative}')
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-5)
    args = parser.parse_args()
    if not all(np.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error('Tolerances must be finite and nonnegative')
    root = Path(__file__).resolve().parent
    verified = verify_manifest(root)
    os.environ['OPENPI_DATA_HOME'] = str(root / 'cache/openpi')
    sys.path.insert(0, str(root / 'code'))
    from b601_policy import load_policy
    x = np.load(root / 'example_observation.npz')
    policy = load_policy(root / 'checkpoint')
    started = time.monotonic()
    result = policy.infer({'observation.images.overhead': x['overhead'],
        'observation.images.wrist': x['wrist'], 'observation.state': x['state'],
        'prompt': '拿起蓝色餐巾纸，把它放进黑色框里'}, noise=x['noise'])
    actions = np.asarray(result['actions'])
    elapsed = time.monotonic() - started
    if actions.shape != (25, 7) or not np.isfinite(actions).all():
        raise ValueError('Delivered model must produce finite 25x7 actions')
    np.testing.assert_allclose(actions, x['expected_actions'], atol=args.atol, rtol=args.rtol,
        err_msg='Delivered inference differs from the server reference; inspect before deployment')
    print(json.dumps({'manifest_files_verified': verified, 'shape': list(actions.shape),
        'all_finite': True, 'golden_matches': True, 'atol': args.atol, 'rtol': args.rtol,
        'max_abs_difference_from_server': float(np.max(np.abs(actions-x['expected_actions']))),
        'first_call_wall_seconds_including_jit': elapsed}))


if __name__ == '__main__':
    main()
