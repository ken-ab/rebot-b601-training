"""Read a real Orbax parameter checkpoint back on CPU and check finite arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import flax.traverse_util
import numpy as np
from openpi.models.model import restore_params

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
started = time.monotonic()
checkpoint = args.checkpoint.resolve()
stats = checkpoint / 'assets/rebot-b601-train/norm_stats.json'
assert stats.is_file(), stats
params = restore_params(checkpoint / 'params', restore_type=np.ndarray)
flat = flax.traverse_util.flatten_dict(params, sep='/')
assert flat and any('lora' in name for name in flat)
for name, value in flat.items():
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite restored array: {name}')
receipt = {'checkpoint': str(checkpoint), 'parameter_arrays': len(flat),
    'lora_arrays': sum('lora' in name for name in flat),
    'parameter_count': sum(value.size for value in flat.values()),
    'restored_parameter_bytes': sum(value.nbytes for value in flat.values()),
    'all_parameters_finite': True, 'norm_stats_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
    'elapsed_seconds': time.monotonic() - started}
args.output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
