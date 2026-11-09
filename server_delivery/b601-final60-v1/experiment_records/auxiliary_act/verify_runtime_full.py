"""Independently test the standalone runtime on all cached held-out observations."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(args.bundle))
from act_runtime import B601ACT
import torch
torch.set_num_threads(8)
policy = B601ACT(args.bundle, args.device)
data = Path('/data2/yujiegao/b601-act-final60/cache_v1/val')
images, states = [np.load(data/(name+'.npy'), mmap_mode='r') for name in ('images', 'state')]
episodes = json.loads((data/'EPISODES.json').read_text())
reference = np.load(args.bundle/'evaluation/predictions.npz')
offsets = {row['local_episode']:row['offset'] for row in episodes}
indices = [offsets[int(ep)]+int(frame) for ep,_,frame in reference['episode_source_frame']]
result = []
started = time.monotonic()
for first in range(0,len(indices),8):
    rows = indices[first:first+8]
    result.append(policy.predict_batch(np.asarray(images[rows]), np.asarray(states[rows])))
prediction = np.concatenate(result)
assert prediction.shape==(211,25,7) and np.isfinite(prediction).all()
np.testing.assert_allclose(prediction, reference['prediction'], atol=1e-3, rtol=1e-4)
errors = (prediction.astype(np.float64)-reference['target'])[reference['valid_mask']]
mae, rmse = np.abs(errors).mean(0), np.sqrt((errors**2).mean(0))
metrics = json.loads((args.bundle/'evaluation/METRICS.json').read_text())
np.testing.assert_allclose(mae, metrics['per_axis_mae'], atol=1e-3, rtol=1e-4)
np.testing.assert_allclose(rmse, metrics['per_axis_rmse'], atol=1e-3, rtol=1e-4)
assert 'jax' not in sys.modules and 'lerobot' not in sys.modules and 'transformers' not in sys.modules
report = {'status': 'passed', 'device': args.device, 'chunks': len(indices), 'valid_action_rows': len(errors),
    'max_abs_prediction_difference_from_training_gpu': float(np.abs(prediction-reference['prediction']).max()),
    'per_axis_mae': mae.tolist(), 'per_axis_rmse': rmse.tolist(),
    'mean_joint_1to6_mae': float(mae[:6].mean()), 'gripper_mae': float(mae[6]),
    'elapsed_seconds': time.monotonic()-started, 'without_jax_lerobot_transformers_imports': True,
    'target_free_zero_latent': True, 'scope': 'Standalonepackage math/readback agreement; notphysicalrobot success'}
(args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(args.output/'predictions.npz', prediction=prediction)
print(json.dumps(report), flush=True)
