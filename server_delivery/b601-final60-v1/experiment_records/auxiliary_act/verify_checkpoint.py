"""Restore the complete real ACT smoke checkpoint on CPU and compare inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from act_common import CAMERAS, CachedDataset, make_policy, move_batch, predict_chunk

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8)
policy = make_policy(pretrained=False, device='cpu')
weights = args.checkpoint / 'policy/model.safetensors'
safetensors.torch.load_model(policy, str(weights), strict=True, device='cpu')
state = policy.state_dict()
assert all(torch.isfinite(value).all().item() for value in state.values())
sample = CachedDataset('val')[0]
inputs = {key: sample[key].unsqueeze(0) for key in (*CAMERAS, 'observation.state')}
prediction = predict_chunk(policy, move_batch(inputs, 'cpu')).numpy()[0]
reference = np.load(args.checkpoint / 'evaluation/predictions.npz')['prediction'][0]
assert prediction.shape == (25, 7) and np.isfinite(prediction).all()
np.testing.assert_allclose(prediction, reference, atol=1e-4, rtol=1e-5)
report = {'status': 'passed', 'checkpoint': str(args.checkpoint), 'device': 'cpu',
    'all_parameters_and_buffers_finite': True, 'state_arrays': len(state),
    'max_abs_difference_from_gpu': float(np.max(np.abs(prediction-reference))),
    'real_observation_shape': [25, 7], 'target_free_zero_latent': True,
    'scope': 'Actualsmokecheckpoint restore/readback; nottrainedtasksuccess'}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
