"""Evaluate actual sampled action chunks on all held-out B601 episodes."""
import argparse
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from b601_data import ACTION_HORIZON, JOINTS, OVERHEAD, VAL_ID, WRIST
from b601_policy import load_policy
from train_pi05 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--normalization-checkpoint', type=Path)
    parser.add_argument('--base-model', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=25)
    parser.add_argument('--denoise-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--max-chunks', type=int, help='Explicit limited smoke evaluation; not a full evaluation')
    args = parser.parse_args()
    if args.stride < 1 or args.denoise_steps < 1:
        parser.error('stride and denoise steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    os.environ['HF_LEROBOT_HOME'] = str(ROOT / 'prepared/datasets')
    from lerobot.common.datasets import lerobot_dataset
    with patch.object(lerobot_dataset, 'get_safe_default_codec', return_value='pyav'):
        dataset = lerobot_dataset.LeRobotDataset(VAL_ID, video_backend='pyav',
            delta_timestamps={'action': [t / 25 for t in range(ACTION_HORIZON)]})
    manifest = json.loads((ROOT / 'prepared/manifest.json').read_text())
    episodes = [json.loads(line) for line in
                (ROOT / 'prepared/datasets' / VAL_ID / 'meta/episodes.jsonl').read_text().splitlines()]
    policy = load_policy(args.checkpoint, normalization_checkpoint=args.normalization_checkpoint,
                         base_model=args.base_model, denoise_steps=args.denoise_steps)
    import jax
    import jaxlib
    rng = np.random.default_rng(args.seed)
    predictions, targets, masks, identifiers = [], [], [], []
    episode_results = []
    offset, chunk_count = 0, 0
    started = time.monotonic()
    for episode in episodes:
        length = episode['length']
        episode_errors = []
        for frame in range(0, length, args.stride):
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break
            sample = dataset[offset + frame]
            observation = {OVERHEAD: np.asarray(sample[OVERHEAD]), WRIST: np.asarray(sample[WRIST]),
                           'observation.state': np.asarray(sample['observation.state']), 'prompt': sample['task']}
            noise = rng.standard_normal((ACTION_HORIZON, 32), dtype=np.float32)
            result = policy.infer(observation, noise=noise)
            predicted = np.asarray(result['actions'], dtype=np.float32)
            truth = np.asarray(sample['action'], dtype=np.float32)
            valid = min(ACTION_HORIZON, length - frame)
            if predicted.shape != (ACTION_HORIZON, len(JOINTS)) or not np.isfinite(predicted).all():
                raise ValueError('Invalid inference output')
            error = predicted[:valid] - truth[:valid]
            episode_errors.append(error)
            predictions.append(predicted)
            targets.append(truth)
            masks.append(np.arange(ACTION_HORIZON) < valid)
            identifiers.append((episode['episode_index'], manifest['val']['source_episode_indices'][episode['episode_index']], frame))
            chunk_count += 1
        if episode_errors:
            errors = np.concatenate(episode_errors)
            episode_results.append({'local_episode': episode['episode_index'],
                'source_episode': manifest['val']['source_episode_indices'][episode['episode_index']],
                'evaluated_action_rows': len(errors), 'per_axis_mae': np.abs(errors).mean(axis=0).tolist(),
                'per_axis_rmse': np.sqrt(np.square(errors).mean(axis=0)).tolist()})
            print(f"Episode {episode['episode_index']}: {len(errors)} action rows", flush=True)
        offset += length
        if args.max_chunks is not None and chunk_count >= args.max_chunks:
            break
    prediction, target, valid_mask = np.stack(predictions), np.stack(targets), np.stack(masks)
    errors = (prediction - target)[valid_mask]
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt(np.square(errors).mean(axis=0))
    metrics = {'checkpoint': str(args.checkpoint.resolve()), 'base_model': args.base_model,
        'split': 'fixed validation split; also used for checkpoint selection', 'source_episodes': manifest['val']['source_episode_indices'],
        'full_protocol': args.max_chunks is None and args.stride == ACTION_HORIZON and len(errors) == manifest['val']['frames'],
        'stride': args.stride, 'action_horizon': ACTION_HORIZON,
        'denoise_steps': args.denoise_steps, 'seed': args.seed, 'chunks': chunk_count,
        'inference_xla_flags': os.environ['XLA_FLAGS'], 'jax_version': jax.__version__,
        'jaxlib_version': jaxlib.__version__, 'inference_devices': [str(device) for device in jax.devices()],
        'evaluated_action_rows': len(errors), 'axis_names': list(JOINTS), 'units': 'original absolute adapter-input action units',
        'per_axis_mae': mae.tolist(), 'per_axis_rmse': rmse.tolist(),
        'mean_joint_1to6_mae': float(mae[:6].mean()), 'gripper_mae': float(mae[6]),
        'macro_episode_per_axis_mae': np.mean([row['per_axis_mae'] for row in episode_results], axis=0).tolist(),
        'macro_episode_per_axis_rmse': np.mean([row['per_axis_rmse'] for row in episode_results], axis=0).tolist(),
        'episodes': episode_results, 'elapsed_seconds': time.monotonic() - started,
        'qualification': 'offline demonstration action agreement; not closed-loop robotic grasp success'}
    if args.max_chunks is None and args.stride == ACTION_HORIZON:
        assert len(episode_results) == 10 and len(errors) == manifest['val']['frames']
    np.savez_compressed(args.output / 'predictions.npz', prediction=prediction, target=target,
                        valid_mask=valid_mask, episode_source_frame=np.asarray(identifiers))
    (args.output / 'METRICS.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
