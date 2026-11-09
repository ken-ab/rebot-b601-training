"""Fixed B601 ACT data/model mapping; target-free inference is mandatory."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

ROOT = Path(__file__).resolve().parent
CACHE = Path('/data2/yujiegao/b601-act-final60/cache_v1')
PREPARED = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/prepared')
CAMERAS = ('observation.images.overhead', 'observation.images.wrist')
JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos',
          'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos')


class CachedDataset(Dataset):
    def __init__(self, split):
        self.root = CACHE / split
        self.episodes = json.loads((self.root / 'EPISODES.json').read_text())
        self.count = sum(row['length'] for row in self.episodes)
        self.maps = None

    def __len__(self):
        return self.count

    def __getstate__(self):
        return {**self.__dict__, 'maps': None}

    def open(self):
        if self.maps is None:
            self.maps = {name: np.load(self.root / (name+'.npy'), mmap_mode='r')
                         for name in ('images', 'state', 'action', 'frame_end')}
        return self.maps

    def __getitem__(self, index):
        data = self.open()
        end = int(data['frame_end'][index])
        future = np.arange(index, index+25)
        return {CAMERAS[0]: torch.from_numpy(np.array(data['images'][index, 0])),
                CAMERAS[1]: torch.from_numpy(np.array(data['images'][index, 1])),
                'observation.state': torch.from_numpy(np.array(data['state'][index])),
                'action': torch.from_numpy(np.array(data['action'][np.minimum(future, end-1)])),
                'action_is_pad': torch.from_numpy(future >= end)}


def config(pretrained=True, device='cuda'):
    return ACTConfig(input_features={
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for key in CAMERAS}},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=25, n_action_steps=25, device=device,
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None)


def train_stats():
    raw = json.loads((PREPARED / 'assets/rebot-b601-train/norm_stats.json').read_text())['norm_stats']
    stats = {'observation.state': {k: torch.tensor(raw['state'][k], dtype=torch.float32) for k in ('mean', 'std')},
             'action': {k: torch.tensor(raw['actions'][k], dtype=torch.float32) for k in ('mean', 'std')}}
    for key in CAMERAS:
        stats[key] = {'mean': torch.tensor([.485, .456, .406]).reshape(3, 1, 1),
                      'std': torch.tensor([.229, .224, .225]).reshape(3, 1, 1)}
    return stats


def make_policy(pretrained=True, device='cuda'):
    return ACTPolicy(config(pretrained=pretrained, device=device), train_stats()).to(device)


def move_batch(batch, device):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    for key in CAMERAS:
        batch[key] = batch[key].float() / 255
    return batch


@torch.inference_mode()
def predict_chunk(policy, inputs):
    if 'action' in inputs or 'action_is_pad' in inputs:
        raise ValueError('Deployment evaluation must never expose action targets to the VAE encoder')
    policy.eval()
    batch = policy.normalize_inputs(inputs)
    batch['observation.images'] = [batch[key] for key in policy.config.image_features]
    normalized, latent = policy.model(batch)
    assert latent == (None, None), 'Expected target-free zero-latent ACT prediction'
    return policy.unnormalize_outputs({'action': normalized})['action']


@torch.inference_mode()
def evaluate(policy, output=None, batch_size=8):
    dataset = CachedDataset('val')
    indices, identifiers = [], []
    for episode in dataset.episodes:
        for frame in range(0, episode['length'], 25):
            indices.append(episode['offset']+frame)
            identifiers.append((episode['local_episode'], episode['source_episode'], frame))
    assert len(indices) == 211
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size)
    predictions, targets, masks = [], [], []
    was_training = policy.training
    for batch in loader:
        truth, valid = batch['action'].numpy(), ~batch['action_is_pad'].numpy()
        inputs = {key: batch[key] for key in (*CAMERAS, 'observation.state')}
        inputs = move_batch(inputs, next(policy.parameters()).device)
        prediction = predict_chunk(policy, inputs).cpu().numpy()
        assert prediction.shape[1:] == (25, 7) and np.isfinite(prediction).all()
        predictions.append(prediction)
        targets.append(truth)
        masks.append(valid)
    policy.train(was_training)
    prediction, target, mask = np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    identifiers = np.asarray(identifiers)
    reference = np.load('/home/yujiegao/桌面/zzk/B601_LONGRUN_20261109/evaluation/checkpoint10000/predictions.npz')
    for actual, key in ((target, 'target'), (mask, 'valid_mask'), (identifiers, 'episode_source_frame')):
        np.testing.assert_array_equal(actual, reference[key])
    errors = (prediction.astype(np.float64)-target)[mask]
    assert errors.shape == (5118, 7)
    mae, rmse = np.abs(errors).mean(0), np.sqrt((errors**2).mean(0))
    std = train_stats()['action']['std'].numpy()
    episode_rows = []
    for episode in dataset.episodes:
        rows = identifiers[:, 0] == episode['local_episode']
        error = (prediction[rows].astype(np.float64)-target[rows])[mask[rows]]
        episode_rows.append({'source_episode': episode['source_episode'], 'rows': len(error),
            'per_axis_mae': np.abs(error).mean(0).tolist(), 'per_axis_rmse': np.sqrt((error**2).mean(0)).tolist()})
    metrics = {'policy': 'ACT', 'evaluated_action_rows': 5118, 'chunks': 211, 'horizon': 25,
        'target_free_zero_latent': True, 'targets_masks_source_match_pi05': True,
        'normalized_valid_l1': float(np.mean(np.abs(errors)/(std+1e-8))),
        'per_axis_mae': mae.tolist(), 'per_axis_rmse': rmse.tolist(),
        'mean_joint_1to6_mae': float(mae[:6].mean()), 'gripper_mae': float(mae[6]),
        'macro_episode_per_axis_mae': np.mean([e['per_axis_mae'] for e in episode_rows], 0).tolist(),
        'macro_episode_per_axis_rmse': np.mean([e['per_axis_rmse'] for e in episode_rows], 0).tolist(),
        'episodes': episode_rows, 'axis_names': list(JOINTS), 'units': 'absolute_adapter_input',
        'scope': 'Validationusedselection, notindependenttest orphysicalgrasp success'}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / 'predictions.npz', prediction=prediction, target=target,
                            valid_mask=mask, episode_source_frame=identifiers)
        (output / 'METRICS.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics
