"""Check all episode tables and decode first/middle/last observations."""
import dataclasses
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

KIT = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601')
OUTPUT = Path(__file__).resolve().parent
os.environ['HF_LEROBOT_HOME'] = str(KIT / 'prepared/datasets')
sys.path.insert(0, str(KIT))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from b601_data import ACTION_HORIZON, JOINTS, OVERHEAD, TRAIN_ID, VAL_ID, WRIST, rgb_uint8
from b601_policy import make_config
from lerobot.common.datasets import lerobot_dataset
from openpi.training import data_loader

config = make_config()
base_data = config.data.create(config.assets_dirs, config.model)
report = {'training_normalization_only': True, 'splits': {}}
pictures = []
manifest = json.loads((KIT / 'prepared/manifest.json').read_text())
with patch.object(lerobot_dataset, 'get_safe_default_codec', return_value='pyav'):
    for split, dataset_id in [('train', TRAIN_ID), ('val', VAL_ID)]:
        data_config = dataclasses.replace(base_data, repo_id=dataset_id)
        raw = data_loader.create_torch_dataset(data_config, ACTION_HORIZON, config.model)
        dataset = data_loader.transform_dataset(raw, data_config)
        directory = KIT / 'prepared/datasets' / dataset_id
        episodes = [json.loads(line) for line in (directory / 'meta/episodes.jsonl').read_text().splitlines()]
        offset, reads = 0, 0
        state_values, action_values = [], []
        for episode in episodes:
            number = episode['episode_index']
            path = directory / f'data/chunk-{number//1000:03d}/episode_{number:06d}.parquet'
            table = pq.read_table(path)
            states = np.asarray(table['observation.state'].to_pylist())
            actions = np.asarray(table['action'].to_pylist())
            assert states.shape == actions.shape == (episode['length'], 7)
            assert np.isfinite(states).all() and np.isfinite(actions).all()
            state_values.append(states)
            action_values.append(actions)
            for frame in sorted({0, episode['length']//2, episode['length']-1}):
                sample = dataset[offset + frame]
                assert sample['actions'].shape == (25, 32) and sample['state'].shape == (32,)
                assert np.isfinite(sample['actions']).all() and np.isfinite(sample['state']).all()
                assert sample['image_mask'] == {'base_0_rgb': True, 'left_wrist_0_rgb': True, 'right_wrist_0_rgb': False}
                assert sample['tokenized_prompt'].size > 0
                assert all(image.shape == (224, 224, 3) for image in sample['image'].values())
                assert np.std(sample['image']['base_0_rgb']) > 0 and np.std(sample['image']['left_wrist_0_rgb']) > 0
                if number == 0 and frame == 0:
                    original = raw[offset + frame]
                    for camera in (OVERHEAD, WRIST):
                        image = Image.fromarray(rgb_uint8(original[camera])).resize((480, 360))
                        panel = Image.new('RGB', (480, 390), 'white')
                        panel.paste(image, (0, 30))
                        ImageDraw.Draw(panel).text((8, 8), split + ' ' + camera, fill='black')
                        pictures.append(panel)
                reads += 1
            offset += episode['length']
        state_values, action_values = np.concatenate(state_values), np.concatenate(action_values)
        assert offset == len(raw) == manifest[split]['frames']
        report['splits'][split] = {'episodes': len(episodes), 'frames': offset, 'decoded_observations': reads,
            'state_min': state_values.min(axis=0).tolist(), 'state_max': state_values.max(axis=0).tolist(),
            'action_min': action_values.min(axis=0).tolist(), 'action_max': action_values.max(axis=0).tolist(),
            'source_episode_indices': manifest[split]['source_episode_indices']}
stats = json.loads((KIT / 'prepared/assets' / TRAIN_ID / 'norm_stats.json').read_text())['norm_stats']
report['training_quantile_ranges'] = {}
for key in ('state', 'actions'):
    for values in stats[key].values():
        assert np.asarray(values).shape == (7,) and np.isfinite(values).all()
    widths = np.asarray(stats[key]['q99']) - np.asarray(stats[key]['q01'])
    assert (widths >= 0).all()
    report['training_quantile_ranges'][key] = widths.tolist()
canvas = Image.new('RGB', (960, 780), 'white')
for index, picture in enumerate(pictures):
    canvas.paste(picture, ((index % 2) * 480, (index // 2) * 390))
canvas.save(OUTPUT / 'DATA_CAMERA_CONTACT_SHEET.jpg')
(OUTPUT / 'DATA_ACCEPTANCE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False), flush=True)
