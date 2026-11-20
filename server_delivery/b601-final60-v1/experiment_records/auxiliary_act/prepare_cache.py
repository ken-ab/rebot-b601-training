"""Read the original50/10 split into a CPU-only RGB224 ACT cache."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PREPARED = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/prepared')
STORE = Path('/data2/yujiegao/b601-act-final60')
PENDING = STORE / 'cache_v1.pending'
FINAL = STORE / 'cache_v1'
assert not PENDING.exists() and not FINAL.exists()
PENDING.mkdir(parents=True)
source_manifest = json.loads((PREPARED / 'manifest.json').read_text())
report = {'at': datetime.datetime.now().astimezone().isoformat(),
    'scope': 'Independent ACT arm; originalpi05 training/data unchanged', 'fps': 25, 'action_horizon': 25,
    'resize': 'RGB640x480 -> PIL bilinear224x168 -> blackletterbox224x224,CHWuint8',
    'image_normalization': 'Fixed ImageNet mean/std; notdatasetmetastats',
    'state_action_normalization_source': 'prepared train-only norm_stats mean/std',
    'source_manifest_sha256': hashlib.sha256((PREPARED / 'manifest.json').read_bytes()).hexdigest(),
    'train_norm_stats_sha256': hashlib.sha256((PREPARED / 'assets/rebot-b601-train/norm_stats.json').read_bytes()).hexdigest(),
    'splits': {}}

for split in ('train', 'val'):
    dataset = PREPARED / 'datasets' / ('rebot-b601-' + split)
    episodes = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines()]
    count = sum(episode['length'] for episode in episodes)
    assert count == source_manifest[split]['frames']
    directory = PENDING / split
    directory.mkdir()
    images = np.lib.format.open_memmap(directory / 'images.npy', mode='w+', dtype=np.uint8,
                                       shape=(count, 2, 3, 224, 224))
    states = np.zeros((count, 7), dtype=np.float32)
    actions = np.zeros_like(states)
    frame_end = np.zeros(count, dtype=np.int64)
    metadata, tasks = [], []
    offset = 0
    for episode in episodes:
        local, length = episode['episode_index'], episode['length']
        table = pq.read_table(dataset / f'data/chunk-000/episode_{local:06d}.parquet')
        assert table.num_rows == length
        np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(length))
        states[offset:offset+length] = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        actions[offset:offset+length] = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        frame_end[offset:offset+length] = offset + length
        metadata.append({'local_episode': local, 'source_episode': source_manifest[split]['source_episode_indices'][local],
                         'offset': offset, 'length': length})
        for camera, key in enumerate(('observation.images.overhead', 'observation.images.wrist')):
            tasks.append((dataset / f'videos/chunk-000/{key}/episode_{local:06d}.mp4', offset, length, camera))
        offset += length
    assert offset == count and np.isfinite(states).all() and np.isfinite(actions).all()

    def decode(task):
        path, first, length, camera = task
        decoded = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            for frame in container.decode(stream):
                if decoded >= length:
                    raise ValueError(f'Unexpected extra video frames:{path}')
                rgb = frame.to_ndarray(format='rgb24')
                assert rgb.shape == (480, 640, 3)
                resized = np.asarray(Image.fromarray(rgb).resize((224, 168), Image.Resampling.BILINEAR))
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                image[28:196] = resized
                images[first+decoded, camera] = image.transpose(2, 0, 1)
                decoded += 1
        assert decoded == length, (path, decoded, length)
        return decoded

    with ThreadPoolExecutor(max_workers=6) as workers:
        decoded = sum(workers.map(decode, tasks))
    assert decoded == 2 * count
    images.flush()
    del images
    for name, value in [('state', states), ('action', actions), ('frame_end', frame_end)]:
        np.save(directory / (name+'.npy'), value)
    (directory / 'EPISODES.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report['splits'][split] = {'episodes': len(episodes), 'frames': count, 'decoded_camera_frames': decoded,
                               'source_episodes': source_manifest[split]['source_episode_indices']}
    print(json.dumps({'cache_split_complete': split, **report['splits'][split]}), flush=True)
(PENDING / 'CACHE_ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
PENDING.rename(FINAL)
(ROOT / 'CACHE_PATH.txt').write_text(str(FINAL) + '\n')
print(json.dumps({'cache': str(FINAL), 'acceptance': report}), flush=True)
