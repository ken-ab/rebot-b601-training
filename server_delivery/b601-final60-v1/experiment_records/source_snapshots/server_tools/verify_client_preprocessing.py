"""Check real validation observations across raw and compact wire inputs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / 'rebot-b601-training/openpi_b601'
os.environ['HF_LEROBOT_HOME'] = str(KIT / 'prepared/datasets')
sys.path.insert(0, str(ROOT / 'windows_client'))
sys.path.insert(0, str(KIT))
from b601_observation import CAMERAS, prepare_observation
from b601_data import B601Inputs, VAL_ID
from openpi.transforms import ResizeImages
from openpi_client import msgpack_numpy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def identical(first, second, digest, path=''):
    if isinstance(first, dict):
        assert set(first) == set(second), path
        for key in sorted(first):
            identical(first[key], second[key], digest, path + '/' + key)
    elif isinstance(first, str):
        assert first == second, path
        digest.update((path + first).encode())
    else:
        first, second = np.asarray(first), np.asarray(second)
        assert first.dtype == second.dtype, path
        np.testing.assert_array_equal(first, second, err_msg=path)
        digest.update((path + str(first.shape) + first.dtype.str).encode())
        digest.update(first.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    canonical = np.load(ROOT / 'evaluation/checkpoint10000/predictions.npz')
    episodes = [json.loads(line) for line in
        (KIT / 'prepared/datasets' / VAL_ID / 'meta/episodes.jsonl').read_text().splitlines()]
    offsets, position = {}, 0
    for episode in episodes:
        offsets[episode['episode_index']] = position
        position += episode['length']
    assert position == 5118 and len(episodes) == 10
    dataset = LeRobotDataset(VAL_ID, video_backend='pyav')
    digest = hashlib.sha256()
    payloads = []
    for local_episode, source_episode, frame in canonical['episode_source_frame']:
        sample = dataset[offsets[int(local_episode)] + int(frame)]
        raw = {key: np.asarray(sample[key]) for key in (*CAMERAS, 'observation.state')}
        raw['prompt'] = '拿起蓝色餐巾纸，把它放进黑色框里'
        prepared = prepare_observation(raw)
        packed = msgpack_numpy.packb(prepared)
        received = msgpack_numpy.unpackb(packed)
        before = ResizeImages(224, 224)(B601Inputs()(raw))
        after = ResizeImages(224, 224)(B601Inputs()(received))
        identical(before, after, digest)
        payloads.append({'source_episode': int(source_episode), 'frame': int(frame),
                        'original_image_bytes': sum(raw[key].nbytes for key in CAMERAS),
                        'wire_image_bytes': sum(received[key].nbytes for key in CAMERAS),
                        'packed_request_bytes': len(packed)})
    assert len(payloads) == 211 and all(row['wire_image_bytes'] == 301056 for row in payloads)
    report = {'status': 'passed', 'at': datetime.datetime.now().astimezone().isoformat(),
        'observations': len(payloads), 'validation_episodes': 10,
        'wire_roundtrip_checked': True, 'effective_inputs_elementwise_identical': True,
        'checked_fields': 'three resized images, image masks, seven-axis raw state and prompt',
        'effective_input_digest_sha256': digest.hexdigest(),
        'all_compact_images_shape': [224, 224, 3], 'all_compact_images_dtype': 'uint8',
        'first_payload': payloads[0], 'all_payloads': payloads,
        'client_source_sha256': hashlib.sha256((ROOT / 'windows_client/b601_observation.py').read_bytes()).hexdigest(),
        'upstream_image_tools_sha256': hashlib.sha256((ROOT / 'windows_client/openpi_client_src/openpi_client/image_tools.py').read_bytes()).hexdigest(),
        'scope': 'CPU proof of equal model inputs for211realvalidationobservations; no robot/WindowsWAN execution'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'all_payloads'}), flush=True)


if __name__ == '__main__':
    main()
