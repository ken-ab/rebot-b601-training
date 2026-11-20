"""Query a B601 policy with a recorded example; does not control a robot."""
import argparse
import http.client
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'openpi_client_src'))
from b601_observation import CAMERAS, prepare_observation
from b601_websocket_client import B601WebsocketClientPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--example', type=Path,
        default=Path(__file__).resolve().parent.parent / 'example_observation.npz')
    parser.add_argument('--raw-images', action='store_true', help='Send original recorded images for comparison')
    parser.add_argument('--timeout', type=float, default=60.0, help='Response timeout, allowing first-call JIT')
    args = parser.parse_args()
    connection = http.client.HTTPConnection(args.host, args.port, timeout=10)
    try:
        connection.request('GET', '/healthz')
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Inference service health endpoint returned{response.status}')
        response.read()
    finally:
        connection.close()
    sample = np.load(args.example)
    client = B601WebsocketClientPolicy(args.host, args.port, timeout=args.timeout)
    try:
        metadata = client.get_server_metadata()
        expected = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos',
                    'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos']
        if (metadata.get('action_space') != 'absolute_adapter_input' or
                metadata.get('fps') != 25 or metadata.get('action_names') != expected):
            raise ValueError('Server metadata differs from the B601 training interface')
        started = time.monotonic()
        observation = {'observation.images.overhead': sample['overhead'],
            'observation.images.wrist': sample['wrist'], 'observation.state': sample['state'],
            'prompt': '拿起蓝色餐巾纸，把它放进黑色框里'}
        original_bytes = sum(observation[key].nbytes for key in CAMERAS)
        if not args.raw_images:
            observation = prepare_observation(observation)
        preprocessing_seconds = time.monotonic()-started
        result = client.infer(observation)
        actions = np.asarray(result['actions'])
        elapsed = time.monotonic() - started
        if actions.shape != (25, 7) or not np.isfinite(actions).all():
            raise ValueError('Expected finite25x7 B601 actions')
        print(json.dumps({'metadata': metadata, 'shape': list(actions.shape), 'all_finite': True,
            'roundtrip_seconds': elapsed, 'server_timing': result.get('server_timing'),
            'client_preprocessing_seconds': preprocessing_seconds,
            'client_preprocessed_rgb224_uint8': not args.raw_images,
            'original_image_bytes': original_bytes,
            'wire_image_bytes': sum(observation[key].nbytes for key in CAMERAS),
            'units': 'absolute adapter-input action units', 'robot_commands_sent': False}, ensure_ascii=False))
    finally:
        client.close()


if __name__ == '__main__':
    main()
