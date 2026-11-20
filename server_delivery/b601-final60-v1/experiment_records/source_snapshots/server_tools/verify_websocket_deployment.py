"""Loopback serving acceptance with real observations; never controls a robot."""
import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np


def observation(sample):
    return {'observation.images.overhead': sample['overhead'],
        'observation.images.wrist': sample['wrist'], 'observation.state': sample['state'],
        'prompt': '拿起蓝色餐巾纸，把它放进黑色框里'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direct-worker', action='store_true')
    parser.add_argument('--scope', choices=('interim', 'final'), default='interim')
    parser.add_argument('--client-dir', type=Path, help='Override clientdirectory for an interim rehearsal')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    # Mirror serve.sh: bundled tokenizer assets must be used by both processes.
    os.environ['OPENPI_DATA_HOME'] = str(bundle / 'cache/openpi')
    sample = np.load(bundle / 'example_observation.npz')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.direct_worker:
        sys.path.insert(0, str(bundle / 'code'))
        from b601_policy import load_policy
        policy = load_policy(bundle / 'checkpoint')
        actions = [np.asarray(policy.infer(observation(sample))['actions']) for _ in range(3)]
        np.savez_compressed(args.output / 'direct.npz', actions=np.stack(actions))
        return

    request_observation = observation(sample)
    client_directory = args.client_dir or bundle / 'client'
    compact_inputs = (client_directory / 'b601_observation.py').is_file()
    preprocessing_started = time.monotonic()
    if compact_inputs:
        sys.path.insert(0, str(client_directory))
        from b601_observation import prepare_observation
        request_observation = prepare_observation(request_observation)
    preprocessing_seconds = time.monotonic() - preprocessing_started
    if args.scope == 'final':
        assert compact_inputs, 'Final package must contain its compact Windowsclient'
        assert json.loads((bundle / 'TRAINING_SUMMARY.json').read_text())['training']['active_seconds'] >= 44100

    def deadline(*_):
        raise TimeoutError('Bounded loopback integration acceptance exceeded240seconds')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(240)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server_log = (args.output / 'server.log').open('w')
    command = [sys.executable, str(bundle / 'code/serve_b601.py'), '--checkpoint',
        str(bundle / 'checkpoint'), '--host', '127.0.0.1', '--port', str(port)]
    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
    client_smoke_passed = False
    try:
        ready_deadline = time.monotonic() + 100
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Own testserver exited{server.returncode}; inspectserver.log')
            try:
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                connection.request('GET', '/healthz')
                response = connection.getresponse()
                ready = response.status == 200 and response.read() == b'OK\n'
                connection.close()
                if ready:
                    break
            except OSError:
                pass
            if time.monotonic() >= ready_deadline:
                raise TimeoutError('Own modelserver did notbecomehealthywithin100seconds')
            time.sleep(0.5)
        if compact_inputs:
            from b601_websocket_client import B601WebsocketClientPolicy
            client = B601WebsocketClientPolicy('127.0.0.1', port)
        else:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy('127.0.0.1', port)
        metadata = client.get_server_metadata()
        assert metadata['fps'] == 25 and metadata['action_space'] == 'absolute_adapter_input'
        assert len(metadata['action_names']) == 7
        network, wall_seconds, server_ms = [], [], []
        for _ in range(3):
            started = time.monotonic()
            result = client.infer(request_observation)
            network.append(np.asarray(result['actions']))
            wall_seconds.append(time.monotonic() - started)
            server_ms.append(result['server_timing']['infer_ms'])
        if compact_inputs:
            client.close()
        else:
            client._ws.close()
        if compact_inputs:
            # Exercise the actual delivered CLI as well as the low-level RPC.
            with (args.output / 'client_smoke.log').open('w') as client_log:
                subprocess.run([sys.executable, str(client_directory / 'client_smoke.py'),
                    '--host', '127.0.0.1', '--port', str(port), '--example',
                    str(bundle / 'example_observation.npz')], check=True, timeout=30,
                    stdout=client_log, stderr=subprocess.STDOUT)
            cli_result = json.loads((args.output / 'client_smoke.log').read_text().splitlines()[-1])
            assert cli_result['all_finite'] and cli_result['shape'] == [25, 7]
            assert cli_result['client_preprocessed_rgb224_uint8'] and cli_result['wire_image_bytes'] == 301056
            client_smoke_passed = True
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--bundle', str(bundle),
                    '--output', str(args.output), '--direct-worker'], check=True, timeout=100)
    network = np.stack(network)
    direct = np.load(args.output / 'direct.npz')['actions']
    assert network.shape == direct.shape == (3, 25, 7) and np.isfinite(network).all()
    np.testing.assert_allclose(network, direct, atol=1e-4, rtol=1e-5)
    np.savez_compressed(args.output / 'network.npz', actions=network)
    report = {'status': 'passed', 'bundle': str(bundle), 'model_metadata': metadata,
        'shape': list(network.shape), 'all_finite': True, 'max_abs_network_vs_direct': float(np.abs(network-direct).max()),
        'roundtrip_wall_seconds': wall_seconds, 'synchronous_server_infer_ms': server_ms,
        'first_call_includes_jit': True, 'rng': 'Three matching default-policy PRNG key0 calls in separate freshserver/directprocesses',
        'host': '127.0.0.1', 'port': port, 'own_test_server_stopped': server.poll() is not None,
        'client_preprocessed_rgb224_uint8': compact_inputs,
        'provided_client_smoke_cli_passed': client_smoke_passed,
        'client_execution_scope': 'ServerLinux only; not Windows target execution',
        'client_preprocessing_seconds': preprocessing_seconds,
        'wire_image_bytes': sum(request_observation[key].nbytes for key in
                                ('observation.images.overhead', 'observation.images.wrist')),
        'robot_connected_or_actuated': False, 'scope': args.scope + ' production-protocol acceptance; no physicalrobot inferenceexecution'}
    (args.output / 'REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
