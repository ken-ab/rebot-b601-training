"""Exercise direct transport and timeout invalidation on an own CPU test server."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'windows_client'))
from b601_websocket_client import B601WebsocketClientPolicy
from openpi_client import msgpack_numpy
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    received = []

    def delayed(connection):
        connection.send(msgpack_numpy.packb({'test_server': True}))
        received.append(msgpack_numpy.unpackb(connection.recv(timeout=2)))
        time.sleep(0.6)
        try:
            connection.send(msgpack_numpy.packb({'late_response': True}))
        except ConnectionClosed:
            pass

    with serve(delayed, '127.0.0.1', 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = B601WebsocketClientPolicy('127.0.0.1', server.socket.getsockname()[1], timeout=0.15)
        try:
            assert client.get_server_metadata()['test_server']
            try:
                client.infer({'trial': 1})
                raise AssertionError('Delayed response should time out')
            except TimeoutError:
                pass
            assert client._ws is None
            try:
                client.infer({'trial': 2})
                raise AssertionError('Discarded connection must reject a second request')
            except RuntimeError as error:
                assert 'discarded' in str(error)
        finally:
            client.close()
            server.shutdown()
            worker.join(timeout=3)
    assert received == [{'trial': 1}]
    assert not any(module in sys.modules for module in ('jax', 'torch', 'lerobot', 'transformers'))
    report = {'status': 'passed', 'at': datetime.datetime.now().astimezone().isoformat(),
        'own_local_test_server': True, 'model_or_robot_loaded': False,
        'direct_connection_metadata_received': True, 'delayed_inference_timed_out': True,
        'connection_discarded_after_timeout': True, 'second_request_rejected_locally': True,
        'requests_seen_by_server': len(received), 'automatic_inference_retries': 0,
        'without_jax_torch_lerobot_transformers': True,
        'source_sha256': hashlib.sha256((ROOT / 'windows_client/b601_websocket_client.py').read_bytes()).hexdigest(),
        'scope': 'LocalLinuxCPUtransport acceptance, not Windows target or robot execution'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
