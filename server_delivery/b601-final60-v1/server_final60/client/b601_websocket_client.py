"""Direct B601 RPC with bounded waits and no reuse after an ambiguous response."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'openpi_client_src'))
import websockets.sync.client
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class B601WebsocketClientPolicy(WebsocketClientPolicy):
    """Preserve the OpenPI wire API without inheriting operating-system proxies.

    A timed-out inference may still produce a late response. This instance
    closes that connection and never retries the observation automatically.
    """

    def __init__(self, host='127.0.0.1', port=8000, api_key=None, timeout=60.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be finite and positive')
        self._response_timeout = float(timeout)
        self._open_timeout = min(10.0, self._response_timeout)
        super().__init__(host, port, api_key)

    def _wait_for_server(self):
        headers = {'Authorization': f'Api-Key {self._api_key}'} if self._api_key else None
        connection = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None, additional_headers=headers,
            proxy=None, open_timeout=self._open_timeout, close_timeout=5.0,
        )
        try:
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=self._open_timeout))
        except Exception:
            connection.close()
            raise
        return connection, metadata

    def infer(self, obs):
        if self._ws is None:
            raise RuntimeError('Connection discarded; create a new client and acquire a fresh observation')
        try:
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv(timeout=self._response_timeout)
            if isinstance(response, str):
                raise RuntimeError('Inference server error: ' + response)
            return msgpack_numpy.unpackb(response)
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self):
        connection, self._ws = self._ws, None
        if connection is not None:
            connection.close()
