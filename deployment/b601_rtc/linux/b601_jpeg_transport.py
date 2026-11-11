"""JPEG transport for B601 RGB observations, with a loopback OpenPI relay."""

from io import BytesIO

import numpy as np
from PIL import Image


CAMERA_KEYS = ('observation.images.overhead', 'observation.images.wrist')
TRANSPORT_KEY = 'b601_image_transport'
JPEG_RGB = 'jpeg-rgb-v1'
IMAGE_SHAPE = (224, 224, 3)
MAX_IMAGE_BYTES = 224 * 224 * 3
MAX_MESSAGE_BYTES = 512 * 1024


def encode_request(observation):
    request = dict(observation)
    for key in CAMERA_KEYS:
        image = observation[key]
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f'Expected a 224×224 RGB uint8 image: {key}')
        with BytesIO() as encoded:
            Image.fromarray(image).save(encoded, format='JPEG', quality=90, subsampling=0)
            request[key] = encoded.getvalue()
    request[TRANSPORT_KEY] = JPEG_RGB
    return request


def decode_jpeg(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError('Invalid JPEG size or type')
    with Image.open(BytesIO(data), formats=('JPEG',)) as image:
        if image.mode != 'RGB' or image.size != IMAGE_SHAPE[:2]:
            raise ValueError('Expected a 224×224 RGB JPEG')
        return np.array(image, dtype=np.uint8)


def run_relay(port, upstream_port):
    from http.client import HTTPConnection, HTTPException
    from http import HTTPStatus

    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    from websockets.sync.server import serve

    host = '127.0.0.1'

    def health(connection, request):
        if request.path != '/healthz':
            return None
        upstream = HTTPConnection(host, upstream_port, timeout=4)
        try:
            upstream.request('GET', '/healthz')
            if upstream.getresponse().status == HTTPStatus.OK:
                return connection.respond(HTTPStatus.OK, 'OK\n')
        except (OSError, HTTPException):
            pass
        finally:
            upstream.close()
        return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, 'OpenPI not ready\n')

    def relay(connection):
        with connect(f'ws://{host}:{upstream_port}', compression=None, proxy=None,
                     open_timeout=5, max_size=MAX_MESSAGE_BYTES) as upstream:
            metadata = msgpack_numpy.unpackb(upstream.recv(timeout=5))
            connection.send(msgpack_numpy.packb({**metadata, TRANSPORT_KEY: JPEG_RGB}))
            for packet in connection:
                try:
                    if not isinstance(packet, bytes):
                        raise ValueError('Expected a binary request')
                    observation = msgpack_numpy.unpackb(packet)
                    if not isinstance(observation, dict):
                        raise ValueError('Expected an observation mapping')
                    transport = observation.pop(TRANSPORT_KEY, None)
                    if transport == JPEG_RGB:
                        for key in CAMERA_KEYS:
                            observation[key] = decode_jpeg(observation[key])
                        packet = msgpack_numpy.packb(observation)
                    elif transport is not None:
                        raise ValueError('Unknown image transport')
                except (ValueError, TypeError, KeyError, OSError):
                    connection.close(1007, 'Invalid B601 observation or JPEG')
                    return
                upstream.send(packet)
                connection.send(upstream.recv(timeout=300))

    with serve(relay, host, port, process_request=health, compression=None,
               max_size=MAX_MESSAGE_BYTES) as server:
        print(f'B601 JPEG relay listening on {host}:{port}, OpenPI port {upstream_port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--upstream-port', type=int, default=8000)
    args = parser.parse_args()
    run_relay(args.port, args.upstream_port)
