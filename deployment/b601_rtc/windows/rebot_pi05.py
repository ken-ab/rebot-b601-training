"""Windows endpoint for remote B601 π₀.₅ inference and read-only previews."""

import argparse
import ctypes
import http.client
import json
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from b601_jpeg_transport import JPEG_RGB, TRANSPORT_KEY, encode_request


HERE = Path(__file__).resolve().parent
CLIENT = (HERE / 'handoff/rebot-b601-training-server/server_delivery/b601-final60-v1'
          / 'server_final60/client')
DATASET = HERE / 'training/blue-tissue-center-20261109-final-60'
HOST = '127.0.0.1'
PORT = 8765
PROMPT = '拿起蓝色餐巾纸，把它放进黑色框里'

if not (CLIENT / 'b601_websocket_client.py').is_file():
    raise SystemExit(f'找不到配套的 GitHub 推理客户端：{CLIENT}')
sys.path.insert(0, str(CLIENT))
from action_guard import JOINTS
from b601_observation import CAMERAS, prepare_observation
from b601_websocket_client import B601WebsocketClientPolicy
from openpi_client import msgpack_numpy

METADATA = {'robot_type': 'seeed_b601_rs_follower', 'fps': 25,
            'action_names': list(JOINTS), 'action_space': 'absolute_adapter_input'}
ACTION_SHAPE = (25, len(JOINTS))


def check_health(port):
    connection = http.client.HTTPConnection(HOST, port, timeout=5)
    try:
        connection.request('GET', '/healthz')
        status = connection.getresponse().status
        if status != 200:
            raise RuntimeError(f'推理服务 /healthz 返回 HTTP {status}。')
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f'推理服务 {HOST}:{port} 未就绪。请检查 SSH 隧道和远端模型服务。{exc}') from exc
    finally:
        connection.close()


def check_local(port, devices):
    from rebot_capture import CAMERA_USB_IDS, directshow_camera_indices
    from rebot_follow import PCAN_DIR

    ssh = shutil.which('ssh.exe')
    if not ssh:
        raise RuntimeError('找不到 Windows OpenSSH 客户端 ssh.exe。')
    dll = PCAN_DIR / 'PCANBasic.dll'
    ctypes.WinDLL(str(dll), winmode=0)
    print(f'本机软件通过：OpenSSH（{ssh}）、推理客户端、PCAN-Basic x64。')
    indices = directshow_camera_indices()
    for name in CAMERA_USB_IDS:
        print(f'{name}: 已识别为相机 {indices[name]}' if name in indices else
              f'{name}: 暂未识别，请接回录制时使用的 USB 相机和接口。')
    if devices:
        observation = live_observation()
        for key in CAMERAS:
            height, width, channels = observation[key].shape
            print(f'{key}: {width}×{height}，{channels} 通道')
        print('七轴原始电机反馈（度）：')
        for joint, angle in zip(JOINTS, observation['observation.state'], strict=True):
            print(f'{joint}: {angle:.2f}')
    if port is not None:
        check_health(port)
        print(f'推理服务 {HOST}:{port}/healthz 正常。')


def recorded_observation(root, episode, frame):
    os.environ['HF_HUB_OFFLINE'] = '1'
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not (root / 'meta/info.json').is_file():
        raise FileNotFoundError(f'找不到本地 LeRobot 数据集：{root}')
    dataset = LeRobotDataset(f'local/{root.name}', root=root, video_backend='pyav')
    if not 0 <= episode < len(dataset.meta.episodes):
        raise ValueError(f'episode {episode} 不在本数据集内。')
    fields = dataset.meta.features['observation.state']
    if tuple(fields['names']) != JOINTS:
        raise ValueError('录像的七轴顺序与 B601 模型配置不一致。')
    row = dataset.meta.episodes[episode]
    start, end = int(row['dataset_from_index']), int(row['dataset_to_index'])
    if not 0 <= frame < end - start:
        raise ValueError(f'第 {episode} 条录像没有第 {frame} 帧，有效范围 0 到 {end - start - 1}。')
    sample = dataset[start + frame]
    return {**{key: sample[key] for key in CAMERAS},
            'observation.state': sample['observation.state'], 'prompt': sample.get('task', PROMPT)}


def live_observation():
    from b601_preflight import feedback_only, read_follower
    from rebot_capture import camera_frames, connect_cameras

    with connect_cameras() as cameras, feedback_only() as follower:
        angles = read_follower(follower)
        images, _ = camera_frames(cameras)
        return {**{key: images[key.rsplit('.', 1)[1]] for key in CAMERAS},
                'observation.state': [angles[joint.removesuffix('.pos')] for joint in JOINTS],
                'prompt': PROMPT}


def query(observation, port, count, timeout, jpeg=False):
    prep_at = perf_counter()
    request = prepare_observation(observation)
    if jpeg:
        request = encode_request(request)
    prep_ms = (perf_counter() - prep_at) * 1000
    client = B601WebsocketClientPolicy(HOST, port, timeout=timeout)
    try:
        metadata = client.get_server_metadata()
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in METADATA.items()):
            raise ValueError('服务端机械臂类型、25 Hz、动作空间或关节顺序与 B601 配置不符。')
        if jpeg and metadata.get(TRANSPORT_KEY) != JPEG_RGB:
            raise ValueError('这个端口没有启用 B601 JPEG 解码，请连接 JPEG 转发端口。')
        elapsed_ms = []
        for _ in range(count):
            started = perf_counter()
            response = client.infer(request)
            elapsed_ms.append((perf_counter() - started) * 1000)
            if not isinstance(response, dict) or 'actions' not in response:
                raise ValueError('服务端响应缺少 actions。')
            actions = np.asarray(response['actions'])
            if (actions.shape != ACTION_SHAPE or actions.dtype.kind not in 'fiu'
                    or not np.isfinite(actions).all()):
                raise ValueError('服务端动作必须是 25×7 个有限实数。')
    finally:
        client.close()
    report = {
        'action_shape': list(actions.shape), 'action_names': list(JOINTS),
        'action_space': METADATA['action_space'], 'first_action': actions[0].tolist(),
        'image_transport': JPEG_RGB if jpeg else 'raw-rgb',
        'image_payload_bytes': sum(len(request[key]) if jpeg else request[key].nbytes for key in CAMERAS),
        'preprocess_ms': round(prep_ms, 1), 'first_roundtrip_ms': round(elapsed_ms[0], 1),
        'robot_actions_sent': False,
    }
    if len(elapsed_ms) > 1:
        p50, p95 = np.percentile(elapsed_ms[1:], [50, 95])
        report.update(later_requests=len(elapsed_ms) - 1, later_p50_ms=round(float(p50), 1),
                      later_p95_ms=round(float(p95), 1), max_roundtrip_ms=round(max(elapsed_ms), 1))
    if 'server_timing' in response:
        report['last_server_timing'] = response['server_timing']
    return report


def self_test(observation):
    from http import HTTPStatus
    from threading import Thread

    from websockets.sync.server import serve

    expected = prepare_observation(observation)
    received = []
    actions = np.arange(np.prod(ACTION_SHAPE), dtype=np.float32).reshape(ACTION_SHAPE)

    def handler(connection):
        connection.send(msgpack_numpy.packb(METADATA))
        for raw in connection:
            received.append(msgpack_numpy.unpackb(raw))
            connection.send(msgpack_numpy.packb({'actions': actions}))

    def health(connection, request):
        if request.path == '/healthz':
            return connection.respond(HTTPStatus.OK, 'OK\n')
        return None

    with serve(handler, HOST, 0, process_request=health, compression=None) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            port = server.socket.getsockname()[1]
            check_health(port)
            report = query(observation, port, count=2, timeout=5)
        finally:
            server.shutdown()
            worker.join(timeout=5)
    if len(received) != 2 or report['first_action'] != actions[0].tolist():
        raise RuntimeError('协议自检未收到预期的请求与回包。')
    for request in received:
        for key in CAMERAS:
            image = request[key]
            if (image.shape != (224, 224, 3) or image.dtype != np.uint8
                    or not np.array_equal(image, expected[key])):
                raise RuntimeError(f'协议自检图像不一致：{key}')
        if (not np.array_equal(request['observation.state'], observation['observation.state'])
                or request['prompt'] != expected['prompt']):
            raise RuntimeError('协议自检七轴读数或任务文字不一致。')
    report['source'] = '本机模拟服务，使用真实录制帧验证协议；未运行模型'
    return report


def bounded_int(lower, upper):
    def parse(value):
        number = int(value)
        if not lower <= number <= upper:
            raise argparse.ArgumentTypeError(f'必须在 {lower} 到 {upper} 之间')
        return number
    return parse


def main():
    parser = argparse.ArgumentParser(description='B601 π₀.₅ Windows 推理检查')
    commands = parser.add_subparsers(dest='command')
    for name, help_text in (
        ('check', '检查本机软件、USB 相机及可选的隧道健康状态'),
        ('self-test', '用本地录像测试协议，无需 GPU 或 SSH'),
        ('replay', '发送本地录像中的一帧，查看模型返回'),
        ('live-preview', '读取两路现场图像和从臂反馈，预览模型返回'),
    ):
        command = commands.add_parser(name, help=help_text)
        if name in ('check', 'replay', 'live-preview'):
            command.add_argument('--port', type=bounded_int(1, 65535),
                                 default=None if name == 'check' else PORT)
        if name == 'check':
            command.add_argument('--devices', action='store_true', help='实际取帧并读取失能从臂的反馈')
        if name in ('self-test', 'replay'):
            command.add_argument('--dataset', type=Path, default=DATASET)
            command.add_argument('--episode', type=int, default=17)
            command.add_argument('--frame', type=int, default=0)
        if name in ('replay', 'live-preview'):
            command.add_argument('--timeout', type=float, default=60)
            command.add_argument('--count', type=bounded_int(1, 100), default=1)
            command.add_argument('--jpeg', action='store_true', help='发送 JPEG 图像，需要云机 JPEG 解码端口')
    args = parser.parse_args()

    if args.command in (None, 'check'):
        check_local(getattr(args, 'port', None), getattr(args, 'devices', False))
        return
    if args.command != 'self-test':
        check_health(args.port)
    if args.command == 'live-preview':
        report = query(live_observation(), args.port, args.count, args.timeout, jpeg=args.jpeg)
        report['source'] = '现场两路相机和从臂只读反馈'
    else:
        observation = recorded_observation(args.dataset, args.episode, args.frame)
        if args.command == 'self-test':
            report = self_test(observation)
        else:
            report = query(observation, args.port, args.count, args.timeout, jpeg=args.jpeg)
        report.update(dataset=args.dataset.name, episode=args.episode, frame=args.frame)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f'检查失败：{exc}', file=sys.stderr)
        raise SystemExit(1) from None
