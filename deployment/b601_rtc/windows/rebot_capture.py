"""Local LeRobot recording and per-frame camera timestamps for rebot-follow.cmd."""

import csv
import json
import subprocess
from collections import deque
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Condition, Event, Thread
from time import perf_counter


FFMPEG = Path(__file__).resolve().parents[2] / "具身/B601-RS/runtime/ffmpeg/ffmpeg.exe"
# The same physical devices used by camera-preview.cmd. These cameras share a model name.
CAMERA_USB_IDS = {
    "wrist": r"USB\VID_0C45&PID_636F&MI_00\6&B738621&0&0000",
    "overhead": r"USB\VID_0C45&PID_636F&MI_00\7&7736773&0&0000",
}
VIDEO_INTERFACE = "{65e8773d-8f56-11d0-a3b9-00a0c9223196}"
RECORD_QUEUE_FRAMES = 8
TELEMETRY_FLUSH_S = 1.0


class CameraFrameError(RuntimeError):
    """The required camera has no usable frame."""


class FollowerReadError(RuntimeError):
    """A follower angle request failed, with the addressed device attached."""

    def __init__(self, message, request):
        super().__init__(message)
        self.request = request


def directshow_camera_indices():
    if not FFMPEG.is_file():
        raise CameraFrameError(f"找不到用于识别相机的 FFmpeg：{FFMPEG}")
    try:
        sources = subprocess.run(
            [str(FFMPEG), "-hide_banner", "-sources", "dshow"],
            capture_output=True, check=True, timeout=8, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CameraFrameError(f"无法读取 DirectShow 相机列表：{exc}") from exc

    # FFmpeg lists the DirectShow video category first, in enumeration order.
    # Keep non-USB entries here because they can also occupy an OpenCV index.
    monikers = [line.strip().partition(" [")[0].casefold()
                for line in sources.splitlines() if line.lstrip().startswith("@device_")]
    indices = {}
    for name, device_id in CAMERA_USB_IDS.items():
        pnp_id = device_id.replace("\\", "#")
        expected = rf"@device_pnp_\\?\{pnp_id}#{VIDEO_INTERFACE}\global".casefold()
        matches = [index for index, moniker in enumerate(monikers) if moniker == expected]
        if len(matches) > 1:
            raise CameraFrameError(f"{name} 的 USB 设备 ID 重复，无法确定相机。")
        if matches:
            indices[name] = matches[0]
    return indices


@contextmanager
def connect_cameras(*, optional_overhead=False):
    import yaml
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    config_path = Path(__file__).with_name("record-smoke.yaml")
    with config_path.open(encoding="utf-8") as file:
        configs = yaml.safe_load(file)["robot"]["cameras"]

    indices = directshow_camera_indices()
    cameras = {}
    opened = []
    with ExitStack() as stack:
        for name in ("wrist", "overhead"):
            config = {key: value for key, value in configs[name].items() if key != "type"}
            if config.get("backend") != Cv2Backends.DSHOW:
                raise CameraFrameError("本机的相机 USB 绑定需要 DirectShow 后端。")
            index = indices.get(name)
            # Keep the configured image dimensions for black-frame fallback.
            config["index_or_path"] = -1 if index is None else index
            camera = OpenCVCamera(OpenCVCameraConfig(**config))
            try:
                if index is None:
                    raise CameraFrameError(f"未找到已绑定的 {name} 相机，请检查原 USB 接口。")
                camera.connect()
            except Exception as exc:
                if name != "overhead" or not optional_overhead:
                    raise
                print(f"俯视相机暂时无法打开，采集时用黑帧占位。{exc}", flush=True)
            else:
                stack.callback(camera.disconnect)
                opened.append(name)
            cameras[name] = camera
        current_indices = directshow_camera_indices()
        if any(current_indices.get(name) != indices[name] for name in opened):
            raise CameraFrameError("相机列表在打开期间发生变化，请放稳 USB 连接后重新打开 CMD。")
        print("相机已按 USB 设备 ID 打开。" + "、".join(
            f"{'腕部' if name == 'wrist' else '俯视'} {indices[name]} 号" for name in opened
        ) + "。", flush=True)
        yield cameras


def camera_frames(cameras, *, overhead_placeholder=None):
    images, timestamps = {}, {}
    for name, camera in cameras.items():
        # The SDK's capture thread publishes the image and host timestamp together.
        with camera.frame_lock:
            image, timestamp = camera.latest_frame, camera.latest_timestamp
        if image is None or timestamp is None or perf_counter() - timestamp > 0.5:
            if name != "overhead" or overhead_placeholder is None:
                raise CameraFrameError(f"{name} 相机没有新画面，请检查相机连接。")
            image, timestamp = overhead_placeholder, None
        images[name], timestamps[name] = image, timestamp
    return images, timestamps


def open_dataset(root, follower, cameras, fps):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import hw_to_dataset_features

    options = dict(
        repo_id=f"local/{root.name}", root=root, video_backend="pyav",
        streaming_encoding=True, encoder_threads=2,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=23, g=12),
    )
    if root.is_dir():
        return LeRobotDataset.resume(**options)

    camera_features = {name: (camera.height, camera.width, 3) for name, camera in cameras.items()}
    features = hw_to_dataset_features(follower.action_features, "action")
    features.update(hw_to_dataset_features(follower.observation_features | camera_features, "observation"))
    return LeRobotDataset.create(
        fps=fps, features=features, robot_type=follower.name, use_videos=True, **options,
    )


def append_frame(dataset, task, observation, action):
    from lerobot.utils.feature_utils import build_dataset_frame

    dataset.add_frame({
        **build_dataset_frame(dataset.features, observation, prefix="observation"),
        **build_dataset_frame(dataset.features, action, prefix="action"),
        "task": task,
    })
    # This LeRobot encoder can drop frames when saturated. Refuse that episode rather
    # than saving image streams whose frame indices no longer match the joint data.
    encoder = dataset.writer._streaming_encoder
    if any(encoder._dropped_frames.values()):
        raise RuntimeError("视频编码没有跟上，当前条未保存为有效 episode。")


class EpisodeWriter:
    """Write paired dataset frames and telemetry outside the control loop."""

    def __init__(self, dataset, task, telemetry, header, *, flush=None):
        self._pending = deque()
        self._ready = Condition()
        self._closing = False
        self._failure = None
        self._finished = Event()
        self._thread = Thread(
            target=self._write_frames, args=(dataset, task, telemetry, header, flush),
            name="episode-writer",
        )

    def __enter__(self):
        self._thread.start()
        return self

    def check(self):
        if self._failure is not None:
            raise RuntimeError(f"录像写入失败（{self._failure}）") from self._failure
        if len(self._pending) >= RECORD_QUEUE_FRAMES:
            raise RuntimeError("录像写入积压，本条录制已中止。")

    def submit(self, observation, action, row):
        with self._ready:
            self.check()
            self._pending.append((observation, action, row))
            self._ready.notify()

    def __exit__(self, exc_type, exc, traceback):
        with self._ready:
            self._closing = True
            if exc_type is not None and not issubclass(exc_type, (CameraFrameError, FollowerReadError)):
                self._pending.clear()
            self._ready.notify()

        # Keep the dataset and CSV open until the writer has stopped, even on Ctrl+C.
        interrupted = None
        while not self._finished.is_set():
            try:
                self._finished.wait(0.05)
            except KeyboardInterrupt as exc:
                interrupted = exc
        self._thread.join()
        if interrupted is not None:
            raise interrupted
        if exc is None:
            self.check()
        elif self._failure is not None:
            exc.add_note(f"后台写入也报告错误：{self._failure}")

    def _write_frames(self, dataset, task, telemetry, header, flush):
        try:
            telemetry.writerow(header)
            next_flush = perf_counter() + TELEMETRY_FLUSH_S
            while True:
                with self._ready:
                    self._ready.wait_for(lambda: self._pending or self._closing)
                    if not self._pending:
                        return
                    observation, action, row = self._pending.popleft()
                append_frame(dataset, task, observation, action)
                telemetry.writerow(row)
                if flush is not None and perf_counter() >= next_flush:
                    flush()
                    next_flush = perf_counter() + TELEMETRY_FLUSH_S
        except BaseException as exc:
            self._failure = exc
        finally:
            self._finished.set()


def write_fault_log(directory, report):
    directory.mkdir(exist_ok=True)
    temporary = directory / "fault.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "fault.json")


def preserve_interrupted_episode(dataset, partial_trace, report, *, keep_video=True):
    directory = partial_trace.with_name(partial_trace.name.removesuffix(".partial.csv") + "-interrupted")
    report["artifacts"] = {}
    if not keep_video:
        report["artifacts"]["partial_telemetry"] = f"../{partial_trace.name}"
        report["video_archive_error"] = "未确认从臂失能，跳过视频收尾。"
        write_fault_log(directory, report)
        return directory

    # Commit the error record before waiting on the video encoder.
    write_fault_log(directory, report)
    try:
        if partial_trace.is_file():
            telemetry = directory / "telemetry.csv"
            partial_trace.replace(telemetry)
            report["artifacts"]["telemetry"] = telemetry.name
            count, last = 0, None
            with telemetry.open(encoding="utf-8", newline="") as file:
                for count, last in enumerate(csv.DictReader(file), 1):
                    pass
            report["telemetry_rows"] = count
            report["last_telemetry_row"] = last
    except (OSError, csv.Error, UnicodeError) as exc:
        report["telemetry_archive_error"] = str(exc)
    write_fault_log(directory, report)

    try:
        buffer = dataset.writer.episode_buffer
        if buffer is not None and buffer["size"]:
            encoder = dataset.writer._streaming_encoder
            report["encoder_dropped_frames"] = dict(encoder._dropped_frames)
            streams = encoder.finish_episode()
            for name in ("wrist", "overhead"):
                path, _ = streams[f"observation.images.{name}"]
                if path.is_file() and path.stat().st_size > 0:
                    video = directory / f"{name}.mp4"
                    path.replace(video)
                    report["artifacts"][name] = video.name
    except Exception as exc:
        report["video_archive_error"] = str(exc)
    write_fault_log(directory, report)
    return directory
