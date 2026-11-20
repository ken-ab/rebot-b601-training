"""Read the existing passive 102 calibration without origin or servo writes."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter
from unittest.mock import patch

import numpy as np

from b601_preflight import JOINTS


@dataclass
class LeaderSample:
    angles: np.ndarray
    raw: np.ndarray
    sampled_at: float
    reliable: list[bool]


class LeaderReader:
    MAX_AGE_S = 0.5

    def __init__(self, leader):
        self.leader = leader
        self._stop = Event()
        self._lock = Lock()
        self._sample = None
        self._error = "主臂读数尚未就绪"
        self.thread = Thread(target=self._run, name="correction-leader", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            started = perf_counter()
            try:
                samples = [self.leader.bus.read_angle(self.leader.config.joint_ids[j]) for j in JOINTS]
                angles = np.asarray([s.filtered_deg for s in samples], dtype=float)
                raw = np.asarray([s.raw_deg for s in samples], dtype=float)
                reliable = [bool(s.reliable) for s in samples]
                if not np.isfinite(angles).all() or not np.isfinite(raw).all() or not all(reliable):
                    raise RuntimeError("主臂至少一轴没有可靠的新鲜读数")
                if perf_counter() - started > self.MAX_AGE_S:
                    raise RuntimeError("主臂七轴读取耗时过长，本组不作为新鲜读数")
                sample = LeaderSample(angles, raw, started, reliable)
                with self._lock:
                    self._sample, self._error = sample, None
            except Exception as error:
                with self._lock:
                    self._error = str(error)
            self._stop.wait(max(0.0, .04 - (perf_counter() - started)))

    def latest(self):
        with self._lock:
            sample, error = self._sample, self._error
        if sample is None or error or perf_counter() - sample.sampled_at > self.MAX_AGE_S:
            return None, error or "主臂读数已过期"
        return sample, None

    def wait_ready(self, timeout=4):
        deadline = perf_counter() + timeout
        while perf_counter() < deadline:
            sample, error = self.latest()
            if sample is not None:
                return sample
            self._stop.wait(.02)
        raise RuntimeError(error or "主臂没有可靠读数")

    def close(self):
        self._stop.set()
        # Motion has already stopped before this cleanup. Wait for the bounded UART read.
        self.thread.join()


@contextmanager
def passive_leader(port="COM3"):
    from lerobot_teleoperator_rebot_arm_102.config_rebot_arm_102_leader import RebotArm102LeaderConfig
    from lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader import RebotArm102Leader
    leader = RebotArm102Leader(RebotArm102LeaderConfig(
        port=port, baudrate=1_000_000, id="rebot_arm_102_leader"))
    if not leader.is_calibrated or set(leader.config.joint_ids) != set(JOINTS):
        raise RuntimeError("主臂七轴标定或关节映射不完整，请先核对原有标定")
    reader = None
    try:
        # Official configure resets multi-turn counters; relative takeover must not do that.
        with patch.object(leader, "configure", lambda: None):
            leader.connect(calibrate=False)
        reader = LeaderReader(leader)
        reader.start()
        reader.wait_ready()
        yield reader
    finally:
        if reader is not None:
            reader.close()
        if leader.is_connected:
            leader.disconnect()
