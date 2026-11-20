"""Paired LeRobot videos and a durable, one-to-one correction telemetry journal."""

from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter

import numpy as np

from b601_preflight import JOINTS
from rebot_capture import append_frame, camera_frames, open_dataset


class RecordingError(RuntimeError):
    pass


def recap_values(row):
    return {
        "recap.advantage_tag": np.array([1 if row["force_positive"] else -1], dtype=np.int64),
        "recap.advantage_pool": np.array([row["advantage_pool"]], dtype=bool),
        "recap.policy_eligible": np.array([row["policy_eligible"]], dtype=bool),
    }


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def validate_saved_media(root, expected, *, telemetry_path=None):
    """After motion stops, verify real encoded frame counts and table/vector alignment."""
    import av
    import pyarrow.parquet as pq
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 1 or info["total_frames"] != expected:
        raise RecordingError("LeRobot 元数据与实际写入帧数不一致")
    tables = list((root / "data").rglob("*.parquet"))
    rows = [r for file in tables for r in pq.read_table(file).to_pylist()]
    if len(rows) != expected or [r["frame_index"] for r in rows] != list(range(expected)):
        raise RecordingError("状态/动作表的帧数或顺序不一致")
    if any(len(r[key]) != 7 or not np.isfinite(r[key]).all()
           for r in rows for key in ("action", "observation.state")):
        raise RecordingError("状态或动作不是七维有限数值")
    names = [f"{j}.pos" for j in JOINTS]
    if any(info["features"][key].get("names") != names for key in ("action", "observation.state")):
        raise RecordingError("数据集状态/动作的关节顺序与本项目不一致")
    if telemetry_path is None:
        telemetry_path = root / "telemetry-episode-000000.partial.jsonl"
        if not telemetry_path.is_file():
            telemetry_path = root / "telemetry-episode-000000.jsonl"
    with telemetry_path.open(encoding="utf-8") as file:
        trace = [json.loads(line) for line in file]
    if len(trace) != expected:
        raise RecordingError("逐帧日志与数据表的行数不一致")
    for i, (row, source) in enumerate(zip(rows, trace, strict=True)):
        if source["frame_index"] != i:
            raise RecordingError("逐帧日志的帧号不连续")
        for field, original in (("observation.state", "measured_motor_deg"), ("action", "action_adapter_deg")):
            if not np.allclose(row[field], source[original], atol=1e-4, rtol=1e-6):
                raise RecordingError(f"第 {i} 帧 {field} 与实际记录不一致")
        if "recap.advantage_tag" in info["features"]:
            for field, value in recap_values(source).items():
                if np.asarray(row[field]).item() != value.item():
                    raise RecordingError(f"第 {i} 帧 {field} 与接管日志不一致")
    counts = {}
    for role in ("wrist", "overhead"):
        files = list((root / "videos" / f"observation.images.{role}").rglob("*.mp4"))
        count = 0
        for file in files:
            with av.open(str(file)) as video:
                count += sum(1 for _ in video.decode(video=0))
        if count != expected:
            raise RecordingError(f"{role} 视频实际 {count} 帧，状态/动作 {expected} 帧")
        counts[role] = count
    return {"state_action_rows": len(rows), "decoded_video_frames": counts}


class CorrectionRecorder:
    """No encoding, CSV, JSON serialization or disk flush runs on the CAN thread."""

    def __init__(self, root, follower, cameras, task, fps, metadata):
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(f"纠错记录目录已存在：{self.root}")
        self.recap = metadata.get("intervention_control") == "black_follower_gravity"
        if self.recap:
            extra = {name: {"dtype": dtype, "shape": (1,), "names": None} for name, dtype in (
                ("recap.advantage_tag", "int64"), ("recap.advantage_pool", "bool"), ("recap.policy_eligible", "bool"))}
            self.dataset = open_dataset(self.root, follower, cameras, fps, extra_features=extra)
        else:
            self.dataset = open_dataset(self.root, follower, cameras, fps)
        self.task = task
        self.direction = np.array([follower.config.joint_directions[j] for j in JOINTS])
        self.pending = Queue(maxsize=16)
        self.finished = Event()
        self.failure = None
        self.submitted = 0
        self.written = 0
        self.closed = False
        self.frame_times = []
        self.mode_frames = {}
        self.missing_camera_frames = {"wrist": 0, "overhead": 0}
        self.skipped_control_ticks = 0
        self.previous_tick = None
        self.manifest = {
            "schema": "b601-correction-v1", "created_at": datetime.now().astimezone().isoformat(),
            "status": "in_progress", "task": task, "fps": fps, "metadata": metadata,
            "joint_order": list(JOINTS), "state_units": "motor_degrees",
            "action_units": "absolute_adapter_input_degrees",
            "joint_directions": self.direction.tolist(),
            "recorded_phases": ["model", "align", "human", "entering_drag", "drag", "resuming"],
            "excluded_phases": ["preflight", "return_to_zero"],
            "telemetry": "telemetry-episode-000000.jsonl",
            "events": "events.jsonl", "model_requests": "model-requests.jsonl",
        }
        if metadata.get("intervention_control") == "black_follower_gravity":
            self.manifest["recap_labels"] = {
                "source": "LeRobot parquet + telemetry on frame_index; recap.advantage_tag: 1 positive, -1 unscored (not negative)",
                "forced_positive": "hand-guided interval, next measured seven-axis pose; when G holds J7, its action is the captured aperture target",
                "ranking_pool": "valid model actions; original teleop demos enter at value scoring",
                "ranking_positive_fraction": 0.4, "forced_positive_outside_ranking": True,
                "waiting_and_transition_frames": "retained for audit, excluded from policy tags",
            }
        atomic_json(self.root / "session.json", self.manifest)
        self.thread = Thread(target=self._write, name="correction-recorder", daemon=True)
        self.thread.start()

    def check(self):
        if self.failure is not None:
            raise RecordingError(f"纠错记录写入失败：{self.failure}") from self.failure
        if self.closed:
            raise RecordingError("纠错记录已关闭")
        if self.pending.full():
            raise RecordingError("纠错记录写入队列已满")

    def _submit(self, kind, payload):
        self.check()
        try:
            self.pending.put_nowait((kind, payload))
        except Full as error:
            raise RecordingError("纠错写入队列已满，不能丢帧后继续作为完整记录") from error

    def journal(self, kind, payload):
        if kind not in ("event", "request"):
            raise ValueError("未知日志类型")
        self._submit(kind, payload)

    def frame(self, observation, images, telemetry):
        row = dict(telemetry, frame_index=self.submitted)
        training_action = telemetry.get("demonstrated_motor_deg", telemetry["target_motor_deg"])
        action = {f"{j}.pos": float(training_action[i] / self.direction[i])
                  for i, j in enumerate(JOINTS)}
        row["action_adapter_deg"] = list(action.values())
        # LeRobot build_dataset_frame expects camera names without the feature prefix.
        copied = {name: image.copy() for name, image in images.items()}
        self._submit("frame", (observation | copied, action, row))
        self.submitted += 1

    @staticmethod
    def flush(files):
        for file in files:
            file.flush()
            os.fsync(file.fileno())

    def _write(self):
        try:
            with (self.root / "telemetry-episode-000000.partial.jsonl").open("w", encoding="utf-8") as telemetry, \
                 (self.root / "events.jsonl").open("w", encoding="utf-8") as events, \
                 (self.root / "model-requests.jsonl").open("w", encoding="utf-8") as requests:
                next_flush = perf_counter() + 1
                files = (telemetry, events, requests)
                while True:
                    item = self.pending.get()
                    if item is None:
                        break
                    kind, value = item
                    if kind == "frame":
                        observation, action, row = value
                        if self.recap:
                            append_frame(self.dataset, self.task, observation, action, extra_values=recap_values(row))
                        else:
                            append_frame(self.dataset, self.task, observation, action)
                        telemetry.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        self.written += 1
                        if "seconds" in row:
                            self.frame_times.append(row["seconds"])
                        mode = row.get("control_mode", "unknown")
                        self.mode_frames[mode] = self.mode_frames.get(mode, 0) + 1
                        for name, valid in row.get("camera_valid", {}).items():
                            self.missing_camera_frames[name] += int(not valid)
                        tick = row.get("controller_tick")
                        if tick is not None and self.previous_tick is not None:
                            self.skipped_control_ticks += max(0, tick - self.previous_tick - 1)
                        self.previous_tick = tick
                    else:
                        target = events if kind == "event" else requests
                        target.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                    if kind == "event" or perf_counter() >= next_flush:
                        self.flush(files)
                        next_flush = perf_counter() + 1
                self.flush(files)
        except BaseException as error:
            self.failure = error
        finally:
            self.finished.set()

    def close(self, outcome, report):
        if self.closed:
            return self.manifest
        # The caller first stops motion. Drain accepted frames, even on an exception.
        interrupted = False
        while self.thread.is_alive():
            try:
                self.pending.put(None, timeout=.05)
                break
            except Full:
                continue
            except KeyboardInterrupt:
                interrupted = True
        while not self.finished.is_set():
            try:
                self.finished.wait(.05)
            except KeyboardInterrupt:
                interrupted = True
        self.thread.join()
        saved = False
        errors = [] if self.failure is None else [str(self.failure)]
        try:
            if self.failure is None and self.written and self.written == self.submitted:
                self.dataset.save_episode()
                saved = True
        except BaseException as error:
            errors.append(str(error) or type(error).__name__)
        finally:
            try:
                self.dataset.finalize()
            except BaseException as error:
                errors.append(str(error) or type(error).__name__)
        partial = self.root / "telemetry-episode-000000.partial.jsonl"
        media_check = None
        if saved and not errors:
            try:
                media_check = validate_saved_media(self.root, self.written)
            except BaseException as error:
                errors.append(str(error) or type(error).__name__)
        if saved and not errors:
            partial.replace(self.root / "telemetry-episode-000000.jsonl")
        self.closed = True
        self.manifest.update({
            "status": "saved" if saved and not errors else "partial",
            "outcome": outcome, "end_reason": report.get("end_reason"),
            "human_intervention_used": report.get("assistance_marked", False),
            "operator_pause_used": report.get("operator_pause_used", False),
            "task_success_requires_review": True,
            "ready_for_training_without_review": False,
            "frames_by_control_mode": self.mode_frames,
            "missing_camera_frames": self.missing_camera_frames,
            "skipped_control_ticks": self.skipped_control_ticks,
            "frames_submitted": self.submitted, "frames_written": self.written,
            "episode_saved": saved, "errors": errors, "close_interrupted": interrupted,
            "media_validation": media_check,
            "finished_at": datetime.now().astimezone().isoformat(),
        })
        if len(self.frame_times) > 1:
            intervals = np.diff(self.frame_times) * 1000
            self.manifest["sample_interval_ms"] = {
                "p50": float(np.percentile(intervals, 50)),
                "p95": float(np.percentile(intervals, 95)), "max": float(intervals.max())}
        atomic_json(self.root / "session.json", self.manifest)
        return self.manifest


def capture(cameras):
    """Keep both video tracks aligned, using black frames for missing images."""
    return camera_frames(cameras, allow_missing=("wrist", "overhead"))
