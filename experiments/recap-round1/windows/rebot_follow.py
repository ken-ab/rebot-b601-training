"""Interactive two-camera LeRobot recording for the Windows B601-RS setup."""

import argparse
import csv
import ctypes
import logging
import math
import msvcrt
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep, time_ns
from unittest.mock import patch

from rebot_capture import (
    EpisodeWriter, FollowerReadError, camera_frames, connect_cameras, open_dataset,
    preserve_interrupted_episode, write_fault_log,
)

JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
)
TELEMETRY_COLUMNS = (
    "frame_index", "unix_time_ns", "cycle_s", "leader_read_s", "follower_read_s", "send_done_s",
    "wrist_frame_s", "overhead_frame_s", "overhead_valid",
    *(f"J{index}_{field}" for field in (
        "leader_raw_deg", "leader_used_deg", "leader_reliable", "requested_motor_deg",
        "sent_motor_deg", "follower_deg", "follower_deg_s", "follower_torque_nm",
    ) for index in range(1, len(JOINTS) + 1)),
    "position_poll_joint", "position_poll_ms",
    *(f"J{index}_{field}" for field in (
        "cached_status_code", "cached_can_arbitration_id", "cached_mos_temp_c",
    ) for index in range(1, len(JOINTS) + 1)),
)
FPS = 25
PERIOD = 1 / FPS
START_DRIFT_DEG = 1.0
REQUEST_TIMEOUT_MS = 150
MECH_POS_REGISTER = 0x7019
# RobStride type 21 fault bits, as exposed by the MotorBridge cache.
DRIVE_FAULT_BITS = {
    0: "过温", 1: "驱动故障", 2: "欠压", 3: "过压", 4: "B 相过流",
    5: "C 相过流", 7: "编码器未标定", 8: "硬件 ID 故障",
    9: "位置初始化故障", 14: "堵转过载", 16: "A 相过流",
}
ESCAPE = "\x1b"
RETURN_ARM_DEG_S = 8.0
RETURN_GRIPPER_DEG_S = 24.0
# The same reference is used throughout a batch. A released joint can settle
# slightly away from the position it held under torque.
RETURN_TOLERANCE_DEG = (1.5, 3.0)  # Arm joints, then the gripper motor.
START_TOLERANCE_DEG = (3.0, 6.0)
RETURN_STILL_DEG = 0.35
RETURN_SETTLE_S = 0.4
RETURN_WAIT_STILL_S = 2.5
PCAN_DIR = Path(__file__).resolve().parents[2] / "具身/B601-RS/runtime/pcan/x64"


@dataclass
class Take:
    frames: int = 0
    missing_frames: int = 0
    start_pose: dict[str, float] | None = None
    stop_key: str | None = None


class StartPoseNotReady(RuntimeError):
    """The follower is outside the pose needed to start another take."""


class StartPoseOutOfRange(StartPoseNotReady):
    def __init__(self, joint, angle, lower, upper):
        super().__init__(f"从臂 {joint} 实测 {angle:+.2f}°，超出起始行程 {lower:g}° 到 {upper:g}°（容差 {START_DRIFT_DEG:g}°）。")
        self.recording_context = {
            "stage": "check_start_pose", "frame_index": None, "detected_unix_ns": time_ns(),
            "joint": joint, "angle_deg": angle, "limits_deg": [lower, upper], "tolerance_deg": START_DRIFT_DEG,
        }


class ReturnPoseOutOfRange(RuntimeError):
    def __init__(self, joint, angle):
        super().__init__(f"{joint} 实际位置 {angle:+.2f}° 超出原有行程。")
        self.recording_context = {
            "stage": "return_to_start", "frame_index": None, "detected_unix_ns": time_ns(),
            "joint": joint, "angle_deg": angle,
        }


class MotorPower:
    def __init__(self, follower, *, needs_disable=True):
        self.follower = follower
        self.needs_disable = needs_disable
        self.last_disable = None

    def enable(self, *, returning=False, stop_check=None):
        # An enable failure can leave some axes on, so arm cleanup first.
        self.last_disable = None
        if returning:
            check_stop = return_stop_check if stop_check is None else stop_check
            for joint in JOINTS:
                check_stop()
                self.needs_disable = True
                self.follower.motors[joint].enable()
        else:
            self.needs_disable = True
            self.follower.bus.enable_all()

    def disable(self):
        if self.needs_disable:
            result = {"started_unix_ns": time_ns(), "joints": []}
            self.last_disable = result
            try:
                disable_motors(self.follower, result["joints"])
                self.needs_disable = False
            finally:
                result["finished_unix_ns"] = time_ns()
                result["all_acknowledged"] = not self.needs_disable


def read_leader(leader):
    samples = {}
    for joint in JOINTS:
        sample = leader.bus.read_angle(leader.config.joint_ids[joint])
        if not math.isfinite(sample.raw_deg) or not math.isfinite(sample.filtered_deg):
            raise RuntimeError(f"主臂 {joint} 读数无效")
        samples[joint] = sample
    return samples


def read_follower_joint(follower, joint, *, timeout_ms=REQUEST_TIMEOUT_MS):
    can_id, host_id = follower.config.motor_can_ids[joint]
    motor = follower.motors[joint]
    started = time_ns()
    try:
        angle = math.degrees(motor.robstride_get_param_f32_host_id(MECH_POS_REGISTER, host_id, timeout_ms))
        if not math.isfinite(angle):
            raise ValueError("收到无效角度")
        return angle
    except Exception as exc:
        label = joint_label(joint)
        raise FollowerReadError(f"从臂 {label}（CAN ID {can_id}）位置读取失败：{exc}", {
            "joint": label,
            "motor_can_id": can_id,
            "host_id": f"0x{host_id:02X}",
            "parameter": f"0x{MECH_POS_REGISTER:04X}",
            "timeout_ms": timeout_ms,
            "started_unix_ns": started,
            "failed_unix_ns": time_ns(),
        }) from exc


def read_follower(follower):
    return {joint: read_follower_joint(follower, joint) for joint in JOINTS}


def read_stop_key():
    stop_key = None
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
        elif key == ESCAPE:
            stop_key = ESCAPE
        elif key.lower() == "q" and stop_key is None:
            stop_key = "q"
    return stop_key


def disable_motors(follower, results):
    errors = []
    for joint, motor in follower.motors.items():
        result = {"joint": joint_label(joint), "started_unix_ns": time_ns()}
        results.append(result)
        try:
            motor.disable()
        except Exception as exc:
            errors.append(f"{joint}: {exc}")
            result.update(acknowledged=False, error=str(exc))
        else:
            result["acknowledged"] = True
        finally:
            result["finished_unix_ns"] = time_ns()
    if errors:
        print("失能未能全部确认，请立即使用物理急停或断电。", flush=True)
        raise RuntimeError("从臂失能回包异常：" + "; ".join(errors))


def describe_fault(power, error, stage, episode_index):
    errors, contexts, request = [], [], None
    seen = set()
    cause = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        errors.append({"type": type(cause).__name__, "message": str(cause),
                       "notes": getattr(cause, "__notes__", [])})
        if context := getattr(cause, "recording_context", None):
            contexts.append(context)
        if isinstance(cause, FollowerReadError):
            request = cause.request
        cause = cause.__cause__ or cause.__context__

    if contexts:
        failure = min(contexts, key=lambda item: item["detected_unix_ns"])
    else:
        failure = {"stage": stage, "frame_index": None,
                   "detected_unix_ns": request["failed_unix_ns"] if request else time_ns()}
    detected_at = datetime.fromtimestamp(failure["detected_unix_ns"] / 1e9).astimezone()
    failure = dict(failure, local_time=detected_at.isoformat(timespec="milliseconds"))

    # This MotorBridge getter reads the last received fault frame from memory.
    fault_words = {}
    for joint, motor in power.follower.motors.items():
        label = joint_label(joint)
        try:
            raw, warning = motor.robstride_get_fault_report()
            fault_words[label] = {
                "fault_raw": raw,
                "warning_raw": warning,
                "faults": [name for bit, name in DRIVE_FAULT_BITS.items() if raw & (1 << bit)],
            }
        except Exception as exc:
            fault_words[label] = {"read_error": str(exc)}

    return {
        "episode_index": episode_index,
        "failure": failure,
        "motor_position_request": request,
        "exceptions": errors,
        "latest_disable_attempt": power.last_disable,
        "cached_drive_fault_reports": fault_words,
        "feedback_note": "状态与故障字来自 SDK 缓存。故障字全零也可能是没有收到过故障帧。",
    }


def disable_and_close(power):
    follower = power.follower
    bus = follower.bus
    try:
        power.disable()
    finally:
        try:
            for motor in follower.motors.values():
                motor.close()
        finally:
            try:
                bus.close()
            finally:
                # Prevent the base class destructor from disconnecting a closed bus.
                follower.bus = None


@contextmanager
def connect_follower_disabled():
    """Connect the B601 for feedback with torque disabled, without opening the leader."""
    dll = PCAN_DIR / "PCANBasic.dll"
    if not dll.is_file():
        raise RuntimeError(f"找不到 PCAN 驱动：{dll}")
    os.environ["PATH"] = str(PCAN_DIR) + os.pathsep + os.environ.get("PATH", "")
    ctypes.WinDLL(str(dll), winmode=0)

    from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
    from lerobot_robot_seeed_b601.seeed_b601_rs_follower import SeeedB601RSFollower

    follower = SeeedB601RSFollower(SeeedB601RSFollowerConfig(
        port="can0@1000000", can_adapter="socketcan", id="follower1",
    ))
    power = MotorPower(follower)
    follower._skip_safe_zero_on_disconnect = True
    if set(follower.calibration) != set(JOINTS):
        raise RuntimeError("未找到从臂完整的标定文件，请核对设备 ID 和现有标定。")
    if set(follower.config.joint_directions) != set(JOINTS):
        raise RuntimeError("从臂的关节映射与本项目配置不一致。")

    def keep_follower_disabled():
        power.disable()
        for joint, motor in follower.motors.items():
            if motor.robstride_get_param_i8(0x7005, REQUEST_TIMEOUT_MS) != 0:
                raise RuntimeError(f"从臂 {joint} 不在 MIT 模式，请核对电机配置。")

    try:
        with patch.object(follower, "configure", keep_follower_disabled):
            follower.connect(calibrate=False)
        yield power
    finally:
        if follower.bus is not None:
            disable_and_close(power)


@contextmanager
def connect_disabled():
    from lerobot_teleoperator_rebot_arm_102.config_rebot_arm_102_leader import RebotArm102LeaderConfig
    from lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader import RebotArm102Leader

    leader = RebotArm102Leader(RebotArm102LeaderConfig(
        port="COM3", baudrate=1_000_000, id="rebot_arm_102_leader",
    ))
    if not leader.is_calibrated:
        raise RuntimeError("未找到主臂完整的标定文件，请核对设备 ID 和现有标定。")
    if set(leader.config.joint_ids) != set(JOINTS):
        raise RuntimeError("主臂的关节映射与本项目配置不一致。")

    try:
        # Relative following uses fresh angles and needs no writes to the passive leader.
        with patch.object(leader, "configure", lambda: None):
            leader.connect(calibrate=False)
        with connect_follower_disabled() as power:
            yield leader, power
    finally:
        if leader.is_connected:
            leader.disconnect()


def wait_for_ring(leader, closed_angle):
    opened_since = None
    opened = False
    while not read_stop_key():
        start = perf_counter()
        ring = read_leader(leader)["gripper"].filtered_deg
        delta = math.remainder(ring - closed_angle, 360.0)
        if opened:
            if abs(delta) <= 0.8:
                return True
        elif delta >= 1.5:
            if opened_since is None:
                opened_since = perf_counter()
            elif perf_counter() - opened_since >= 0.45:
                opened = True
                print("指环动作已读到，合回后开始采集。", flush=True)
        else:
            opened_since = None
        sleep(max(0.0, PERIOD - (perf_counter() - start)))
    return False


def clamp(value, lower, upper):
    return min(upper, max(lower, value))


def joint_label(joint):
    return f"J{JOINTS.index(joint) + 1}"


def outside_pose_tolerance(pose, reference, tolerance_deg=RETURN_TOLERANCE_DEG):
    arm_tolerance, gripper_tolerance = tolerance_deg
    for joint in JOINTS:
        tolerance = gripper_tolerance if joint == "gripper" else arm_tolerance
        if abs(pose[joint] - reference[joint]) > tolerance:
            return joint
    return None


def check_start_pose(pose, reference, *, tolerance_deg=START_TOLERANCE_DEG):
    if reference is not None and (joint := outside_pose_tolerance(pose, reference, tolerance_deg)):
        offset = pose[joint] - reference[joint]
        raise StartPoseNotReady(f"{joint_label(joint)} 比本批起始位偏 {offset:+.2f}°。")


def checked_start_origin(pose, limits):
    origin = {}
    for joint in JOINTS:
        angle = pose[joint]
        lower, upper = limits[joint]
        target = clamp(angle, lower, upper)
        if abs(angle - target) > START_DRIFT_DEG:
            raise StartPoseOutOfRange(joint_label(joint), angle, lower, upper)
        origin[joint] = target
    return origin


def return_stop_check():
    # Q may repeat while it is held down to save. Esc is the separate cancel key.
    if read_stop_key() == ESCAPE:
        raise KeyboardInterrupt


def read_return_pose(follower, *, stop_check=return_stop_check):
    pose = {}
    for joint in JOINTS:
        stop_check()
        angle = read_follower_joint(follower, joint)
        state = follower.motors[joint].get_state()
        if state is None or state.status_code:
            raise RuntimeError(f"{joint_label(joint)} 电机反馈缺失或报告故障。")
        lower, upper = follower.config.joint_limits[joint]
        if not lower - START_DRIFT_DEG <= angle <= upper + START_DRIFT_DEG:
            raise ReturnPoseOutOfRange(joint_label(joint), angle)
        pose[joint] = angle
    return pose


def stable_disabled_pose(follower, *, verify_motion_pose=None, stop_check=None):
    stop_options = {} if stop_check is None else {'stop_check': stop_check}
    anchor = read_return_pose(follower, **stop_options)
    if verify_motion_pose is not None:
        verify_motion_pose(anchor)
    still_since = perf_counter()
    deadline = still_since + RETURN_WAIT_STILL_S
    while True:
        sleep(PERIOD)
        actual = read_return_pose(follower, **stop_options)
        if verify_motion_pose is not None:
            verify_motion_pose(actual)
        now = perf_counter()
        if now >= deadline:
            raise StartPoseNotReady("从臂失能后还未停稳。")
        if any(abs(actual[joint] - anchor[joint]) > RETURN_STILL_DEG for joint in JOINTS):
            anchor, still_since = actual, now
        elif now - still_since >= RETURN_SETTLE_S:
            return actual


def return_to_start(power, reference, *, verify_motion_pose=None, verify_start_pose=None,
                    stop_check=None, max_enable_change_deg=None, audit=None):
    """Return after saving, with no dataset, camera or leader input."""
    follower = power.follower
    for joint in JOINTS:
        lower, upper = follower.config.joint_limits[joint]
        if not lower <= reference[joint] <= upper:
            raise RuntimeError(f"{joint_label(joint)} 的起始位超出原有行程。")
    if verify_motion_pose is not None:
        verify_motion_pose(reference)
    check_stop = return_stop_check if stop_check is None else stop_check
    stop_options = {} if stop_check is None else {'stop_check': stop_check}
    start_options = dict(stop_options)
    if verify_start_pose is not None or verify_motion_pose is not None:
        start_options['verify_motion_pose'] = verify_start_pose or verify_motion_pose
    # Saving occurs with torque off, so the last recorded pose can be stale.
    measured = stable_disabled_pose(follower, **start_options)
    if outside_pose_tolerance(measured, reference) is None:
        check_stop()
        return

    print("正在从当前姿态慢速回起始位。请留在活动范围外，Esc 可取消。", flush=True)
    try:
        if audit is not None:
            audit.update(enable_requested_monotonic_s=perf_counter(), commands_accepted=0)
        power.enable(returning=True, **stop_options)
        if audit is not None:
            audit['enable_completed'] = True
        enabled_pose = read_return_pose(follower, **stop_options)
        if verify_motion_pose is not None:
            verify_motion_pose(enabled_pose)
        tolerance = max_enable_change_deg
        if tolerance is None:
            tolerance = RETURN_TOLERANCE_DEG
        elif not isinstance(tolerance, tuple):
            tolerance = (tolerance, tolerance)
        if joint := outside_pose_tolerance(enabled_pose, measured, tolerance):
            raise StartPoseNotReady(f"从臂 {joint_label(joint)} 使能时移位，已停止回位。")
        limits = follower.config.joint_limits
        start = {joint: clamp(enabled_pose[joint], *limits[joint]) for joint in JOINTS}
        delta = {joint: reference[joint] - start[joint] for joint in JOINTS}
        direction = follower.config.joint_directions
        # Smoothstep peaks at 1.5 times the average rate. Never skip steps to catch up.
        duration = max(1.0, max(
            1.5 * abs(delta[joint]) / (RETURN_GRIPPER_DEG_S if joint == "gripper" else RETURN_ARM_DEG_S)
            for joint in JOINTS
        ))
        steps = math.ceil(duration / PERIOD)
        deadline = perf_counter() + 3.0 * duration + 4.0
        step = 0
        settled = None
        while True:
            actual = read_return_pose(follower, **stop_options)
            if verify_motion_pose is not None:
                verify_motion_pose(actual)
            now = perf_counter()
            if now >= deadline:
                raise StartPoseNotReady("从臂未在回位时间内到达并停稳。")
            ratio = step / steps
            fraction = ratio * ratio * (3.0 - 2.0 * ratio)
            target = {joint: start[joint] + fraction * delta[joint] for joint in JOINTS}
            for joint in JOINTS:
                max_error = 12.0 if joint == "gripper" else 4.0
                if abs(actual[joint] - target[joint]) > max_error:
                    raise StartPoseNotReady(f"{joint_label(joint)} 未跟上回位目标。")

            if step == steps and outside_pose_tolerance(actual, reference) is None:
                if settled is None or any(abs(actual[j] - settled[1][j]) > RETURN_STILL_DEG for j in JOINTS):
                    settled = (now, actual)
                elif now - settled[0] >= RETURN_SETTLE_S:
                    break
            else:
                settled = None

            check_stop()
            if verify_motion_pose is not None:
                verify_motion_pose(target)
            follower.send_action({f"{joint}.pos": target[joint] / direction[joint] for joint in JOINTS})
            if audit is not None:
                audit['commands_accepted'] += 1
            step = min(steps, step + 1)
            sleep(PERIOD)
    except (Exception, KeyboardInterrupt) as exc:
        if audit is not None:
            audit.update(error=str(exc) or type(exc).__name__, failure_monotonic_s=perf_counter(),
                         plugin_overheat=isinstance(exc, KeyboardInterrupt) and str(exc) == 'Overheat detected')
        if isinstance(exc, Exception) and not hasattr(exc, "recording_context"):
            exc.recording_context = {"stage": "return_to_start", "frame_index": None, "detected_unix_ns": time_ns()}
        raise
    finally:
        try:
            power.disable()
        finally:
            if audit is not None:
                audit['disable'] = power.last_disable

    final_options = dict(stop_options)
    if verify_motion_pose is not None:
        final_options['verify_motion_pose'] = verify_motion_pose
    final = stable_disabled_pose(follower, **final_options)
    check_start_pose(final, reference)
    check_stop()


def return_for_next_take(power, reference, saved_trace):
    try:
        return_to_start(power, reference)
    except StartPoseNotReady as exc:
        return str(exc)
    except KeyboardInterrupt:
        print("已取消回位，已经保存的条目保留。", flush=True)
        raise
    except Exception as exc:
        directory = saved_trace.with_name(saved_trace.stem + "-return-fault")
        try:
            episode_index = int(saved_trace.stem.rsplit("-", 1)[1])
            report = describe_fault(power, exc, "return_to_start", episode_index)
            report["artifacts"] = {"saved_telemetry": f"../{saved_trace.name}"}
            write_fault_log(directory, report)
            print(f"已保存回位故障日志。\n{directory / 'fault.json'}", flush=True)
        except Exception as archive_error:
            print(f"写入故障日志失败：{archive_error}", file=sys.stderr, flush=True)
        if isinstance(exc, ReturnPoseOutOfRange) and not power.needs_disable:
            return f"{exc} 请确认从臂有稳固支撑，恢复到原有行程内并放稳。"
        raise RuntimeError(f"已保存的条目保留，回位未完成：{exc}") from exc
    print("从臂已在本批起始位附近停稳并失能。可以复位物体、准备下一条。", flush=True)
    return None


def follow(leader, power, cameras, writer, reference):
    import numpy as np

    follower = power.follower
    overhead = cameras["overhead"]
    black_overhead = np.zeros((overhead.height, overhead.width, 3), dtype=np.uint8)
    limits = follower.config.joint_limits
    initial = read_follower(follower)
    checked_start_origin(initial, limits)
    check_start_pose(initial, reference)
    closed_ring = read_leader(leader)["gripper"].filtered_deg
    print("已待命，从臂失能。白色指环向外打开，停约一秒，再合回。", flush=True)
    if not wait_for_ring(leader, closed_ring):
        return Take()

    current = read_follower(follower)
    origin = checked_start_origin(current, limits)
    if any(abs(current[joint] - initial[joint]) > START_DRIFT_DEG for joint in JOINTS):
        raise StartPoseNotReady("待命时从臂姿态变了。")
    check_start_pose(current, reference)
    camera_frames(cameras, overhead_placeholder=black_overhead)
    previous = {joint: sample.filtered_deg for joint, sample in read_leader(leader).items()}
    if read_stop_key():
        return Take()

    direction = follower.config.joint_directions
    displacement = dict.fromkeys(JOINTS, 0.0)
    frame_index = 0
    overhead_missing = False
    missing_frames = 0
    writer.check()
    stage = "enable_follower"
    try:
        power.enable()
        epoch = perf_counter()
        print("正在采集。确认夹爪已空，带离桌面并留出回程空间，再按 Q 保存并回位。", flush=True)
        print("Esc 停止且不回位。异常时直接急停或断电。", flush=True)
        while True:
            if stop_key := read_stop_key():
                break
            start, wall_time = perf_counter(), time_ns()
            stage = "read_leader"
            samples = read_leader(leader)
            leader_time = perf_counter()
            polled_joint, poll_ms = None, None
            stage = "read_follower"
            if frame_index % 2 == 0:
                # The adapter can return cached positions. Check fresh CAN replies in turn.
                polled_joint = JOINTS[(frame_index // 2) % len(JOINTS)]
                poll_start = perf_counter()
                read_follower_joint(follower, polled_joint)
                poll_ms = (perf_counter() - poll_start) * 1000
            observation = follower.get_observation()
            follower_time = perf_counter()
            stage = "read_cameras"
            images, image_times = camera_frames(cameras, overhead_placeholder=black_overhead)
            angles = {joint: sample.filtered_deg for joint, sample in samples.items()}
            action = {}
            feedback = {}
            stage = "check_follower_feedback"
            for joint in JOINTS:
                fields = (observation[f"{joint}.{suffix}"] for suffix in ("pos", "vel", "torque"))
                state = follower.motors[joint].get_state()
                if state is None or not all(map(math.isfinite, fields)):
                    raise RuntimeError(f"从臂 {joint_label(joint)} 未读到有效反馈")
                feedback[joint] = state
                displacement[joint] += math.remainder(angles[joint] - previous[joint], 360.0)
                action[f"{joint}.pos"] = origin[joint] / direction[joint] + displacement[joint]
            if stop_key := read_stop_key():
                break
            stage = "recording_queue"
            writer.check()
            stage = "send_action"
            sent = follower.send_action(action)
            send_time = perf_counter()
            # Convert the applied target back to action coordinates so replay maps it only once.
            applied_action = {f"{joint}.pos": sent[f"{joint}.pos"] / direction[joint] for joint in JOINTS}
            overhead_time = image_times["overhead"]
            missing = overhead_time is None
            if missing != overhead_missing:
                if missing:
                    print("俯视相机断流，正在用黑帧占位。主从跟随和录制继续。", flush=True)
                else:
                    print("俯视相机恢复，已接回实际画面。", flush=True)
                overhead_missing = missing
            missing_frames += int(missing)
            telemetry_row = [
                frame_index, wall_time, start - epoch, leader_time - epoch, follower_time - epoch,
                send_time - epoch, image_times["wrist"] - epoch,
                None if missing else overhead_time - epoch, int(not missing),
                *(samples[joint].raw_deg for joint in JOINTS),
                *(angles[joint] for joint in JOINTS),
                *(int(samples[joint].reliable) for joint in JOINTS),
                *(action[f"{joint}.pos"] * direction[joint] for joint in JOINTS),
                *(sent[f"{joint}.pos"] for joint in JOINTS),
                *(observation[f"{joint}.{suffix}"] for suffix in ("pos", "vel", "torque") for joint in JOINTS),
                joint_label(polled_joint) if polled_joint is not None else None, poll_ms,
                *(getattr(feedback[joint], field) for field in ("status_code", "arbitration_id", "t_mos")
                  for joint in JOINTS),
            ]
            stage = "recording_queue"
            writer.submit(observation | images, applied_action, telemetry_row)
            previous = angles
            frame_index += 1
            sleep(max(0.0, PERIOD - (perf_counter() - start)))
    except BaseException as exc:
        exc.recording_context = {"stage": stage, "frame_index": frame_index, "detected_unix_ns": time_ns()}
        raise
    finally:
        try:
            power.disable()
        except BaseException as exc:
            exc.recording_context = {"stage": "disable_follower", "frame_index": frame_index,
                                     "detected_unix_ns": time_ns()}
            raise
    return Take(frame_index, missing_frames, origin, stop_key)


def record_session(leader, power, cameras, task):
    follower = power.follower
    run_name = datetime.now().strftime("capture-%Y%m%d-%H%M%S-%f")
    root = Path(__file__).resolve().parent / "recordings" / run_name
    reference = None
    pose_issue = None
    retry_return = False
    last_saved_trace = None
    while True:
        if pose_issue is None:
            if reference is None:
                print("摆好能稳住的从臂起始位。第一条开录时会记住它，本批每条按 Q 保存后都回到这里。", flush=True)
            print("放稳两臂并合回白色指环。按 Enter 准备一条，输入 Q 后回车结束。", flush=True)
        else:
            next_action = ("清空回程范围后按 Enter 重试自动回位"
                           if retry_return else "放稳两臂后按 Enter 复查起始姿态")
            print(f"{pose_issue} 从臂保持失能。{next_action}，输入 Q 后回车结束。", flush=True)
        if input().strip():
            return
        if retry_return:
            pose_issue = return_for_next_take(power, reference, last_saved_trace)
            retry_return = pose_issue is not None
            continue
        pose_issue = None
        dataset = open_dataset(root, follower, cameras, FPS)
        episode_index = dataset.num_episodes
        trace = root / f"telemetry-episode-{episode_index:06d}.csv"
        partial = trace.with_suffix(".partial.csv")
        stage = "prepare_or_record"
        try:
            with partial.open("w", encoding="utf-8", newline="") as file:
                with EpisodeWriter(dataset, task, csv.writer(file), TELEMETRY_COLUMNS, flush=file.flush) as writer:
                    take = follow(leader, power, cameras, writer, reference)
            if take.frames:
                stage = "save_episode"
                if take.stop_key == "q":
                    print("已停止采集。正在保存本条，随后自动回位。请等回位结束再复位物体。", flush=True)
                else:
                    print("已停止采集，从臂已失能。正在保存已录内容。", flush=True)
                dataset.save_episode()
        except StartPoseNotReady as exc:
            partial.unlink(missing_ok=True)
            pose_issue = str(exc)
            range_blocked = isinstance(exc, StartPoseOutOfRange)
            retry_return = reference is not None and not range_blocked
            if range_blocked:
                directory = trace.with_name(trace.stem + "-start-check")
                try:
                    report = describe_fault(power, exc, "check_start_pose", episode_index)
                    write_fault_log(directory, report)
                    print(f"已保存起始姿态检查日志。\n{directory / 'fault.json'}", flush=True)
                except Exception as archive_error:
                    print(f"写入起始姿态检查日志失败，{archive_error}", file=sys.stderr, flush=True)
            continue
        except BaseException as exc:
            try:
                report = describe_fault(power, exc, stage, episode_index)
                if stage == "save_episode":
                    directory = trace.with_name(trace.stem + "-interrupted")
                    report["artifacts"] = {"partial_telemetry": f"../{partial.name}"}
                    write_fault_log(directory, report)
                else:
                    directory = preserve_interrupted_episode(dataset, partial, report, keep_video=not power.needs_disable)
                print(f"已保存故障日志和可保留的诊断文件。\n{directory}", flush=True)
            except Exception as archive_error:
                print(f"写入故障日志失败：{archive_error}", file=sys.stderr, flush=True)
            finally:
                if stage != "save_episode":
                    dataset.clear_episode_buffer(delete_images=False)
            raise
        finally:
            dataset.finalize()
        if not take.frames:
            partial.unlink(missing_ok=True)
            return
        partial.replace(trace)
        last_saved_trace = trace
        print(f"第 {episode_index + 1} 条已保存，共 {take.frames} 帧。\n{root}", flush=True)
        if take.missing_frames:
            print(f"俯视缺失 {take.missing_frames} 帧，已用黑帧占位并在 CSV 中标记。", flush=True)
        if take.stop_key != "q":
            return
        if reference is None:
            reference = take.start_pose
        pose_issue = return_for_next_take(power, reference, trace)
        retry_return = pose_issue is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check both arms and cameras without enabling the follower")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        if not args.check:
            if not sys.stdin.isatty():
                raise RuntimeError("请在本机终端运行或双击 rebot-follow.cmd，以便随时按 Esc 停止。")
            print("B601-RS + reBot Arm 102  |  LeRobot 双相机采集")
            print("请关闭其他机械臂控制程序，清空从臂活动范围，并确认能立即急停或断电。")
            task = input("输入本次任务描述，例如「拿起蓝色餐巾纸，把它放进黑色框里」\n").strip()
            if not task:
                return 0
        with connect_disabled() as (leader, power), connect_cameras(optional_overhead=not args.check) as cameras:
            if args.check:
                lead, follow_pos = read_leader(leader), read_follower(power.follower)
                for index, joint in enumerate(JOINTS, 1):
                    print(f"J{index}  主臂 {lead[joint].filtered_deg:+.2f}°  从臂 {follow_pos[joint]:+.2f}°")
                images, _ = camera_frames(cameras)
                for name, image in images.items():
                    print(f"{name} 相机 {image.shape[1]}×{image.shape[0]}")
                print("两臂七轴和两路相机通过检查，从臂保持失能。")
            else:
                record_session(leader, power, cameras, task)
        return 0
    except (KeyboardInterrupt, EOFError):
        return 0
    except Exception as exc:
        print(f"采集已退出：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
