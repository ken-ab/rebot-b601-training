"""Windows PCAN access using Seeed RS settings and explicit SDK I/O failures."""

import argparse
from contextlib import contextmanager
import ctypes
import json
import logging
import math
import os
from pathlib import Path

from motorbridge.errors import MotorBridgeError
from lerobot_robot_seeed_b601.seeed_b601_rs_follower import SeeedB601RSFollower

from rebot_pi05 import JOINTS as ACTION_NAMES

JOINTS = tuple(name.removesuffix(".pos") for name in ACTION_NAMES)
PCAN_DIR = Path(__file__).resolve().parents[2] / "具身/B601-RS/runtime/pcan/x64"


class DeviceCommunicationError(RuntimeError):
    """A device SDK call failed or returned unusable feedback."""


class DriveFaultError(RuntimeError):
    """MotorBridge reported a RobStride fault status."""

    def __init__(self, joint, status_code):
        self.joint = joint
        self.status_code = int(status_code)
        super().__init__(f"{joint} RobStride status_code=0x{self.status_code:02x}")


def device_call(operation, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (MotorBridgeError, OSError) as error:
        raise DeviceCommunicationError(f"{operation}: {error}") from error


def official_config():
    from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
    return SeeedB601RSFollowerConfig(
        port="can0@1000000", can_adapter="socketcan", id="follower1",
    )


class CheckedSeeedB601RSFollower(SeeedB601RSFollower):
    """Keep Seeed's motion methods while propagating feedback failures."""

    def _states(self):
        for joint in JOINTS:
            device_call(f"{joint} request_feedback", self.motors[joint].request_feedback)
        device_call("poll_feedback_once", self.bus.poll_feedback_once)
        states = {}
        for joint in JOINTS:
            state = device_call(f"{joint} get_state", self.motors[joint].get_state)
            if state is None:
                raise DeviceCommunicationError(f"{joint} has no feedback")
            if state.status_code:
                raise DriveFaultError(joint, state.status_code)
            if not all(math.isfinite(value) for value in (state.pos, state.vel, state.torq, state.t_mos)):
                raise DeviceCommunicationError(f"{joint} returned non-finite feedback")
            states[joint] = state
        return states

    def get_observation(self):
        states = self._states()
        observation = {}
        for joint, state in states.items():
            observation.update({
                f"{joint}.pos": math.degrees(state.pos),
                f"{joint}.vel": math.degrees(state.vel),
                f"{joint}.torque": state.torq,
            })
        observation.update({name: camera.async_read() for name, camera in self.cameras.items()})
        return observation

    def _read_motor_temperatures(self):
        return {joint: state.t_mos for joint, state in self._states().items()}

    def send_action(self, action):
        # A register round-trip detects a lost link even when get_state has cached data.
        # Rotate through the seven motors, using MotorBridge's default request timeout.
        index = getattr(self, "_feedback_probe_index", 0)
        joint = JOINTS[index]
        _, host_id = self.config.motor_can_ids[joint]
        position = device_call(
            f"{joint} mechPos",
            self.motors[joint].robstride_get_param_f32_host_id, 0x7019, host_id,
        )
        if not math.isfinite(position):
            raise DeviceCommunicationError(f"{joint} returned non-finite mechPos")
        self._feedback_probe_index = (index + 1) % len(JOINTS)
        return device_call("Seeed send_action", super().send_action, action)


def configured_limits(follower):
    return {
        joint: {
            "can_id": follower.config.motor_can_ids[joint][0],
            "direction": follower.config.joint_directions[joint],
            "motor_deg": list(follower.config.joint_limits[joint]),
        }
        for joint in JOINTS
    }


def read_follower(follower):
    result = {}
    for joint in JOINTS:
        _, host_id = follower.config.motor_can_ids[joint]
        value = device_call(
            f"{joint} mechPos",
            follower.motors[joint].robstride_get_param_f32_host_id, 0x7019, host_id,
        )
        angle = math.degrees(value)
        if not math.isfinite(angle):
            raise DeviceCommunicationError(f"{joint} returned non-finite mechPos")
        result[joint] = angle
    return result


def error_record(error):
    return {"type": type(error).__name__, "message": str(error)}


def enable_feedback_reporting(follower):
    """Request SDK status reports without enabling torque or changing limits."""
    follower._active_report_joints = set()
    for joint in JOINTS:
        follower._active_report_joints.add(joint)
        device_call(f"{joint} enable status reports",
                    follower.motors[joint].robstride_set_active_report, True)
    # The official register transactions give the background receiver a full
    # bus round-trip before the first temperature/status cache is consumed.
    read_follower(follower)


def disable_motors(follower):
    results = []
    for joint in JOINTS:
        result = {"joint": joint, "sdk_call_completed": False}
        try:
            follower.motors[joint].disable()
        except (Exception, KeyboardInterrupt) as error:
            result["error"] = error_record(error)
        else:
            result["sdk_call_completed"] = True
        results.append(result)
    return {
        "motors": results,
        "all_sdk_calls_completed": all(result["sdk_call_completed"] for result in results),
        "physical_disable_confirmation": None,
    }


def close_handles(follower, disabled=None):
    results = []
    disabled = disabled or {}
    for joint, motor in follower.motors.items():
        result = {"joint": joint}
        if joint in getattr(follower, "_active_report_joints", ()):
            try:
                motor.robstride_set_active_report(False)
            except (Exception, KeyboardInterrupt) as error:
                result["stop_status_reports"] = {"completed": False, "error": error_record(error)}
            else:
                result["stop_status_reports"] = {"completed": True}
                follower._active_report_joints.remove(joint)
        operations = ["close"]
        if disabled.get(joint):
            operations.insert(0, "clear_error")
        for operation in operations:
            try:
                getattr(motor, operation)()
            except (Exception, KeyboardInterrupt) as error:
                result[operation] = {"completed": False, "error": error_record(error)}
            else:
                result[operation] = {"completed": True}
        results.append(result)
    bus_result = {"completed": follower.bus is None}
    if follower.bus is not None:
        try:
            follower.bus.close()
        except (Exception, KeyboardInterrupt) as error:
            bus_result["error"] = error_record(error)
        else:
            follower.bus = None
            bus_result["completed"] = True
    return {
        "motors": results, "bus": bus_result,
        "all_calls_completed": bus_result["completed"] and all(
            operation["completed"] for result in results
            for name, operation in result.items() if name != "joint"
        ),
    }


@contextmanager
def feedback_only():
    os.environ["PATH"] = str(PCAN_DIR) + os.pathsep + os.environ.get("PATH", "")
    ctypes.WinDLL(str(PCAN_DIR / "PCANBasic.dll"), winmode=0)
    from motorbridge import Controller

    follower = CheckedSeeedB601RSFollower(official_config())
    if set(follower.calibration) != set(JOINTS):
        raise RuntimeError("找不到 follower1 的七轴标定。")
    if any(follower.calibration[j].id != follower.config.motor_can_ids[j][0] for j in JOINTS):
        raise RuntimeError("标定文件与 CAN ID 不一致。")
    follower.bus = Controller(channel=follower.config.port)
    try:
        follower._add_motors_to_bus()
        enable_feedback_reporting(follower)
        yield follower
    finally:
        if follower.bus is not None:
            cleanup = close_handles(follower)
            if not cleanup["all_calls_completed"]:
                logging.error("B601 handle cleanup incomplete: %s", cleanup)


def probe(rounds=1):
    with feedback_only() as follower:
        return {
            "configured_limits": configured_limits(follower),
            "feedback_rounds": [read_follower(follower) for _ in range(rounds)],
            "calibration": str(follower.calibration_fpath),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(probe(args.rounds), ensure_ascii=False, indent=2))

