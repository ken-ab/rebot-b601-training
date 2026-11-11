"""Seeed two-stage return and best-effort cleanup with explicit outcomes."""

import math
from types import FunctionType, MethodType, SimpleNamespace
from time import sleep
from time import perf_counter

from b601_preflight import (
    DeviceCommunicationError, DriveFaultError, JOINTS, close_handles,
    disable_motors, error_record, read_follower,
)


class ReturnTemperatureError(RuntimeError):
    pass


class ReturnInterrupted(KeyboardInterrupt):
    pass


def stop_follower(follower, last, max_delta_deg, fps, *, return_to_zero, reason, stop_key):
    """Use official safe_zero directly so its failure cannot be swallowed.

    Homing is skipped after a CAN/drive fault or explicit emergency stop.
    SDK cleanup is attempted independently for every axis even after failure.
    """
    result = {
        "reason": reason,
        "return_to_zero": {
            "attempted": False, "sequence_completed": False,
            "result": f"skipped_{reason}",
        },
    }
    returning = result["return_to_zero"]
    native_send = follower.send_action
    previous = {f"{joint}.pos": float(value) for joint, value in zip(JOINTS, last, strict=True)}
    interval = 1 / fps
    started = perf_counter()
    returning["steps"] = []

    def record_send(action, phase):
        accepted = native_send(action)
        previous.update(accepted)
        observation = follower.get_observation()
        returning["steps"].append({
            "seconds": round(perf_counter() - started, 4), "phase": phase,
            "target_motor_deg": [previous[f"{joint}.pos"] for joint in JOINTS],
            "measured_motor_deg": [observation[f"{joint}.pos"] for joint in JOINTS],
            "gripper_torque_nm": observation["gripper.torque"],
        })
        return accepted

    def check_return_state():
        if stop_key() == "\x1b":
            raise ReturnInterrupted("Escape during return")
        temperatures = follower._read_motor_temperatures()
        returning["last_temperature_c"] = temperatures
        threshold = follower.config.temp_emergency_disable_threshold_c
        hot = {joint: temp for joint, temp in temperatures.items() if temp > threshold}
        if hot:
            follower._emergency_disable_requested = True
            raise ReturnTemperatureError(f"Official {threshold:g} C return limit exceeded: {hot}")

    def send_return_action(action):
        action = dict(action)
        # RS gripper is host-side PD with motor kp=0. If native safe_zero
        # omits J7, its last torque otherwise persists throughout the return.
        action.setdefault("gripper.pos", previous["gripper.pos"]
                          / follower.config.joint_directions["gripper"])
        targets = {}
        for name, value in action.items():
            joint = name.removesuffix(".pos")
            low, high = follower.config.joint_limits[joint]
            targets[name] = max(low, min(high, value * follower.config.joint_directions[joint]))
        starts = {name: previous[name] for name in targets}
        steps = max(1, math.ceil(max(abs(targets[name] - starts[name]) for name in targets)
                                 / max_delta_deg))
        accepted = None
        for step in range(1, steps + 1):
            # safe_zero skips the normal send_action temperature check. Each
            # inserted command must still observe its official emergency line.
            check_return_state()
            accepted = record_send({
                name: (starts[name] + (target - starts[name]) * step / steps)
                / follower.config.joint_directions[name.removesuffix(".pos")]
                for name, target in targets.items()
            }, "official_return")
            if step < steps:
                sleep(interval)
        return accepted

    def refresh_while_waiting(duration):
        # Preserve the SDK's exact waits, including its final 2 s settling.
        # Refresh all held targets at its native step interval; single CAN
        # thread, unchanged gains/torque/temperature rules and no new limit.
        native_interval = 0.02
        frames = max(1, math.ceil(duration / native_interval))
        for _ in range(frames):
            check_return_state()
            record_send({name: value / follower.config.joint_directions[name.removesuffix(".pos")]
                         for name, value in previous.items()}, "return_hold")
            sleep(duration / frames)

    def run_official_zero():
        method = follower.safe_zero
        if not isinstance(method, MethodType):
            # Permit dependency injection of SDK failures in offline tests.
            return method(exit_on_complete=False)
        function = method.__func__
        # Local function namespace only: do not monkeypatch process-global
        # time.sleep, which is also used by the camera and inference threads.
        namespace = {**function.__globals__, "time": SimpleNamespace(sleep=refresh_while_waiting)}
        refreshed = FunctionType(function.__code__, namespace, function.__name__,
                                 function.__defaults__, function.__closure__)
        refreshed.__kwdefaults__ = function.__kwdefaults__
        return refreshed(follower, exit_on_complete=False)

    try:
        if return_to_zero:
            returning["attempted"] = True
            check_return_state()
            # Fresh register replies avoid starting a return from cached state.
            returning["before_motor_deg"] = read_follower(follower)
            follower.send_action = send_return_action
            run_official_zero()
            if follower._emergency_disable_requested:
                returning["result"] = "aborted_official_emergency_temperature"
            else:
                returning["sequence_completed"] = True
                returning["result"] = "command_sequence_completed"
                returning["after_motor_deg"] = read_follower(follower)
                returning["arm_error_from_zero_deg"] = {
                    joint: angle for joint, angle in returning["after_motor_deg"].items()
                    if joint != "gripper"
                }
                # Record encoders without inventing a new final pose tolerance.
                returning["physical_zero_verified"] = None
    except (DeviceCommunicationError, DriveFaultError) as error:
        returning.update(result="aborted_device_fault", error=error_record(error))
    except ReturnTemperatureError as error:
        returning.update(result="aborted_official_emergency_temperature", error=error_record(error))
    except KeyboardInterrupt as error:
        returning.update(result="interrupted", error=error_record(error))
    except Exception as error:
        returning.update(result="failed", error=error_record(error))
    finally:
        follower.send_action = native_send
        result["disable"] = disable_motors(follower)
        disabled = {entry["joint"]: entry["sdk_call_completed"]
                    for entry in result["disable"]["motors"]}
        result["close"] = close_handles(follower, disabled)

    result["completed_without_errors"] = (
        (not return_to_zero or returning["result"] == "command_sequence_completed")
        and result["disable"]["all_sdk_calls_completed"]
        and result["close"]["all_calls_completed"]
    )
    return result
