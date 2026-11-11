"""Restore the saved start pose using the existing delta and Seeed settings."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from b601_policy_episode import (
    MAX_JOINT_DELTA_DEG, METADATA, check_start, measured_positions,
    motor_vector, next_target, send_motor_target, stop_key,
)
from b601_preflight import (
    DeviceCommunicationError, DriveFaultError, JOINTS, device_call,
    error_record, feedback_only, read_follower,
)
from b601_stop import stop_follower


def restore(follower, reference, report):
    direction = np.array([follower.config.joint_directions[j] for j in JOINTS])
    lower, upper = np.array([follower.config.joint_limits[j] for j in JOINTS]).T
    last = np.clip(motor_vector(list(read_follower(follower).values())), lower, upper)
    goal = np.clip(reference, lower, upper)
    report["before_motor_deg"] = last.tolist()
    report["goal_motor_deg"] = goal.tolist()
    frames = math.ceil(float(np.max(np.abs(goal - last))) / MAX_JOINT_DELTA_DEG)
    attempted = configured = False
    emergency = device_fault = False
    try:
        attempted = True
        device_call("Seeed configure", follower.configure)
        configured = True
        last = send_motor_target(follower, last, direction)
        deadline = perf_counter()
        for frame in range(frames):
            key = stop_key()
            if key:
                emergency = key == "\x1b"
                raise KeyboardInterrupt("Operator stopped start-pose restoration")
            measured = measured_positions(follower)
            last = send_motor_target(follower, next_target(last, goal, lower, upper), direction)
            report["steps"].append({"measured_motor_deg": measured.tolist(),
                                    "target_motor_deg": last.tolist()})
            if (frame + 1) % METADATA["fps"] == 0:
                print(f"恢复起点 {frame + 1}/{frames}，夹爪电机角 {measured[-1]:.2f}°", flush=True)
            period = 1 / METADATA["fps"]
            deadline += max(1, int((perf_counter() - deadline) / period) + 1) * period
            sleep(max(0, deadline - perf_counter()))
        actual = motor_vector(list(read_follower(follower).values()))
        report["after_motor_deg"] = actual.tolist()
        check_start(actual, reference)
        report["restored"] = True
    except (DeviceCommunicationError, DriveFaultError):
        device_fault = True
        raise
    finally:
        if attempted:
            report["stop"] = stop_follower(
                follower, last, MAX_JOINT_DELTA_DEG, METADATA["fps"],
                return_to_zero=(configured and not report.get("restored")
                                and not emergency and not device_fault),
                reason="start_pose_restored" if report.get("restored") else "restoration_interrupted",
                stop_key=stop_key,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    reference = motor_vector(json.loads((root / "b601-start-pose.json").read_text(encoding="utf-8"))["start_motor_deg"])
    report = {"restored": False, "steps": []}
    try:
        with feedback_only() as follower:
            restore(follower, reference, report)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = error_record(error)
    path = root / "inference-evals" / f"b601-start-restore-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "steps"}, ensure_ascii=False))
    print(path)
    return int(not report["restored"] or not report.get("stop", {}).get("completed_without_errors", False))


if __name__ == "__main__":
    raise SystemExit(main())
