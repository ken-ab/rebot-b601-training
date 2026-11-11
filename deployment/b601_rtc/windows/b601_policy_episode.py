"""B601 VLA execution with Seeed defaults and the user's two motion limits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from b601_action_timeline import (
    ActionTimeline, RTC_KEY, RTC_PROTOCOL, delay_ticks, elapsed_tick, require_rtc,
)
from b601_task_completion import PlacementDetector

from b601_preflight import (
    DeviceCommunicationError, DriveFaultError, JOINTS, configured_limits,
    device_call, error_record, feedback_only, official_config, read_follower,
)
from b601_stop import stop_follower
from rebot_capture import connect_cameras
from rebot_pi05 import (
    ACTION_SHAPE, B601WebsocketClientPolicy, HOST, JPEG_RGB, METADATA, PROMPT,
    TRANSPORT_KEY, check_health, encode_request, prepare_observation,
)

MAX_JOINT_DELTA_DEG = 0.3
START_TOLERANCE_DEG = 5.0


def motor_vector(values):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(JOINTS),) or not np.isfinite(vector).all():
        raise ValueError("需要七个有限的电机角度。")
    return vector


def check_start(actual, reference):
    error = np.abs(motor_vector(actual) - motor_vector(reference))
    if np.any(error > START_TOLERANCE_DEG):
        axis = int(np.argmax(error))
        raise RuntimeError(
            f"J{axis + 1} 起始偏差 {error[axis]:.3f}° 超过 {START_TOLERANCE_DEG}°。"
        )


def next_target(last, proposed, lower, upper):
    last = motor_vector(last)
    proposed = np.clip(motor_vector(proposed), lower, upper)
    return np.clip(
        last + np.clip(proposed - last, -MAX_JOINT_DELTA_DEG, MAX_JOINT_DELTA_DEG),
        lower, upper,
    )


def camera_images(cameras):
    return {
        f"observation.images.{name}": camera.async_read()
        for name, camera in cameras.items()
    }


def measured_positions(follower):
    observation = follower.get_observation()
    return motor_vector([observation[f"{joint}.pos"] for joint in JOINTS])


def infer(client, positions, images, rtc=None):
    observation = {
        **images, "observation.state": positions.copy(), "prompt": PROMPT,
    }
    started = perf_counter()
    request = encode_request(prepare_observation(observation))
    if rtc is not None:
        request[RTC_KEY] = rtc
    reply = client.infer(request)
    actions = np.asarray(reply["actions"])
    if actions.shape != ACTION_SHAPE or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
        raise ValueError("模型必须返回有限的 25×7 动作。")
    if rtc is not None:
        response = reply.get(RTC_KEY, {})
        if (response.get("protocol") != RTC_PROTOCOL or response.get("request_id") != rtc["request_id"]
                or response.get("observation_tick") != rtc["observation_tick"]):
            raise ValueError("RTC 响应与本次观测的时间编号不一致。")
    return actions.astype(float), (perf_counter() - started) * 1000


def send_motor_target(follower, target, direction):
    action = dict(zip(METADATA["action_names"], target / direction, strict=True))
    accepted = follower.send_action(action)
    # Seeed send_action returns accepted motor degrees, including the gripper scale.
    return motor_vector([accepted[name] for name in METADATA["action_names"]])


def infer_and_observe(client, positions, images, rtc, detector, seconds):
    # Vision/JPEG/RPC all run off the 25 Hz CAN thread.
    started = perf_counter()
    completion = (detector.update(images["observation.images.overhead"], seconds)
                  if "observation.images.overhead" in images else {"success": False})
    if completion["success"]:
        return (None, 0.0), completion
    actions, _ = infer(client, positions, images, rtc)
    return (actions, (perf_counter() - started) * 1000), completion


def stop_key():
    import msvcrt
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
        elif key == "\x1b" or key.lower() == "q":
            return key
    return None


def run_episode(follower, cameras, client, reference, seconds, execute, report):
    direction = np.array([follower.config.joint_directions[j] for j in JOINTS])
    lower, upper = np.array([follower.config.joint_limits[j] for j in JOINTS]).T
    positions = motor_vector(list(read_follower(follower).values()))
    initial_images = camera_images(cameras)
    chunk, elapsed = infer(client, positions, initial_images)
    report["request_ms"].append(round(elapsed, 2))
    positions = motor_vector(list(read_follower(follower).values()))
    check_start(positions, reference)
    last = np.clip(positions, lower, upper)
    enable_attempted = False
    configured = False
    stop_reason = "normal_exit"
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    pending_context = None
    timeline = ActionTimeline(direction, lower, upper, MAX_JOINT_DELTA_DEG)
    timeline.install(chunk, 0, 0)
    detector = PlacementDetector()
    if "observation.images.overhead" in initial_images:
        detector.update(initial_images["observation.images.overhead"], 0)
    report["requests"] = [{"request_id": 0, "observation_tick": 0,
        "roundtrip_ms": elapsed, "raw_actions_adapter": chunk.tolist(), "initial": True}]
    report["completion_observations"] = []
    latency_ms = elapsed
    request_id = 0
    try:
        if execute:
            enable_attempted = True
            report["enable_attempted"] = True
            device_call("Seeed configure", follower.configure)
            configured = True
            report["physical_trial_started"] = True
            last = send_motor_target(follower, last, direction)
            print("已使能，开始整臂 VLA 抓放。Q 正常回零退出，Esc 立即失能。", flush=True)
        started = perf_counter()
        while perf_counter() - started < seconds:
            key = stop_key()
            if key is not None:
                emergency = key == "\x1b"
                stop_reason = "operator_emergency" if emergency else "normal_exit"
                report["end_reason"] = "escape" if emergency else "operator_q"
                break
            observation = follower.get_observation()
            positions = motor_vector([observation[f"{joint}.pos"] for joint in JOINTS])
            tick = elapsed_tick(perf_counter(), started, METADATA["fps"])
            completed = pending.result() if pending is not None and pending.done() else None
            if completed is not None and completed[1]["success"]:
                completion = completed[1]
                report["completion_observations"].append({
                    "seconds": pending_context["observation_tick"] / METADATA["fps"], **completion})
                report["end_reason"] = "placement_detected"
                stop_reason = "task_completed"
                report["success_evidence"] = completion
                print("检测到纸巾已入框、夹爪撤离且纸巾稳定，结束模型动作并回零。", flush=True)
                break
            if (completed is not None
                    and tick >= pending_context["observation_tick"] + pending_context["inference_delay"]):
                (chunk, elapsed), completion = completed
                report["completion_observations"].append({
                    "seconds": pending_context["observation_tick"] / METADATA["fps"], **completion})
                report["request_ms"].append(round(elapsed, 2))
                latency_ms = elapsed
                origin = pending_context["observation_tick"]
                # Keep the observation's original clock. A response never
                # rewinds to action 0 after 6-10 ticks have already elapsed.
                timeline.install(chunk, origin, pending_context["request_id"])
                report["requests"].append({
                    **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in pending_context.items()},
                    "arrival_tick": tick, "first_usable_index": tick - origin,
                    "roundtrip_ms": round(elapsed, 2), "raw_actions_adapter": chunk.tolist(),
                })
                pending = None
            if pending is None:
                images = camera_images(cameras)
                now = perf_counter()
                tick = elapsed_tick(now, started, METADATA["fps"])
                previous, overlap = timeline.prefix(tick, last)
                request_id += 1
                delay = min(25, delay_ticks(latency_ms, METADATA["fps"]))
                pending_context = {"protocol": RTC_PROTOCOL, "request_id": request_id,
                    "observation_tick": tick, "inference_delay": delay,
                    # If the old chunk expires during inference, its tail
                    # becomes a committed hold; include that hold in RTC.
                    "prefix_horizon": max(overlap, delay), "previous_actions": previous}
                pending = pool.submit(infer_and_observe, client, positions.copy(), images,
                                      pending_context, detector, now - started)
            tick = elapsed_tick(perf_counter(), started, METADATA["fps"])
            proposed, index = timeline.proposed(tick, last)
            command = next_target(last, proposed, lower, upper)
            last = send_motor_target(follower, command, direction) if execute else command
            report["steps"].append({
                "seconds": round(perf_counter() - started, 4),
                "measured_motor_deg": positions.tolist(),
                "target_motor_deg": last.tolist(),
                "proposed_motor_deg": proposed.tolist(), "controller_tick": tick,
                "action_index": index, "source_request_id": timeline.request_id,
                "gripper_torque_nm": observation.get("gripper.torque"),
            })
            if execute and len(report["steps"]) % METADATA["fps"] == 0:
                print(f"抓放 {perf_counter() - started:.1f}s，反馈电机角 "
                      f"{np.round(positions, 2).tolist()}", flush=True)
            # Use the model's 25 Hz clock. A late tick resumes without a catch-up burst.
            period = 1 / METADATA["fps"]
            deadline = started + (elapsed_tick(perf_counter(), started, METADATA["fps"]) + 1) * period
            sleep(max(0, deadline - perf_counter()))
        else:
            report["end_reason"] = "duration_completed"
    except (DeviceCommunicationError, DriveFaultError):
        stop_reason = "device_fault"
        report["end_reason"] = "device_fault"
        raise
    except KeyboardInterrupt as error:
        report["end_reason"] = "official_overheat" if str(error) == "Overheat detected" else "operator_ctrl_c"
        stop_reason = report["end_reason"]
        if str(error):
            raise
    except BaseException:
        stop_reason = "runtime_error"
        report["end_reason"] = "runtime_error"
        raise
    finally:
        try:
            if enable_attempted:
                if not configured:
                    stop_reason = "configure_failed"
                print(f"结束模型动作，处理停止：{stop_reason}", flush=True)
                report["stop"] = stop_follower(
                    follower, last, MAX_JOINT_DELTA_DEG, METADATA["fps"],
                    return_to_zero=configured and stop_reason not in ("device_fault", "operator_emergency"),
                    reason=stop_reason, stop_key=stop_key,
                )
                report["stop_failed"] = not report["stop"]["completed_without_errors"]
        finally:
            if pending is not None:
                pending.cancel()
            try:
                client.close()
            except (Exception, KeyboardInterrupt) as error:
                report["client_close_error"] = error_record(error)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)


def configuration_report():
    config = official_config()
    return {
        "source": "installed Seeed B601-RS Follower defaults",
        "custom_motion_limits": {
            "max_joint_delta_motor_deg_per_vla_step": MAX_JOINT_DELTA_DEG,
            "start_pose_tolerance_motor_deg": START_TOLERANCE_DEG,
        },
        "execution": {"action_clock": "observation_timestamp", "rtc": RTC_PROTOCOL,
                      "success_stop": "blue_tissue_in_goal_and_gripper_withdrawn",
                      "homing_gripper": "continuous_official_host_pd"},
        "joint_limits_motor_deg": config.joint_limits,
        "joint_directions": config.joint_directions,
        "mit_kp": config.mit_kp,
        "mit_kd": config.mit_kd,
        "gravity_compensation": config.gravity_compensation,
        "gripper_torque_limit_nm": config.gripper_mit_torque_limit,
        "gripper_hold_torque_limit_nm": config.gripper_mit_hold_torque_limit,
        "temperature_c": {
            "alarm": config.temp_alarm_threshold_c,
            "overheat": config.temp_overheat_threshold_c,
            "emergency_disable": config.temp_emergency_disable_threshold_c,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--start-pose", type=Path, default=Path(__file__).with_name("b601-start-pose.json"))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(configuration_report(), ensure_ascii=False, indent=2))
        return 0
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds 必须为正数。")
    setup = json.loads(args.start_pose.read_text(encoding="utf-8"))
    reference = motor_vector(setup["start_motor_deg"])
    report = {
        "configuration": configuration_report(), "physical_trial_started": False,
        "requested_seconds": args.seconds,
        "request_ms": [], "steps": [], "start_reference_motor_deg": reference.tolist(),
    }
    try:
        check_health(args.port)
        with feedback_only() as follower, connect_cameras() as cameras:
            report["official_joint_limits"] = configured_limits(follower)
            client = B601WebsocketClientPolicy(HOST, args.port)
            try:
                metadata = client.get_server_metadata()
                if metadata.get(TRANSPORT_KEY) != JPEG_RGB or any(metadata.get(k) != v for k, v in METADATA.items()):
                    raise ValueError("服务端的机器人、动作单位或图像协议与训练配套不一致。")
                require_rtc(metadata)
                run_episode(follower, cameras, client, reference, args.seconds, args.execute, report)
            finally:
                client.close()
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = str(error) or type(error).__name__
        report["error_detail"] = error_record(error)
    root = Path(__file__).with_name("inference-evals")
    root.mkdir(exist_ok=True)
    path = root / f"b601-policy-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**{key: value for key, value in report.items()
                         if key not in ("steps", "requests", "request_ms", "completion_observations", "stop")},
                      "steps_recorded": len(report["steps"]),
                      "requests_recorded": len(report.get("requests", [])),
                      "stop_completed": report.get("stop", {}).get("completed_without_errors")}, ensure_ascii=False))
    print(str(path))
    return int("error" in report or report.get("stop_failed", False))


if __name__ == "__main__":
    raise SystemExit(main())

