"""B601 VLA execution with Seeed defaults and the user's two motion limits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from b601_jpeg_transport import CAMERA_KEYS
from b601_action_timeline import (
    ActionTimeline, DelayEstimator, RTC_KEY, RTC_PROTOCOL, delay_ticks, elapsed_tick,
    require_rtc, rtc_context,
)
from b601_execution_config import DEFAULT_CONFIG, load_execution_config, motor_deltas
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


def next_target(last, proposed, lower, upper, max_delta=MAX_JOINT_DELTA_DEG):
    last = motor_vector(last)
    proposed = np.clip(motor_vector(proposed), lower, upper)
    return np.clip(
        last + np.clip(proposed - last, -np.asarray(max_delta), np.asarray(max_delta)),
        lower, upper,
    )


def camera_images(cameras, *, request_id=None):
    return {
        f"observation.images.{name}": camera.async_read()
        for name, camera in cameras.items()
    }


def measured_positions(follower):
    observation = follower.get_observation()
    return motor_vector([observation[f"{joint}.pos"] for joint in JOINTS])


def infer(client, positions, images, rtc=None, timings=None):
    observation = {
        **images, "observation.state": positions.copy(), "prompt": PROMPT,
    }
    started = perf_counter()
    prepared = prepare_observation(observation)
    prepared_at = perf_counter()
    request = encode_request(prepared)
    encoded_at = perf_counter()
    if rtc is not None:
        request[RTC_KEY] = rtc
    reply = client.infer(request)
    received_at = perf_counter()
    actions = np.asarray(reply["actions"])
    if actions.shape != ACTION_SHAPE or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
        raise ValueError("模型必须返回有限的 25×7 动作。")
    if rtc is not None:
        response = reply.get(RTC_KEY, {})
        if (response.get("protocol") != RTC_PROTOCOL or response.get("request_id") != rtc["request_id"]
                or response.get("observation_tick") != rtc["observation_tick"]):
            raise ValueError("RTC 响应与本次观测的时间编号不一致。")
    elapsed = (perf_counter() - started) * 1000
    if timings is not None:
        timings.update({
            "preprocess_ms": (prepared_at - started) * 1000,
            "jpeg_encode_ms": (encoded_at - prepared_at) * 1000,
            "jpeg_image_bytes": sum(len(request[key]) for key in CAMERA_KEYS),
            "rpc_roundtrip_ms": (received_at - encoded_at) * 1000,
            "client_total_ms": elapsed,
            "policy_timing": reply.get("policy_timing"),
            "server_timing": reply.get("server_timing"),
        })
    return actions.astype(float), elapsed


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
    timings = {"completion_check_ms": (perf_counter() - started) * 1000}
    if completion["success"]:
        return (None, 0.0), completion, timings
    actions, _ = infer(client, positions, images, rtc, timings=timings)
    return (actions, (perf_counter() - started) * 1000), completion, timings


def stop_key(*, include_annotations=False):
    import msvcrt
    selected = None
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
        elif key == "\x1b":
            selected = key
        elif key.lower() == "q" and selected != "\x1b":
            selected = "q"
        elif include_annotations and key.lower() in ("s", "h"):
            if selected is None or (selected == "h" and key.lower() == "s"):
                selected = key.lower()
    return selected


def run_episode(follower, cameras, client, reference, seconds, execute, report,
                *, policy_mode="aligned", gripper_delta_deg=MAX_JOINT_DELTA_DEG):
    direction = np.array([follower.config.joint_directions[j] for j in JOINTS])
    lower, upper = np.array([follower.config.joint_limits[j] for j in JOINTS]).T
    positions = motor_vector(list(read_follower(follower).values()))
    max_delta = motor_deltas(gripper_delta_deg)
    initial_images = camera_images(cameras, request_id=0)
    initial_timings = {}
    chunk, elapsed = infer(client, positions, initial_images, timings=initial_timings)
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
    timeline = ActionTimeline(direction, lower, upper, max_delta)
    timeline.install(chunk, 0, 0)
    detector = PlacementDetector(initial_rgb=initial_images.get("observation.images.overhead"))
    if "observation.images.overhead" in initial_images:
        initial_scene = detector.update(initial_images["observation.images.overhead"], 0)
        if initial_scene.get("goal_registered") is False:
            print("首帧黑框配准未确认；完成放置后可目视确认并按 S。", flush=True)
        fraction = initial_scene.get("fraction_in_goal") if initial_scene.get("goal_registered") else None
        report["task_start"] = {**initial_scene,
            "valid_for_pick_and_place": None if fraction is None else fraction < .2}
        if fraction is not None and fraction >= .95:
            print("纸巾开跑前已在黑框内；本条将标记为无效抓放起点。", flush=True)
    report["requests"] = [{"request_id": 0, "observation_tick": 0,
        "roundtrip_ms": elapsed, "raw_actions_adapter": chunk.tolist(), "initial": True,
        "sampling_mode": "plain", "timings": initial_timings}]
    report["completion_observations"] = []
    report["operator_events"] = []
    report["assistance_marked"] = False
    report["outcome"] = "not_completed"
    estimator = DelayEstimator(delay_ticks(elapsed, METADATA["fps"]))
    request_id = 0
    try:
        if execute:
            enable_attempted = True
            report["enable_attempted"] = True
            device_call("Seeed configure", follower.configure)
            configured = True
            report["physical_trial_started"] = True
            last = send_motor_target(follower, last, direction)
            print(f"已使能，开始整臂 VLA 抓放（{policy_mode}）。S 确认放好并回零，Q 退出回零，Esc 立即失能；H 标记人工辅助。", flush=True)
        started = perf_counter()
        report["model_started_at"] = datetime.now().astimezone().isoformat()
        while perf_counter() - started < seconds:
            key = stop_key(include_annotations=True)
            if key == "h":
                report["assistance_marked"] = True
                report["operator_events"].append({"seconds": perf_counter() - started,
                    "event": "manual_assistance_mark", "source": "operator_h_key"})
                print("已记录人工辅助；本条完成后将列为辅助完成。", flush=True)
                key = None
            if key == "s":
                stop_reason = "task_completed"
                report["end_reason"] = "operator_confirmed_placement"
                report["success_evidence"] = {"success": True, "source": "operator_s_key",
                                              "seconds": perf_counter() - started}
                report["outcome"] = ("assisted_completion" if report["assistance_marked"]
                                     else "operator_confirmed_completion")
                break
            if key is not None:
                emergency = key == "\x1b"
                stop_reason = "operator_emergency" if emergency else "normal_exit"
                report["end_reason"] = "escape" if emergency else "operator_q"
                break
            observation = follower.get_observation()
            positions = motor_vector([observation[f"{joint}.pos"] for joint in JOINTS])
            state_read_seconds = perf_counter() - started
            tick = elapsed_tick(perf_counter(), started, METADATA["fps"])
            completed = pending.result() if pending is not None and pending.done() else None
            if completed is not None and completed[1]["success"]:
                completion = completed[1]
                report["completion_observations"].append({
                    "request_id": pending_context["request_id"],
                    "seconds": pending_context["observation_tick"] / METADATA["fps"], **completion})
                report["requests"].append({
                    **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in pending_context.items()},
                    "arrival_tick": tick, "sampling_mode": "completion_only",
                    "timings": completed[2],
                })
                report["end_reason"] = "placement_detected"
                stop_reason = "task_completed"
                report["success_evidence"] = completion
                report["outcome"] = ("assisted_completion" if report["assistance_marked"]
                                     else "placement_completed_requires_visual_review")
                print("检测到纸巾已入框、夹爪撤离且纸巾稳定，结束模型动作并回零。", flush=True)
                break
            if completed is not None:
                (chunk, elapsed), completion, timings = completed
                report["completion_observations"].append({
                    "request_id": pending_context["request_id"],
                    "seconds": pending_context["observation_tick"] / METADATA["fps"], **completion})
                report["request_ms"].append(round(elapsed, 2))
                origin = pending_context["observation_tick"]
                estimator.observe(tick - origin)
                if policy_mode == "aligned":
                    first = timeline.install_aligned(chunk, origin, pending_context["request_id"], tick)
                else:
                    timeline.install(chunk, origin, pending_context["request_id"])
                    first = tick - origin
                report["requests"].append({
                    **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in pending_context.items()},
                    "arrival_tick": tick, "first_usable_index": first,
                    "playout_delay_ticks": None if first is None else tick - origin - first,
                    "used": first is not None,
                    **({"discard_reason": "action_chunk_expired"} if first is None else {}),
                    "roundtrip_ms": round(elapsed, 2), "raw_actions_adapter": chunk.tolist(),
                    "timings": timings,
                })
                pending = None
            if pending is None:
                request_id += 1
                capture_started = perf_counter()
                images = camera_images(cameras, request_id=request_id)
                now = perf_counter()
                tick = elapsed_tick(now, started, METADATA["fps"])
                delay = estimator.ticks
                rtc, fallback, overlap = (rtc_context(timeline, tick, last, delay, request_id)
                                           if policy_mode == "rtc" else (None, None, 0))
                pending_context = {"request_id": request_id,
                    "observation_tick": tick, "inference_delay": delay,
                    "sampling_mode": "rtc" if rtc is not None else "plain",
                    "rtc_fallback_reason": fallback, "valid_old_overlap": overlap,
                    "prefix_horizon": rtc["prefix_horizon"] if rtc is not None else 0,
                    "capture_ms": (now - capture_started) * 1000,
                    "state_read_seconds": state_read_seconds,
                    **({"previous_actions": rtc["previous_actions"]} if rtc is not None else {})}
                pending = pool.submit(infer_and_observe, client, positions.copy(), images,
                                      rtc, detector, now - started)
            tick = elapsed_tick(perf_counter(), started, METADATA["fps"])
            proposed, index = timeline.proposed(tick, last)
            command = next_target(last, proposed, lower, upper, max_delta)
            last = send_motor_target(follower, command, direction) if execute else command
            report["steps"].append({
                "seconds": round(perf_counter() - started, 4),
                "measured_motor_deg": positions.tolist(),
                "target_motor_deg": last.tolist(),
                "proposed_motor_deg": proposed.tolist(), "controller_tick": tick,
                "action_index": index, "source_request_id": timeline.request_id,
                "gripper_torque_nm": observation.get("gripper.torque"),
                "gripper_control_dt_s": getattr(follower, "_gripper_control_dt_s", None),
                "state_read_seconds": state_read_seconds,
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
                    follower, last, max_delta, METADATA["fps"],
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


def configuration_report(policy_mode="aligned", gripper_delta_deg=MAX_JOINT_DELTA_DEG):
    config = official_config()
    return {
        "source": "installed Seeed B601-RS Follower defaults",
        "custom_motion_limits": {
            "max_joint_delta_motor_deg_per_vla_step": MAX_JOINT_DELTA_DEG,
            "start_pose_tolerance_motor_deg": START_TOLERANCE_DEG,
        },
        "execution": {"action_clock": "observation_timestamp", "policy_mode": policy_mode,
                      "max_delta_motor_deg_per_vla_step_by_joint": dict(zip(JOINTS, motor_deltas(gripper_delta_deg).tolist())),
                      "rtc": RTC_PROTOCOL if policy_mode == "rtc" else None,
                      "rtc_delay_estimator": "max_last_10_arrival_delays",
                      "success_stop": "blue_tissue_in_goal_and_gripper_withdrawn",
                      "homing_gripper": "continuous_official_host_pd",
                      "gripper_pd_interval": "measured_monotonic_dt"},
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


def main(argv=None, *, capture_directory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--policy-mode", choices=("aligned", "rtc"))
    parser.add_argument("--gripper-delta-deg", type=float)
    parser.add_argument("--trial-label", default="")
    parser.add_argument("--start-pose", type=Path, default=Path(__file__).with_name("b601-start-pose.json"))
    args = parser.parse_args(argv)
    settings = load_execution_config(args.config)
    seconds = settings["seconds"] if args.seconds is None else args.seconds
    policy_mode = args.policy_mode or settings["policy_mode"]
    gripper_delta = (settings["gripper_delta_motor_deg_per_step"] if args.gripper_delta_deg is None
                     else args.gripper_delta_deg)
    motor_deltas(gripper_delta)
    if args.describe:
        print(json.dumps(configuration_report(policy_mode, gripper_delta), ensure_ascii=False, indent=2))
        return 0
    if not math.isfinite(seconds) or seconds <= 0:
        parser.error("--seconds 必须为正数。")
    setup = json.loads(args.start_pose.read_text(encoding="utf-8"))
    reference = motor_vector(setup["start_motor_deg"])
    report = {
        "configuration": configuration_report(policy_mode, gripper_delta), "physical_trial_started": False,
        "requested_seconds": seconds, "trial_label": args.trial_label,
        "created_at": datetime.now().astimezone().isoformat(),
        "frame_capture_directory": capture_directory,
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
                if policy_mode == "rtc":
                    require_rtc(metadata)
                report["server_metadata"] = metadata
                run_episode(follower, cameras, client, reference, seconds, args.execute, report,
                            policy_mode=policy_mode, gripper_delta_deg=gripper_delta)
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

