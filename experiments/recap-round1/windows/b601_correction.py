"""Model/102 handover and continuous correction collection for B601-RS."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter, sleep, time_ns

import numpy as np

import b601_policy_episode as policy
from b601_action_timeline import ActionTimeline, DelayEstimator, delay_ticks, elapsed_tick, rtc_context
from b601_correction_recording import CorrectionRecorder, RecordingError, atomic_json, capture
from b601_execution_config import DEFAULT_CONFIG, load_execution_config, motor_deltas
from b601_gripper_control import HAND_GUIDED_GRIPPER_KD
from b601_handover import CorrectionKeys, Handover
from b601_hand_guide import GravityGuide, joint_pose
from b601_leader_source import passive_leader
from b601_preflight import DeviceCommunicationError, DriveFaultError, JOINTS, error_record, feedback_only, read_follower
from b601_task_completion import PlacementDetector
from rebot_capture import CAMERA_LABELS, CAMERA_PREFLIGHT_EXIT, CameraFrameError, connect_cameras
from rebot_pi05 import B601WebsocketClientPolicy, HOST, METADATA, PROMPT, JPEG_RGB, TRANSPORT_KEY


HERE = Path(__file__).resolve().parent


def validate_service(client, policy_mode):
    metadata = client.get_server_metadata()
    if metadata.get(TRANSPORT_KEY) != JPEG_RGB or any(metadata.get(k) != v for k, v in METADATA.items()):
        raise ValueError("模型服务的图像、七轴动作或单位与本项目不一致")
    if policy_mode == "rtc":
        policy.require_rtc(metadata)
    return metadata


class InferenceWorker:
    """One RPC at a time; obsolete replies are drained, never issued as commands."""

    def __init__(self, client, policy_mode, factory=None):
        self.client, self.policy_mode, self.factory = client, policy_mode, factory
        self.broken = False
        self.closing = False
        self.lock = Lock()

    def __call__(self, positions, images, rtc, detector, seconds):
        try:
            with self.lock:
                if self.closing:
                    raise RuntimeError("模型工作线程正在退出")
                if self.broken:
                    if self.factory is None:
                        raise RuntimeError("模型连接已断，请退出后重新启动")
                    try:
                        self.client.close()
                    except Exception:
                        pass
                    self.client = self.factory()
                    validate_service(self.client, self.policy_mode)
                    self.broken = False
                client = self.client
            return policy.infer_and_observe(client, positions, images, rtc, detector, seconds)
        except Exception:
            self.broken = True
            raise

    def close(self):
        with self.lock:
            if self.closing:
                return
            self.closing = True
            client = self.client
        client.close()


def alignment_text(sample, position, handover):
    if sample is None:
        return "主臂暂无可靠读数，黑臂继续保持。"
    reference = handover.alignment_reference(position)
    differences = [math.remainder(float(a - b), 360.0) for a, b in zip(sample.angles, reference)]
    return ("对齐待命（黑臂保持）：" + "  ".join(
        f"J{i+1} 白{a:+.1f}/黑换算{b:+.1f}/差{d:+.1f}°"
        for i, (a, b, d) in enumerate(zip(sample.angles, reference, differences)))
        + "。手动摆到相近姿态后按 Enter；程序以当前两臂位置建锚，不会命令白臂摆位。")


def run_session(follower, cameras, worker, leader, recorder, reference, seconds, execute, report,
                *, policy_mode="aligned", gripper_delta_deg=.3, keys=None, guide=None):
    direction = np.array([follower.config.joint_directions[j] for j in JOINTS])
    lower, upper = np.array([follower.config.joint_limits[j] for j in JOINTS]).T
    max_delta = motor_deltas(gripper_delta_deg)
    handover = Handover(direction)
    keys = keys or CorrectionKeys()
    positions = policy.motor_vector(list(read_follower(follower).values()))
    initial_images, initial_times = capture(cameras)
    missing = [label for name, label in CAMERA_LABELS.items() if initial_times.get(name) is None]
    if missing:
        raise CameraFrameError(f"开录前{'、'.join(missing)}相机尚无新画面，请恢复连接后重试")
    initial_timings = {}
    chunk, elapsed = policy.infer(worker.client, positions,
        {f"observation.images.{k}": v for k, v in initial_images.items()}, timings=initial_timings)
    positions = policy.motor_vector(list(read_follower(follower).values()))
    policy.check_start(positions, reference)
    last = np.clip(positions, lower, upper)
    timeline = ActionTimeline(direction, lower, upper, max_delta)
    timeline.install(chunk, 0, 0)
    detector = PlacementDetector(initial_rgb=initial_images["overhead"])
    initial_scene = detector.update(initial_images["overhead"], 0)
    if initial_scene.get("goal_registered") is False:
        print("首帧黑框配准未确认；完成放置后可目视确认并按 S。", flush=True)
    fraction = initial_scene.get("fraction_in_goal") if initial_scene.get("goal_registered") else None
    report.update({"task_start": {**initial_scene, "valid_for_pick_and_place":
                    None if fraction is None else fraction < .2},
                   "steps": [], "requests": [], "operator_events": [], "request_ms": [elapsed],
                   "assistance_marked": False, "operator_pause_used": False, "outcome": "not_completed"})
    estimator = DelayEstimator(delay_ticks(elapsed, METADATA["fps"]))
    request_id = 0
    pending, context = None, None
    enable_attempted, configured = False, False
    stop_reason = "normal_exit"
    pool = ThreadPoolExecutor(max_workers=1)
    started = perf_counter()
    previous_time, previous_mode = started, handover.mode
    model_seconds = 0.0
    last_status = -10.0
    deferred_drag = None
    deferred_stop = None
    resume_after = None
    drag_modes = ("entering_drag", "drag")

    def finish_drag_frame(next_positions):
        nonlocal deferred_drag
        if deferred_drag is None:
            return
        obs, imgs, row = deferred_drag
        deferred_drag = None
        if next_positions is not None:
            demonstrated = next_positions.copy()
            if row["gripper_control"] == "operator_hold":
                demonstrated[-1] = row["target_motor_deg"][-1]
            row["demonstrated_motor_deg"] = demonstrated.tolist()
            row["next_state_observed"] = True
        else:
            row.update(next_state_observed=False, force_positive=False, policy_tag=None,
                       policy_eligible=False, exclusion_reason="missing_next_state")
        recorder.frame(obs, imgs, row)

    def event(name, **data):
        row = {"event": name, "seconds": perf_counter() - started, "unix_time_ns": time_ns(),
               "mode": handover.mode, "control_epoch": handover.epoch,
               "intervention_id": handover.intervention, "next_frame_index": recorder.submitted, **data}
        report["operator_events"].append(row)
        recorder.journal("event", row)

    def hold_for_alignment(reason):
        nonlocal timeline
        previous = handover.mode
        if previous == "human":
            event("intervention_end", reason=reason)
        handover.request_human()
        timeline = ActionTimeline(direction, lower, upper, max_delta)
        event("alignment_requested", previous_mode=previous, reason=reason,
              hold_motor_deg=last.tolist())

    try:
        before_enable_key = keys.poll()
        if before_enable_key in ("q", "\x1b"):
            report["end_reason"] = "escape" if before_enable_key == "\x1b" else "operator_q"
            return
        if execute:
            enable_attempted = True
            report["enable_attempted"] = True
            policy.device_call("Seeed configure", follower.configure)
            configured = True
            report["physical_trial_started"] = True
            last = policy.send_motor_target(follower, last, direction)
        started = perf_counter()
        previous_time = started
        report["model_started_at"] = datetime.now().astimezone().isoformat()
        event("session_started", execute=execute)
        recorder.journal("request", {"request_id": 0, "control_epoch": 0,
            "observation_tick": 0, "sampling_mode": "plain", "roundtrip_ms": elapsed,
            "raw_actions_adapter": chunk.tolist(), "used": True, "initial": True})
        control = ("H 切入黑从臂手拖，看到「可手拖」后可从夹指外侧捏合、撑开，G 保持当前开度或解除保持；松手离开后按 H 交回模型"
                   if guide else "H 暂停/交回模型" + ("，Enter 确认主臂接管" if leader else "（从臂保持，请勿手推）"))
        print(f"纠错采集开始。{control}，S 确认放好并回零，Q 保存退出，Esc 立即失能。", flush=True)
        while True:
            now = perf_counter()
            if previous_mode in ("model", "resuming"):
                model_seconds += now - previous_time
            previous_time = now
            key = before_enable_key if before_enable_key in ("h", "\r") else keys.poll()
            before_enable_key = None
            if handover.mode in drag_modes and key in ("s", "q"):
                deferred_stop, key = key, None
                print("已记下结束请求。请松手并离开活动区，再按 H 确认回零。", flush=True)
            if deferred_stop and key == "h":
                last = guide.anchor(last, perf_counter())
                finish_drag_frame(guide.anchor_measured)
                event("intervention_end", reason="hands_clear_for_stop")
                handover.request_model()
                key, deferred_stop = deferred_stop, None
            if key in ("q", "\x1b"):
                stop_reason = "operator_emergency" if key == "\x1b" else "normal_exit"
                report["end_reason"] = "escape" if key == "\x1b" else "operator_q"
                event("operator_stop", key="Esc" if key == "\x1b" else "Q")
                break
            if key == "s" and recorder.submitted:
                stop_reason = "task_completed"
                report["end_reason"] = "operator_confirmed_placement"
                completion = {"success": True, "source": "operator_s_key",
                              "terminal_frame_index": recorder.submitted - 1}
                report["success_evidence"] = completion
                report["outcome"] = ("assisted_completion_requires_review"
                                     if report["assistance_marked"] or report["operator_pause_used"]
                                     else "operator_confirmed_completion")
                event("placement_confirmed", evidence=completion)
                print("已记录人工确认完成，结束抓放并回零。", flush=True)
                break
            sample, leader_error = leader.latest() if leader is not None else (None, None)
            observation = follower.get_observation()
            frame_positions = joint_pose(observation)
            positions = frame_positions
            state_read = perf_counter() - started
            finish_drag_frame(positions)

            if key == "h":
                if guide and handover.mode in ("model", "resuming"):
                    handover.request_drag()
                    timeline = ActionTimeline(direction, lower, upper, max_delta)
                    guide.begin(last, perf_counter())
                    event("hand_guide_entry", measured_motor_deg=positions.tolist())
                    print("正在切入重力补偿，请等「可手拖」提示。", flush=True)
                elif handover.mode in ("model", "resuming"):
                    hold_for_alignment("operator_h")
                    if leader is None:
                        report["operator_pause_used"] = True
                        print("模型已暂停，从臂保持当前位置。确认活动区清空后按 H 恢复。", flush=True)
                    else:
                        print("模型已暂停。黑臂保持当前目标，请手动对齐白臂；按 Enter 才会开始跟随。", flush=True)
                elif model_seconds >= seconds and handover.mode not in drag_modes:
                    event("model_resume_blocked", reason="model_duration_budget_exhausted")
                    print("模型时限已用完，保持当前控制。可按 S 确认完成，或按 Q 退出回零。", flush=True)
                else:
                    resume_readback = {}
                    if handover.mode in drag_modes:
                        last = guide.anchor(last, perf_counter())
                        positions = guide.anchor_measured.copy()
                        resume_after = perf_counter()
                        resume_readback = {"handover_register_motor_deg": positions.tolist(),
                                           "handover_register_seconds": resume_after - started}
                    if handover.mode in ("human", "drag"):
                        event("intervention_end", reason="return_to_model")
                    handover.request_model()
                    timeline = ActionTimeline(direction, lower, upper, max_delta)
                    # A pending inference can still update its old detector.
                    # Use independent evidence once the control epoch changes.
                    detector = detector.new_window()
                    event("model_resume_requested", hold_motor_deg=last.tolist(), **resume_readback)
                    print("正在从最新画面重新准备模型，黑臂保持；旧动作包不会恢复执行。", flush=True)
            elif key == "g" and guide and handover.mode == "drag" and deferred_stop is None:
                held = guide.toggle_gripper_hold(positions[-1])
                event("gripper_opening_released" if held is None else "gripper_opening_latched",
                      measured_gripper_motor_deg=float(positions[-1]), hold_gripper_motor_deg=held)
                print("夹爪可再次用手调开度。" if held is None
                      else f"已保持夹爪开度（电机 {held:.1f}°），可继续拖臂。再按 G 可调开度。", flush=True)
            elif key == "\r" and handover.mode == "align" and leader is not None:
                if sample is None:
                    event("alignment_not_ready", reason=leader_error)
                    print(f"暂不能跟随：{leader_error}。恢复可靠读数后再按 Enter。", flush=True)
                else:
                    handover.confirm_alignment(sample.angles, last)
                    report["assistance_marked"] = True
                    event("intervention_start", leader_anchor_deg=sample.angles.tolist(),
                          command_anchor_motor_deg=last.tolist(), measured_motor_deg=positions.tolist())
                    print("主臂已接管，当前姿态已作为相对锚点。修正后按 H 交回模型。", flush=True)
            if handover.mode == "human" and sample is None:
                hold_for_alignment("leader_feedback_unavailable")
                event("leader_feedback_unavailable", error=leader_error)
                print(f"主臂跟随已暂停：{leader_error}。黑臂保持；读数恢复后按 Enter 重新建锚。", flush=True)

            if guide and handover.mode == "entering_drag" and guide.phase == "drag":
                handover.drag_ready()
                report["assistance_marked"] = True
                event("intervention_start", source="black_follower_hand_guiding")
                print(f"可手拖黑色从臂。夹爪手调阻尼 Kd={HAND_GUIDED_GRIPPER_KD:.1f}。"
                      "可从夹指外侧捏合或撑开。要松手继续拖臂，按 G 保持开度。松手离开后按 H 交回模型。", flush=True)

            if model_seconds >= seconds:
                if handover.mode == "resuming":
                    hold_for_alignment("model_duration_budget_exhausted")
                    print("等待恢复期间达到模型时限，黑臂继续保持。可按 S 或 Q 结束。", flush=True)
                elif handover.mode == "model":
                    report["end_reason"] = "duration_completed"
                    break
            recorder.check()
            images, image_times = capture(cameras)
            missing_cameras = [name for name, timestamp in image_times.items() if timestamp is None]
            if missing_cameras and handover.mode in ("model", "resuming"):
                hold_for_alignment(f"{missing_cameras[0]}_camera_unavailable")
                event("camera_feedback_unavailable", cameras=missing_cameras)
                roles = "、".join("腕部" if name == "wrist" else "俯视" for name in missing_cameras)
                print(f"{roles}画面暂缺，模型已暂停；黑帧和缺失标志继续记录。相机恢复后按 H。", flush=True)
            tick = elapsed_tick(perf_counter(), started, METADATA["fps"])

            if pending is not None and pending.done():
                old = context["control_epoch"] != handover.epoch or handover.mode not in ("model", "resuming")
                request = {**context, "arrival_tick": tick, "used": not old}
                try:
                    (new_chunk, duration), completion, timings = pending.result()
                    request.update(roundtrip_ms=duration, timings=timings, completion=completion,
                                   raw_actions_adapter=new_chunk.tolist() if new_chunk is not None else None)
                    if old:
                        request["discard_reason"] = "control_epoch_changed"
                    else:
                        report["request_ms"].append(duration)
                        if completion.get("success"):
                            report["success_evidence"] = completion
                            report["end_reason"] = "placement_detected"
                            report["outcome"] = ("assisted_completion_requires_review"
                                                 if report["assistance_marked"] or report["operator_pause_used"]
                                                 else "autonomous_completion_requires_review")
                            stop_reason = "task_completed"
                            event("placement_detected", evidence=completion)
                        else:
                            origin = context["observation_tick"]
                            age = tick - origin
                            estimator.observe(age)
                            if policy_mode == "aligned":
                                first = timeline.install_aligned(new_chunk, origin, context["request_id"],
                                    tick, hold_first=handover.mode == "resuming")
                            else:
                                timeline.install(new_chunk, origin, context["request_id"])
                                first = age
                            request.update(first_usable_index=first, playout_delay_ticks=
                                           None if first is None else age - first)
                            if first is None:
                                request.update(used=False, discard_reason="action_chunk_expired")
                            elif handover.accept_model(context["control_epoch"]):
                                event("model_control_resumed", request_id=context["request_id"],
                                      first_usable_index=first)
                                print("新模型动作已到，已平滑交回模型。", flush=True)
                except RecordingError:
                    raise
                except Exception as error:
                    request.update(error=error_record(error), used=False)
                    if not old:
                        hold_for_alignment("model_request_failed")
                        event("model_request_failed", error=error_record(error))
                        print("模型请求失败，黑臂保持。可按 H 重试模型。", flush=True)
                report["requests"].append(request)
                recorder.journal("request", request)
                pending, context = None, None
                if stop_reason == "task_completed":
                    break

            new_views = resume_after is None or all(t is not None and t >= resume_after for t in image_times.values())
            arm_ready = guide is None or guide.settled(perf_counter())
            if pending is None and handover.mode in ("model", "resuming") and new_views and arm_ready:
                resume_after = None
                request_id += 1
                tick = elapsed_tick(perf_counter(), started, METADATA["fps"])
                delay = estimator.ticks
                rtc, fallback, overlap = (rtc_context(timeline, tick, last, delay, request_id)
                    if policy_mode == "rtc" and handover.mode == "model" else (None, None, 0))
                context = {"request_id": request_id, "control_epoch": handover.epoch,
                           "observation_tick": tick, "inference_delay": delay,
                           "sampling_mode": "rtc" if rtc is not None else "plain",
                           "rtc_fallback_reason": fallback, "valid_old_overlap": overlap,
                           "state_read_seconds": state_read,
                           "prefix_horizon": rtc["prefix_horizon"] if rtc is not None else 0}
                pending = pool.submit(worker, positions.copy(),
                    {f"observation.images.{k}": v for k, v in images.items()}, rtc, detector,
                    perf_counter() - started)
            tick = elapsed_tick(perf_counter(), started, METADATA["fps"])
            if handover.mode in drag_modes:
                proposed, action_index = positions.copy(), None
                if handover.mode == "entering_drag":
                    proposed[-1] = last[-1]
                elif guide.gripper_hold is not None:
                    proposed[-1] = guide.gripper_hold
                source = "hand_drag" if handover.mode == "drag" else "transition"
            elif handover.mode == "human":
                proposed, action_index, source = handover.human_target(sample.angles), None, "leader"
            elif handover.mode == "model" and not handover.hold_first_model_tick:
                proposed, action_index = timeline.proposed(tick, last)
                source = "model" if action_index is not None else "hold"
            else:
                proposed, action_index, source = last.copy(), None, "hold"
                if handover.mode == "model":
                    handover.hold_first_model_tick = False
            command = policy.next_target(last, proposed, lower, upper, max_delta)
            recorder.check()
            if guide and guide.phase != "off":
                if source == "model":
                    guide.release(perf_counter())
                last = guide.send(positions, command, perf_counter(), observation)
            else:
                last = policy.send_motor_target(follower, command, direction) if execute else command
            row = {
                "seconds": perf_counter() - started, "unix_time_ns": time_ns(),
                "controller_tick": tick, "control_mode": handover.mode, "command_source": source,
                "control_epoch": handover.epoch, "intervention_id": handover.intervention if handover.mode in ("human", "drag") else None,
                "command_executed": execute, "model_active_seconds": model_seconds,
                "measured_motor_deg": frame_positions.tolist(), "target_motor_deg": last.tolist(),
                "proposed_motor_deg": proposed.tolist(), "state_read_seconds": state_read,
                "follower_velocity_deg_s": [observation.get(f"{j}.vel") for j in JOINTS],
                "follower_torque_nm": [observation.get(f"{j}.torque") for j in JOINTS],
                "leader_deg": sample.angles.tolist() if sample is not None else None,
                "leader_raw_deg": sample.raw.tolist() if sample is not None else None,
                "leader_read_seconds": sample.sampled_at - started if sample is not None else None,
                "leader_reliable": sample.reliable if sample is not None else None,
                "leader_error": leader_error,
                "camera_frame_seconds": {k: t - started if t is not None else None for k, t in image_times.items()},
                "camera_valid": {k: t is not None for k, t in image_times.items()},
                "source_request_id": timeline.request_id if source == "model" else None,
                "action_index": action_index,
            }
            if guide:
                dragging = source == "hand_drag"
                if dragging:
                    row["gripper_control"] = "hand_guided" if guide.gripper_hold is None else "operator_hold"
                eligible = not missing_cameras and (dragging or source == "model")
                row.update(policy_eligible=eligible, force_positive=dragging,
                           policy_tag="positive" if dragging else None,
                           advantage_pool=eligible and not dragging)
            report["steps"].append(row)
            if source == "hand_drag":
                deferred_drag = observation, {name: img.copy() for name, img in images.items()}, row
            else:
                recorder.frame(observation, images, row)
            if perf_counter() - started - last_status >= 1:
                last_status = perf_counter() - started
                if handover.mode == "align" and leader is not None:
                    print(alignment_text(sample, positions, handover), flush=True)
                else:
                    grip_status = ""
                    if handover.mode == "drag":
                        grip_status = f" · J7 {'手调' if guide.gripper_hold is None else '保持'}，实测 {positions[-1]:.1f}°"
                    print(f"记录 {last_status:.1f}s · {handover.mode}{grip_status} · 模型计时 {model_seconds:.1f}/{seconds:g}s · 已接收 {recorder.submitted} 帧", flush=True)
            previous_mode = handover.mode
            deadline = started + (elapsed_tick(perf_counter(), started, METADATA["fps"]) + 1) / METADATA["fps"]
            if guide and guide.phase != "off":
                last = guide.refresh_until(deadline, last)
            else:
                sleep(max(0.0, deadline - perf_counter()))
    except (DeviceCommunicationError, DriveFaultError):
        stop_reason = "device_fault"
        report["end_reason"] = "device_fault"
        raise
    except BaseException as error:
        stop_reason = "recording_error" if isinstance(error, RecordingError) else "runtime_error"
        if isinstance(error, KeyboardInterrupt):
            stop_reason = "official_overheat" if str(error) == "Overheat detected" else "operator_ctrl_c"
        report["end_reason"] = stop_reason
        raise
    finally:
        report["model_active_seconds"] = model_seconds
        try:
            if deferred_drag is not None:
                finish_drag_frame(None)
            if handover.mode in ("human", "drag"):
                event("intervention_end", reason=report.get("end_reason", "exit"))
            if context is not None:
                recorder.journal("request", {**context, "used": False,
                    "discard_reason": "session_ended_before_application", "status": "pending_at_stop"})
            event("control_recording_ended", end_reason=report.get("end_reason"))
        except Exception as error:
            report["journal_close_error"] = error_record(error)
        try:
            if enable_attempted:
                if not configured:
                    stop_reason = "configure_failed"
                if guide and handover.mode in drag_modes and stop_reason not in ("device_fault", "operator_emergency"):
                    if stop_reason == "official_overheat":
                        # A person may still be holding the arm. Do not initiate a homing path.
                        stop_reason = "operator_emergency"
                    else:
                        print("采集已停止，手拖仍生效。请松手离开活动区，按 H 允许回零。", flush=True)
                        try:
                            while True:
                                key = keys.poll()
                                if key == "\x1b":
                                    stop_reason = "operator_emergency"
                                    break
                                if key == "h":
                                    last = guide.anchor(last, perf_counter())
                                    break
                                obs = follower.get_observation()
                                last = guide.send(joint_pose(obs), last, perf_counter(), obs)
                        except (Exception, KeyboardInterrupt) as error:
                            report["guide_exit_error"] = error_record(error)
                            stop_reason = "device_fault"
                if (guide and guide.phase != "off" and configured and
                        stop_reason not in ("device_fault", "official_overheat", "operator_emergency")):
                    try:
                        last = guide.prepare_return(last, keys.poll)
                    except InterruptedError:
                        stop_reason = "operator_emergency"
                    except (Exception, KeyboardInterrupt) as error:
                        report["guide_exit_error"] = error_record(error)
                        stop_reason = "device_fault"
                print("模型/人工动作已结束，正在执行原停止流程；回位过程不录入纠错数据。", flush=True)
                report["stop"] = policy.stop_follower(follower, last, max_delta, METADATA["fps"],
                    return_to_zero=configured and stop_reason not in ("device_fault", "operator_emergency"),
                    reason=stop_reason, stop_key=lambda: "\x1b" if keys.poll() == "\x1b" else None)
                report["stop_failed"] = not report["stop"]["completed_without_errors"]
        finally:
            if pending is not None:
                pending.cancel()
            try:
                worker.close()
            except Exception as error:
                report["client_close_error"] = error_record(error)
            pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", type=int, default=8766)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--leader-port", help="仅显式选择白色 102 主臂时指定，例如 COM3")
    modes.add_argument("--no-leader", action="store_true", help="H 暂停和恢复，暂停时从臂保持位置")
    modes.add_argument("--hand-guide", action="store_true", help="H 黑从臂重力补偿手拖，再按 H 交回模型")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--policy-mode", choices=("aligned", "rtc"))
    parser.add_argument("--gripper-delta-deg", type=float)
    parser.add_argument("--policy-label", default="", help="本轮模型权重/版本标识，随纠错数据保存")
    parser.add_argument("--start-pose", type=Path, default=HERE / "b601-start-pose.json")
    parser.add_argument("--output", type=Path, default=HERE / "corrections")
    parser.add_argument("--session-dir", type=Path, help="本条采集的输出目录，由连续采集入口分配")
    args = parser.parse_args(argv)
    args.hand_guide = args.hand_guide or not (args.no_leader or args.leader_port)
    settings = load_execution_config(args.config)
    seconds = settings["seconds"] if args.seconds is None else args.seconds
    policy_mode = args.policy_mode or settings["policy_mode"]
    gripper_delta = (settings["gripper_delta_motor_deg_per_step"] if args.gripper_delta_deg is None
                     else args.gripper_delta_deg)
    motor_deltas(gripper_delta)
    if not math.isfinite(seconds) or seconds <= 0:
        parser.error("模型运行秒数必须为正数")
    if args.describe:
        print(json.dumps({"keys": {"H": "黑从臂手拖/从新姿态恢复模型" if args.hand_guide else "模型暂停/交回模型",
                                   **({"G": "手拖期间保持当前夹爪开度／解除保持"} if args.hand_guide else {}),
                                   **({} if args.no_leader or args.hand_guide else {"Enter": "手动对齐后确认主臂接管"}),
                                   "S": "确认放好并回零", "Q": "保存并回位", "Esc": "立即失能"},
                          "execution": {**settings, "seconds": seconds, "policy_mode": policy_mode,
                                        "gripper_delta_motor_deg_per_step": gripper_delta},
                          **({"hand_guided_gripper_mit_kd": HAND_GUIDED_GRIPPER_KD} if args.hand_guide else {}),
                          "leader_port": None if args.no_leader or args.hand_guide else args.leader_port,
                          "recording_root": str(args.output), "leader_origin_writes": False,
                          "seconds_semantics": "仅统计模型运行和恢复等待；对齐与人工段暂停此计时"}, ensure_ascii=False, indent=2))
        return 0
    if args.execute and not sys.stdin.isatty():
        parser.error("真机纠错需要本机交互终端接收 H / Enter / Esc，请双击桌面「启动纠错采集.cmd」")
    if args.hand_guide and not args.execute:
        parser.error("黑从臂手拖需要实际执行；离线查看使用 --describe")
    reference = json.loads(args.start_pose.read_text(encoding="utf-8"))["start_motor_deg"]
    root = args.session_dir or args.output / datetime.now().strftime("correction-%Y%m%d-%H%M%S-%f")
    report = {"physical_trial_started": False, "execute": args.execute,
              "configuration": policy.configuration_report(policy_mode, gripper_delta),
              "requested_model_seconds": seconds, "policy_label": args.policy_label}
    recorder, worker = None, None
    camera_preflight_failed = False
    try:
        policy.check_health(args.port)
        leader_context = nullcontext() if args.no_leader or args.hand_guide else passive_leader(args.leader_port)
        with connect_cameras() as cameras, leader_context as leader, feedback_only() as follower:
            guide = GravityGuide(follower) if args.hand_guide else None
            client = B601WebsocketClientPolicy(HOST, args.port)
            worker = InferenceWorker(client, policy_mode, lambda: B601WebsocketClientPolicy(HOST, args.port))
            metadata = validate_service(client, policy_mode)
            def file_hash(path):
                path = Path(path)
                return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            provenance = {
                "execute": args.execute, "policy_label": args.policy_label,
                "intervention_control": "black_follower_gravity" if guide else ("leader" if leader else "pause"),
                "gravity_profile": guide.signature() if guide else None,
                "server_metadata": metadata, "configuration": report["configuration"],
                "execution_config": file_hash(args.config),
                "calibrations": {"leader": None if leader is None else file_hash(leader.leader.calibration_fpath),
                                 "follower": file_hash(follower.calibration_fpath)},
                "sources": {name: file_hash(HERE / name) for name in
                            ("b601_correction.py", "b601_handover.py", "b601_leader_source.py",
                             "b601_correction_recording.py", "b601_hand_guide.py", "b601_gripper_control.py",
                             "b601_preflight.py", "rebot_capture.py", "b601_policy_episode.py",
                             "b601_task_completion.py", "b601-task-reference.jpg", "b601_stop.py")},
            }
            root.parent.mkdir(parents=True, exist_ok=True)
            recorder = CorrectionRecorder(root, follower, cameras, PROMPT, METADATA["fps"], provenance)
            print(f"纠错数据保存到：{root}", flush=True)
            run_session(follower, cameras, worker, leader, recorder, reference, seconds, args.execute, report,
                        policy_mode=policy_mode, gripper_delta_deg=gripper_delta, guide=guide)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = error_record(error)
        camera_preflight_failed = isinstance(error, CameraFrameError) and not report.get("enable_attempted")
        print(f"纠错采集结束：{error}", flush=True)
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass
        if recorder is not None:
            summary = recorder.close(report.get("outcome", "not_completed"), report)
            atomic_json(root / "report.json", report)
            report["recording_summary"] = summary
            print(f"记录状态：{summary['status']}，已写 {summary['frames_written']} 帧。\n{root}", flush=True)
    if camera_preflight_failed:
        return CAMERA_PREFLIGHT_EXIT
    return int(bool(report.get("error") or report.get("stop_failed")
                    or report.get("recording_summary", {}).get("status") != "saved"))


if __name__ == "__main__":
    raise SystemExit(main())
