"""Handoff, next-frame training actions, two-camera media and the new batch; no CAN."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyarrow.parquet as pq

import b601_correction as live
import b601_recap_batch as batch
from b601_correction_recording import CorrectionRecorder, atomic_json
from b601_hand_guide import GravityGuide, joint_pose
from b601_preflight import JOINTS
from b601_recap_batch import GOAL, review, scan
from test_b601_official_stop import fake_follower


class GuideTests(unittest.TestCase):
    def test_hand_can_close_and_open_j7_latch_it_and_return_at_the_measured_aperture(self):
        follower = fake_follower([1.] * 6 + [120.])
        guide = GravityGuide(follower)
        gripper = follower.motors[JOINTS[-1]]
        entry = joint_pose(follower.get_observation())
        guide.begin(entry, 0.)
        for now in (0., .25, .5, .75, 1.):
            guide.send(entry, entry, now)
        self.assertEqual(guide.phase, "drag")

        def move_jaws(angle, now):
            gripper.state.pos = np.radians(angle)
            return guide.send(joint_pose(follower.get_observation()), entry, now)

        with patch.object(gripper, "send_mit", wraps=gripper.send_mit) as mit:
            for angle, now in ((60., 1.1), (90., 1.2)):
                before = mit.call_count
                last = move_jaws(angle, now)
                self.assertAlmostEqual(last[-1], angle)
                self.assertEqual(mit.call_count, before + 1)
                self.assertEqual(mit.call_args.args, (0., 0., 0., 1.0, 0.))
                self.assertIs(follower.motors[JOINTS[-1]], gripper)
                self.assertIsNone(follower._gripper_prev_target_pos)

            self.assertAlmostEqual(guide.toggle_gripper_hold(last[-1]), 90.)
            held = move_jaws(92., 1.3)
            self.assertAlmostEqual(held[-1], 90.)
            self.assertEqual(mit.call_args.args[3], 1.5)
            self.assertLess(mit.call_args.args[-1], 0.)
            self.assertGreaterEqual(mit.call_args.args[-1], -follower.config.gripper_mit_torque_limit)
            gripper.state.pos = np.radians(94.)
            anchored = guide.anchor(held, 1.4)
            self.assertAlmostEqual(guide.anchor_measured[-1], 94.)
            self.assertAlmostEqual(anchored[-1], 90.)

            guide.begin(anchored, 2.)
            for now in (2., 2.25, 2.5, 2.75, 3.):
                guide.send(joint_pose(follower.get_observation()), anchored, now)
            self.assertIsNone(guide.gripper_hold)
            guide.toggle_gripper_hold(94.)
            self.assertIsNone(guide.toggle_gripper_hold(94.))
            last = move_jaws(70., 3.1)
            self.assertEqual(mit.call_args.args, (0., 0., 0., 1.0, 0.))
            gripper.state.pos = np.radians(65.)
            anchored = guide.anchor(last, 3.2)
            self.assertAlmostEqual(anchored[-1], 65.)
            guide.send(joint_pose(follower.get_observation()), anchored, 3.2)
            self.assertEqual(mit.call_args.args[3], 1.5)
            self.assertAlmostEqual(mit.call_args.args[-1], 0.)
        self.assertFalse(any(e[1] in ("disable", "disable_all", "enable_all") for e in follower.trace))

    def test_free_gripper_obeys_official_overheat_abort_and_restores_native_handle(self):
        follower = fake_follower()
        gripper = follower.motors["gripper"]
        action = {f"{joint}.pos": 0. for joint in JOINTS}
        gripper.state.t_mos = follower.config.temp_overheat_threshold_c + 1.
        with patch.object(gripper, "send_mit", wraps=gripper.send_mit) as mit, redirect_stdout(StringIO()):
            with self.assertRaisesRegex(KeyboardInterrupt, "Overheat"):
                follower.send_gravity_action(action, [15.] * 6, [2.5] * 6, [0.] * 6, hand_guided_gripper=True)
            mit.assert_not_called()
            self.assertIs(follower.motors["gripper"], gripper)
            self.assertFalse(follower.config.gravity_compensation)
            gripper.state.t_mos = 25.
            follower.send_action(action)
            self.assertEqual(mit.call_args.args[3], 1.5)

    def test_transfer_carries_a_real_position_error_without_a_mid_fade_torque_dip(self):
        f = fake_follower()
        guide = GravityGuide(f)
        pos = joint_pose(f.get_observation())
        obs = f.get_observation()
        target = pos.copy()
        target[2] += 2.
        load = 150. * np.radians(2.)
        obs[f"{JOINTS[2]}.torque"] = load
        guide.begin(target, 0.)
        with patch.object(f.motors[JOINTS[2]], "send_mit", wraps=f.motors[JOINTS[2]].send_mit) as motor:
            for t in (0., .25, .5, .625, .75, .875, 1.):
                guide.send(pos, target, t, obs)
                q, _, kp, _, tau = motor.call_args.args
                self.assertAlmostEqual(kp * (q - np.radians(pos[2])) + tau, load, places=9)

    def test_supported_entry_free_motion_and_reacquisition_never_cycle_power(self):
        f = fake_follower()
        guide = GravityGuide(f)
        p = joint_pose(f.get_observation())
        with patch.object(f.motors[JOINTS[2]], "send_mit", wraps=f.motors[JOINTS[2]].send_mit) as motor:
            guide.begin(p, 0.)
            for t in np.arange(0, 1.3, .02):
                guide.send(p, p, t)
            self.assertEqual(guide.phase, "drag")
            self.assertAlmostEqual(motor.call_args.args[4], 0.)
            for i in range(45):
                moved = p.copy()
                moved[0] += .16 * i
                guide.send(moved, p, 1.4 + i * .02)
            self.assertGreater(motor.call_args.args[4], 6.)
            anchor = guide.anchor(p, 2.4)
            self.assertGreater(anchor[0], p[0] + 6)
            guide.send(moved, anchor, 2.4)
            guide.send(moved, anchor, 2.91)
            self.assertEqual(motor.call_args.args[2], 150.)
            guide.release(3.)
            guide.send(moved, anchor, 3.51)
            self.assertEqual(guide.phase, "off")
            self.assertEqual(motor.call_args.args[4], 0.)
            self.assertFalse(f.config.gravity_compensation)
            self.assertFalse(any("disable" in str(entry) or "enable" in str(entry) for entry in f.trace))

    def test_real_dataset_drag_h_hands_back_to_new_pose_and_views(self):
        f = fake_follower([5.,10.,20.,3.,0.,4.,6.])
        f.configure = Mock()
        start = joint_pose(f.get_observation())
        den = np.array([f.config.joint_directions[j] for j in JOINTS])
        cams = {name:SimpleNamespace(width=160, height=120) for name in ("wrist","overhead")}
        imgs = {name:np.zeros((120,160,3),np.uint8) for name in cams}
        guide = GravityGuide(f)
        key_count = drag_ticks = 0
        resume_key = None
        resume = None
        anchor = guide.anchor
        def anchor_after_sampling(last, now):
            # The independent register read may move after the recorded camera/state observation.
            f.motors[JOINTS[0]].state.pos += np.radians(.04)
            return anchor(last, now)
        def key():
            nonlocal key_count, drag_ticks, resume, resume_key
            key_count += 1
            if guide.phase == "drag":
                drag_ticks += 1
                f.motors[JOINTS[0]].state.pos += np.radians(.4)
                # Hand motion under free control, with a small rebound during the grip hold.
                f.motors[JOINTS[-1]].state.pos += np.radians(.25 if guide.gripper_hold is not None else 1.)
                if drag_ticks in (4, 9):
                    return "g"
                if drag_ticks == 14:
                    resume_key, resume = key_count, perf_counter()
                    return "h"
            if resume_key is not None and key_count == resume_key + 20:
                return "q"
            return "h" if key_count == 3 else None
        client = SimpleNamespace(infer=lambda req: {"actions": np.tile(start/den,(25,1))}, close=Mock())
        worker = live.InferenceWorker(client, "aligned")
        report = {}
        def capture(c):
            # Cache only old photographs during the first 2 video frames after the H return.
            now = resume - .02 if resume and key_count < resume_key + 3 else perf_counter()
            return imgs, dict.fromkeys(cams, now)
        detector = Mock()
        detector.update.return_value = {"success":False}
        detector.new_window.return_value = detector
        with TemporaryDirectory() as directory:
            root = Path(directory)/"simulated-handoff"
            rec = CorrectionRecorder(root, f, cams, "test", 25, {"intervention_control":"black_follower_gravity"})
            with patch.object(guide,"anchor",side_effect=anchor_after_sampling), \
                 patch.object(live,"capture",side_effect=capture), patch.object(live,"PlacementDetector",return_value=detector), \
                 patch.object(live.policy,"stop_follower",return_value={"completed_without_errors":True}), redirect_stdout(StringIO()):
                try:
                    live.run_session(f,cams,worker,None,rec,start,5,True,report,keys=SimpleNamespace(poll=key),guide=guide)
                finally:
                    info = rec.close(report.get("outcome"),report)
            self.assertEqual(info["status"],"saved",info["errors"])
            trace = [json.loads(line) for line in (root/"telemetry-episode-000000.jsonl").read_text(encoding="utf-8").splitlines()]
            data = pq.read_table(next((root/"data").rglob("*.parquet"))).to_pylist()
            manual = [i for i,r in enumerate(trace) if r["command_source"]=="hand_drag"]
            self.assertGreater(len(manual), 2)
            self.assertTrue(any(trace[i]["demonstrated_motor_deg"][0] > trace[i]["measured_motor_deg"][0]+.1 for i in manual))
            grip_events = [e for e in report["operator_events"] if e["event"].startswith("gripper_opening_")]
            self.assertEqual([e["event"] for e in grip_events], ["gripper_opening_latched", "gripper_opening_released"])
            latched = grip_events[0]["hold_gripper_motor_deg"]
            self.assertAlmostEqual(latched, grip_events[0]["measured_gripper_motor_deg"])
            self.assertIsNone(grip_events[1]["hold_gripper_motor_deg"])
            self.assertEqual({trace[i]["gripper_control"] for i in manual}, {"operator_hold", "hand_guided"})
            for i in manual:
                expected = np.array(trace[i+1]["measured_motor_deg"])
                if trace[i]["gripper_control"] == "operator_hold":
                    expected[-1] = latched
                np.testing.assert_allclose(data[i]["action"], expected/den, atol=1e-5)
            self.assertTrue(any(abs(trace[i]["measured_motor_deg"][-1]-latched) > .2
                                for i in manual if trace[i]["gripper_control"] == "operator_hold"))
            for row,frame in zip(trace,data,strict=True):
                self.assertEqual(frame["recap.advantage_tag"],1 if row["command_source"]=="hand_drag" else -1)
                self.assertEqual(frame["recap.advantage_pool"],row["command_source"]=="model")
            ev = next(e for e in report["operator_events"] if e["event"]=="model_resume_requested")
            self.assertGreater(ev["hold_motor_deg"][0],start[0]+1.)
            self.assertGreater(ev["hold_motor_deg"][-1],start[-1]+1.)
            self.assertAlmostEqual(ev["hold_motor_deg"][-1],ev["handover_register_motor_deg"][-1])
            before = trace[ev["next_frame_index"]]["measured_motor_deg"]
            self.assertAlmostEqual(ev["handover_register_motor_deg"][0] - before[0], .04, places=5)
            req = next(r for r in report["requests"] if r["control_epoch"] == ev["control_epoch"])
            self.assertGreaterEqual(req["state_read_seconds"]-ev["seconds"], .06)
            self.assertEqual(info["media_validation"]["decoded_video_frames"],dict.fromkeys(cams,len(data)))
            self.assertTrue(info["human_intervention_used"])


class BatchTests(unittest.TestCase):
    def test_unplugged_camera_skips_restore_and_failed_open_retries_the_same_number(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            checks = [batch.CameraFrameError("俯视未连接"), {"wrist": 0, "overhead": 2},
                      {"wrist": 0, "overhead": 2}]
            with patch.object(batch, "ROOT", root), patch.object(batch, "require_camera_indices", side_effect=checks), \
                 patch.object(batch, "run", side_effect=[0, batch.CAMERA_PREFLIGHT_EXIT, 0, 1]) as commands, \
                 patch("builtins.input", side_effect=["n", "n", "n"]), redirect_stdout(StringIO()):
                result = batch.main(["--config", "dummy", "--gripper-delta-deg", "1"])
            # Two explicit retries: the first N never restored, the last non-camera failure paused the batch.
            self.assertEqual(result, 1)
            self.assertEqual([call.args[0] for call in commands.call_args_list],
                             ["b601_restore_start.py", "b601_correction.py"] * 2)
            self.assertTrue(all(Path(call.args[-1]).name.startswith("rollout-001-")
                                for call in commands.call_args_list[1::2]))
            self.assertEqual(batch.scan(root), ([], []))

    def test_completed_stop_with_bad_recording_returns_to_next_prompt(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            attempts = []
            def fake_run(script, *args):
                if script == "b601_restore_start.py":
                    return 0
                dest = Path(args[args.index("--session-dir") + 1])
                dest.mkdir()
                attempts.append(dest)
                partial = len(attempts) == 1
                atomic_json(dest/"session.json", {"status":"partial" if partial else "saved",
                    "frames_written":10, "end_reason":"placement_detected",
                    "metadata":{"execute":True, "intervention_control":"black_follower_gravity"}})
                atomic_json(dest/"report.json", {"physical_trial_started":True,"stop_failed":False,
                    "end_reason":"placement_detected", "stop":{"completed_without_errors":True,
                    "disable":{"all_sdk_calls_completed":True},"close":{"all_calls_completed":True}}})
                return int(partial)
            answers = ["n", "", "n", "", "r", "1", "e"]
            with patch.object(batch,"ROOT",root), patch.object(batch,"require_camera_indices"), \
                 patch.object(batch,"run",side_effect=fake_run), \
                 patch("builtins.input",side_effect=answers), redirect_stdout(StringIO()):
                self.assertEqual(batch.main(["--config","dummy","--gripper-delta-deg","1"]),0)
            self.assertEqual(len(attempts),2)
            self.assertEqual(batch.scan(root),([attempts[1]],[]))
            report = batch.read(attempts[0]/"report.json")
            report["stop"]["disable"]["all_sdk_calls_completed"] = False
            atomic_json(attempts[0]/"report.json",report)
            self.assertFalse(batch.recoverable_recording_stop(attempts[0]))

    def test_saved_success_and_failure_count_across_restart_old_and_invalid_excluded(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            meta = {"status":"saved","frames_written":10,"end_reason":"duration_completed",
                    "metadata":{"execute":True,"intervention_control":"black_follower_gravity"}}
            for n,choice in (("rollout-001","1"),("rollout-002","0"),("rollout-bad","x"),("rollout-pending",""),("old-run","1")):
                p = root/n
                p.mkdir()
                atomic_json(p/"session.json",meta)
                review(p,meta,lambda _prompt:choice)
            counted, pending = scan(root)
            self.assertEqual([p.name for p in counted],["rollout-001","rollout-002"])
            self.assertEqual([p.name for p,_ in pending],["rollout-pending"])
            self.assertEqual(GOAL,100)


if __name__ == "__main__":
    unittest.main()
