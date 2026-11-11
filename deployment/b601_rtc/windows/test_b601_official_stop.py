"""Fault injection against installed Seeed motion code; no hardware is opened."""

from contextlib import redirect_stdout
from io import StringIO
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from motorbridge.errors import CallError

import b601_policy_episode as policy
from b601_preflight import (
    CheckedSeeedB601RSFollower, DeviceCommunicationError, DriveFaultError,
    JOINTS, close_handles, disable_motors, enable_feedback_reporting, official_config,
)
from b601_stop import stop_follower


class FakeMotor:
    def __init__(self, name, trace, angle):
        self.name = name
        self.trace = trace
        self.state = SimpleNamespace(pos=math.radians(angle), vel=0.0, torq=0.0,
                                     t_mos=25.0, status_code=0)
        self.disable_error = None
        self.close_error = None
        self.parameter_error = None
        self.after_send = None

    def request_feedback(self):
        pass

    def get_state(self):
        return self.state

    def robstride_get_param_f32_host_id(self, param_id, host_id, timeout_ms=1000):
        if self.parameter_error:
            raise self.parameter_error
        return self.state.pos

    def ensure_mode(self, mode):
        self.trace.append((self.name, "ensure_mode"))

    def robstride_set_active_report(self, enabled):
        self.trace.append((self.name, "status_reports", enabled))

    def send_mit(self, position, velocity, kp, kd, torque):
        self.trace.append((self.name, "send", math.degrees(position)))
        if self.name != "gripper":
            self.state.pos = position
        if self.after_send:
            self.after_send()

    def disable(self):
        self.trace.append((self.name, "disable"))
        if self.disable_error:
            raise self.disable_error

    def clear_error(self):
        self.trace.append((self.name, "clear_error"))

    def close(self):
        self.trace.append((self.name, "close"))
        if self.close_error:
            raise self.close_error


class FakeBus:
    def __init__(self, trace):
        self.trace = trace
        self.poll_error = None

    def poll_feedback_once(self):
        if self.poll_error:
            raise self.poll_error

    def disable_all(self):
        self.trace.append(("bus", "disable_all"))

    def enable_all(self):
        self.trace.append(("bus", "enable_all"))

    def close(self):
        self.trace.append(("bus", "close"))


def fake_follower(angles=None):
    angles = [1.0] * 6 + [15.0] if angles is None else angles
    follower = object.__new__(CheckedSeeedB601RSFollower)
    follower.config = official_config()
    follower.id = follower.config.id
    follower.motor_names = list(JOINTS)
    follower.cameras = {}
    follower.trace = []
    follower.bus = FakeBus(follower.trace)
    follower.motors = {name: FakeMotor(name, follower.trace, angle)
                       for name, angle in zip(JOINTS, angles, strict=True)}
    follower._in_safe_zero = False
    follower._emergency_disable_requested = False
    return follower


def stop(follower, *, returning=True, key=lambda: None, last=None):
    last = [math.degrees(motor.state.pos) for motor in follower.motors.values()] if last is None else last
    with patch("b601_stop.sleep"), patch("lerobot_robot_seeed_b601.seeed_b601_follower.time.sleep"):
        return stop_follower(follower, last, policy.MAX_JOINT_DELTA_DEG, 25,
                             return_to_zero=returning, reason="test", stop_key=key)


class FeedbackTests(unittest.TestCase):
    def test_status_reporting_has_no_torque_or_motion_and_is_closed(self):
        follower = fake_follower()
        enable_feedback_reporting(follower)
        self.assertEqual([event[0] for event in follower.trace if event[1:] == ("status_reports", True)], list(JOINTS))
        self.assertFalse(any(event[1] in ("send", "enable_all", "disable") for event in follower.trace))
        cleanup = close_handles(follower)
        self.assertTrue(cleanup["all_calls_completed"])
        self.assertEqual([event[0] for event in follower.trace if event[1:] == ("status_reports", False)], list(JOINTS))

    def test_poll_failure_does_not_reuse_cached_state(self):
        follower = fake_follower()
        follower.bus.poll_error = CallError("CAN disconnected")
        with self.assertRaises(DeviceCommunicationError):
            follower.get_observation()

    def test_each_official_fault_bit_aborts(self):
        for fault in (1, 2, 4, 8, 16, 32):
            with self.subTest(fault=fault):
                follower = fake_follower()
                follower.motors[JOINTS[2]].state.status_code = fault
                with self.assertRaises(DriveFaultError):
                    follower.get_observation()

    def test_missing_and_nonfinite_feedback_are_rejected(self):
        follower = fake_follower()
        follower.motors[JOINTS[0]].state = None
        with self.assertRaises(DeviceCommunicationError):
            follower.get_observation()
        follower = fake_follower()
        follower.motors[JOINTS[0]].state.pos = float("nan")
        with self.assertRaises(DeviceCommunicationError):
            follower.get_observation()

    def test_lost_register_reply_blocks_send_despite_cached_feedback(self):
        follower = fake_follower()
        follower.motors[JOINTS[0]].parameter_error = CallError("request timed out")
        with self.assertRaises(DeviceCommunicationError):
            follower.send_action({"shoulder_pan.pos": 1.0})
        self.assertFalse(any(event[1] == "send" for event in follower.trace))

    def test_official_warning_and_abort_thresholds(self):
        follower = fake_follower()
        motor = follower.motors[JOINTS[0]]
        motor.state.t_mos = follower.config.temp_alarm_threshold_c + 0.1
        output = StringIO()
        with redirect_stdout(output):
            follower.send_action({"shoulder_pan.pos": 1.0})
        self.assertIn("HIGH TEMP", output.getvalue())
        motor.state.t_mos = follower.config.temp_overheat_threshold_c + 0.1
        with redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
            follower.send_action({"shoulder_pan.pos": 1.0})


class ShutdownTests(unittest.TestCase):
    def test_all_axes_and_bus_cleaned_after_first_disable_failure(self):
        follower = fake_follower()
        follower.motors[JOINTS[0]].disable_error = CallError("first disable failed")
        result = stop(follower, returning=False)
        self.assertEqual([event[0] for event in follower.trace if event[1] == "disable"], list(JOINTS))
        self.assertEqual([event[0] for event in follower.trace if event[1] == "close"], [*JOINTS, "bus"])
        self.assertFalse(result["disable"]["all_sdk_calls_completed"])
        self.assertIsNone(result["disable"]["physical_disable_confirmation"])
        self.assertNotIn((JOINTS[0], "clear_error"), follower.trace)
        self.assertFalse(result["completed_without_errors"])

    def test_close_failure_does_not_skip_other_handles(self):
        follower = fake_follower()
        follower.motors[JOINTS[0]].close_error = CallError("close failed")
        result = close_handles(follower)
        self.assertEqual([event[0] for event in follower.trace if event[1] == "close"], [*JOINTS, "bus"])
        self.assertFalse(result["all_calls_completed"])

    def test_interrupted_disable_still_attempts_remaining_axes(self):
        follower = fake_follower()
        follower.motors[JOINTS[0]].disable_error = KeyboardInterrupt()
        result = disable_motors(follower)
        self.assertEqual(len(result["motors"]), 7)
        self.assertTrue(result["motors"][-1]["sdk_call_completed"])

    def test_official_safe_zero_stages_and_final_encoders_are_recorded(self):
        follower = fake_follower()
        result = stop(follower)
        returning = result["return_to_zero"]
        self.assertTrue(returning["sequence_completed"])
        self.assertTrue(result["completed_without_errors"])
        self.assertEqual(len(returning["after_motor_deg"]), 7)
        self.assertIsNone(returning["physical_zero_verified"])
        sent = [event for event in follower.trace if event[1] == "send" and event[0] != "gripper"]
        first_shoulder_reduction = next(i for i, event in enumerate(sent)
                                        if event[0] == JOINTS[1] and event[2] < 1.0 - 1e-9)
        self.assertTrue(any(event[0] == JOINTS[0] and abs(event[2]) < 1e-9
                            for event in sent[:first_shoulder_reduction]))
        for joint in JOINTS[:6]:
            values = [1.0] + [event[2] for event in sent if event[0] == joint]
            self.assertLessEqual(max(abs(b - a) for a, b in zip(values, values[1:])), 0.3 + 1e-9)

    def test_temperature_rise_after_first_substep_stops_next_command(self):
        follower = fake_follower()
        motor = follower.motors[JOINTS[0]]
        motor.after_send = lambda: setattr(motor.state, "t_mos", 136.0)
        result = stop(follower)
        sends = [event for event in follower.trace if event[:2] == (JOINTS[0], "send")]
        self.assertEqual(len(sends), 1)
        self.assertEqual(result["return_to_zero"]["result"], "aborted_official_emergency_temperature")
        self.assertFalse(result["completed_without_errors"])
        self.assertEqual(len(result["disable"]["motors"]), 7)

    def test_thermal_return_uses_official_emergency_line_not_normal_abort_line(self):
        follower = fake_follower()
        for motor in follower.motors.values():
            motor.state.t_mos = follower.config.temp_overheat_threshold_c + 0.1
        result = stop(follower)
        self.assertTrue(result["return_to_zero"]["sequence_completed"])
        self.assertTrue(result["completed_without_errors"])

    def test_official_outer_temperature_abort_is_not_success(self):
        follower = fake_follower()
        reader = follower._read_motor_temperatures
        calls = 0

        def temperatures():
            nonlocal calls
            calls += 1
            if calls == 2:
                follower.motors[JOINTS[0]].state.t_mos = 136.0
            return reader()

        follower._read_motor_temperatures = temperatures
        result = stop(follower)
        self.assertEqual(result["return_to_zero"]["result"], "aborted_official_emergency_temperature")
        self.assertFalse(result["return_to_zero"]["sequence_completed"])

    def test_homing_exception_is_preserved_and_all_motors_disabled(self):
        follower = fake_follower()
        follower.safe_zero = Mock(side_effect=RuntimeError("stage two failed"))
        result = stop(follower)
        self.assertEqual(result["return_to_zero"]["error"]["message"], "stage two failed")
        self.assertEqual(result["return_to_zero"]["result"], "failed")
        self.assertFalse(result["completed_without_errors"])
        self.assertEqual(len(result["disable"]["motors"]), 7)

    def test_escape_and_second_ctrl_c_abort_return_and_disable_all(self):
        for interruption in ("escape", "ctrl_c"):
            with self.subTest(interruption=interruption):
                follower = fake_follower()
                if interruption == "escape":
                    result = stop(follower, key=lambda: "\x1b")
                else:
                    follower.safe_zero = Mock(side_effect=KeyboardInterrupt())
                    result = stop(follower)
                self.assertEqual(result["return_to_zero"]["result"], "interrupted")
                self.assertEqual(len(result["disable"]["motors"]), 7)

    def test_encoder_failure_after_sequence_is_not_hidden(self):
        follower = fake_follower()

        def return_then_break(**kwargs):
            follower.motors[JOINTS[0]].parameter_error = CallError("final feedback missing")

        follower.safe_zero = return_then_break
        result = stop(follower)
        self.assertTrue(result["return_to_zero"]["sequence_completed"])
        self.assertEqual(result["return_to_zero"]["result"], "aborted_device_fault")
        self.assertFalse(result["completed_without_errors"])


class EpisodeTests(unittest.TestCase):
    def run_mock_episode(self, failure=None, key=None, configure_error=None):
        follower = fake_follower()
        if configure_error:
            follower.configure = Mock(side_effect=configure_error)
        if failure:
            # First hold succeeds; fail while acquiring the next control state.
            original_observation = follower.get_observation
            def fail_once():
                follower.get_observation = original_observation
                raise failure
            follower.get_observation = fail_once
        reference = [1.0] * 6 + [15.0]
        client = SimpleNamespace(close=Mock())
        report = {"physical_trial_started": False, "request_ms": [], "steps": []}
        actions = np.zeros((25, 7))
        with patch.object(policy, "infer", return_value=(actions, 1.0)), \
             patch.object(policy, "camera_images", return_value={}), \
             patch.object(policy, "stop_key", return_value=key), \
             patch("b601_stop.sleep"), \
             patch("lerobot_robot_seeed_b601.seeed_b601_follower.time.sleep"):
            try:
                policy.run_episode(follower, {}, client, reference, 1.0, True, report)
            except (Exception, KeyboardInterrupt) as error:
                report["caught_error"] = error
        return follower, report

    def test_can_fault_skips_homing(self):
        follower, report = self.run_mock_episode(DeviceCommunicationError("CAN gone"))
        self.assertEqual(report["stop"]["return_to_zero"]["attempted"], False)
        self.assertEqual(report["stop"]["reason"], "device_fault")
        self.assertEqual(len(report["stop"]["disable"]["motors"]), 7)
        self.assertEqual(sum(event == ("bus", "enable_all") for event in follower.trace), 1)

    def test_drive_fault_skips_homing(self):
        _, report = self.run_mock_episode(DriveFaultError(JOINTS[1], 16))
        self.assertFalse(report["stop"]["return_to_zero"]["attempted"])

    def test_inference_oserror_is_not_classified_as_can_fault(self):
        _, report = self.run_mock_episode(OSError("inference connection lost"))
        self.assertTrue(report["stop"]["return_to_zero"]["attempted"])
        self.assertEqual(report["stop"]["reason"], "runtime_error")

    def test_q_and_first_ctrl_c_return_normally(self):
        for kwargs in ({"key": "q"}, {"failure": KeyboardInterrupt()}):
            with self.subTest(kwargs=kwargs):
                _, report = self.run_mock_episode(**kwargs)
                self.assertTrue(report["stop"]["return_to_zero"]["sequence_completed"])
                self.assertNotIn("caught_error", report)

    def test_escape_skips_homing(self):
        _, report = self.run_mock_episode(key="\x1b")
        self.assertFalse(report["stop"]["return_to_zero"]["attempted"])

    def test_partial_enable_failure_attempts_cleanup(self):
        _, report = self.run_mock_episode(configure_error=CallError("partial enable failed"))
        self.assertTrue(report["enable_attempted"])
        self.assertFalse(report["physical_trial_started"])
        self.assertEqual(report["stop"]["reason"], "configure_failed")
        self.assertFalse(report["stop"]["return_to_zero"]["attempted"])
        self.assertEqual(len(report["stop"]["disable"]["motors"]), 7)

    def test_first_hold_failure_is_recorded_as_powered_trial(self):
        # The same startup path is covered by injection at the first send.
        with patch.object(policy, "send_motor_target", side_effect=DeviceCommunicationError("hold send failed")):
            _, report = self.run_mock_episode(key="q")
        self.assertTrue(report["physical_trial_started"])
        self.assertFalse(report["stop"]["return_to_zero"]["attempted"])

    def test_only_two_custom_motion_limits_and_motor_units(self):
        config = policy.configuration_report()
        self.assertEqual(list(config["custom_motion_limits"].values()), [0.3, 5.0])
        follower = fake_follower()
        target = np.array([2.0] * 6 + [15.3])
        directions = np.array(list(follower.config.joint_directions.values()))
        accepted = policy.send_motor_target(follower, target, directions)
        np.testing.assert_allclose(accepted, target)
        policy.check_start(np.ones(7) * 5.0, np.zeros(7))
        with self.assertRaises(RuntimeError):
            policy.check_start(np.ones(7) * 5.01, np.zeros(7))


class RestoreStartTests(unittest.TestCase):
    def test_saved_start_is_restored_with_existing_delta(self):
        import b601_restore_start as restoration
        follower = fake_follower([0.0] * 6 + [51.0])
        reference = np.array([5.154, -0.017, 1.077, -2.828, -0.501, 6.192, 15.642])
        report = {"restored": False, "steps": []}

        def send_and_update_feedback(follower, target, direction):
            accepted = policy.send_motor_target(follower, target, direction)
            for joint, angle in zip(JOINTS, accepted, strict=True):
                follower.motors[joint].state.pos = math.radians(angle)
            return accepted

        with patch.object(restoration, "send_motor_target", side_effect=send_and_update_feedback), \
             patch.object(restoration, "stop_key", return_value=None), \
             patch.object(restoration, "sleep"), redirect_stdout(StringIO()):
            restoration.restore(follower, reference, report)
        self.assertTrue(report["restored"])
        self.assertFalse(report["stop"]["return_to_zero"]["attempted"])
        self.assertTrue(report["stop"]["completed_without_errors"])
        targets = np.array([report["before_motor_deg"]] + [step["target_motor_deg"] for step in report["steps"]])
        self.assertLessEqual(float(np.abs(np.diff(targets, axis=0)).max()), 0.3 + 1e-9)
        policy.check_start(report["after_motor_deg"], reference)

    def test_device_failure_during_restoration_skips_return_and_cleans_all(self):
        import b601_restore_start as restoration
        follower = fake_follower()
        follower.get_observation = Mock(side_effect=DeviceCommunicationError("CAN lost during restore"))
        report = {"restored": False, "steps": []}
        with patch.object(restoration, "stop_key", return_value=None), \
             self.assertRaises(DeviceCommunicationError):
            restoration.restore(follower, np.array([0.0] * 6 + [15.0]), report)
        self.assertFalse(report["stop"]["return_to_zero"]["attempted"])
        self.assertEqual(len(report["stop"]["disable"]["motors"]), 7)


if __name__ == "__main__":
    unittest.main()
