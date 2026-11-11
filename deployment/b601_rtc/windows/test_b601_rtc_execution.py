"""Offline checks for clocks, units, completion and continuous homing control."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import b601_policy_episode as policy
from b601_action_timeline import ActionTimeline, delay_ticks, elapsed_tick, require_rtc, RTC_PROTOCOL
from b601_task_completion import PlacementDetector
from test_b601_official_stop import fake_follower, stop


class TimelineTests(unittest.TestCase):
    def make(self):
        return ActionTimeline([1, 1, -1, -1, -1, 1, 6], [-100]*7, [270]*7, 0.3)

    def test_late_packet_uses_observation_clock_not_zero(self):
        timeline = self.make()
        actions = np.repeat(np.arange(25)[:, None], 7, axis=1)
        timeline.install(actions, 100, 17)
        proposed, index = timeline.proposed(107, np.zeros(7))
        self.assertEqual(index, 7)
        np.testing.assert_allclose(proposed, [7, 7, -7, -7, -7, 7, 42])

    def test_expired_packet_holds_without_rewinding(self):
        timeline = self.make()
        timeline.install(np.ones((25, 7)), 10, 1)
        last = np.arange(7)
        proposed, index = timeline.proposed(35, last)
        self.assertIsNone(index)
        np.testing.assert_equal(proposed, last)

    def test_prefix_is_post_limiter_in_adapter_units(self):
        timeline = self.make()
        timeline.install(np.ones((25, 7))*100, 0, 1)
        prefix, overlap = timeline.prefix(7, np.zeros(7))
        self.assertEqual(overlap, 18)
        motor = prefix * timeline.direction
        self.assertLessEqual(np.abs(np.diff(np.vstack([np.zeros(7), motor]), axis=0)).max(), 0.300001)
        np.testing.assert_allclose(motor[0], [.3, .3, -.3, -.3, -.3, .3, .3], atol=1e-6)
        np.testing.assert_allclose(motor[17:], np.repeat(motor[17:18], 8, axis=0))

    def test_late_ticks_and_latency_rounding(self):
        self.assertEqual(elapsed_tick(1.281, 1, 25), 7)
        self.assertEqual(delay_ticks(241, 25), 7)

    def test_old_service_is_rejected(self):
        with self.assertRaises(ValueError):
            require_rtc({})
        require_rtc({'b601_rtc': RTC_PROTOCOL})


class GripperReturnTests(unittest.TestCase):
    def test_sub_180_gripper_refreshed_through_final_settling(self):
        follower = fake_follower([1]*6+[171.7])
        result = stop(follower)
        steps = result['return_to_zero']['steps']
        self.assertTrue(result['completed_without_errors'])
        self.assertGreaterEqual(sum(s['phase']=='return_hold' for s in steps), 100)
        np.testing.assert_allclose([s['target_motor_deg'][6] for s in steps], 171.7)
        self.assertTrue(all('gripper_torque_nm' in s for s in steps))
        self.assertEqual(sum(e[0]=='gripper' and e[1]=='send' for e in follower.trace), len(steps))

    def test_official_wide_open_170_return_remains_with_delta(self):
        result = stop(fake_follower([1]*6+[220]))
        targets = np.array([s['target_motor_deg'] for s in result['return_to_zero']['steps']])
        self.assertTrue(result['completed_without_errors'])
        self.assertAlmostEqual(targets[-1, 6], 170)
        self.assertLessEqual(np.abs(np.diff(targets, axis=0)).max(), .300001)


class CompletionTests(unittest.TestCase):
    def test_initially_in_goal_is_not_success(self):
        detector = PlacementDetector()
        rgb = detector.reference.copy()
        # Move the reference's blue packet into the target in a synthetic image.
        patch_image = rgb[157:280, 463:565].copy()
        rgb[157:280, 463:565] = [170, 170, 170]
        rgb[174:297, 280:382] = patch_image
        for seconds in range(6):
            self.assertFalse(detector.update(rgb, seconds)['success'])

    def test_absent_or_occluded_object_does_not_end_episode(self):
        detector = PlacementDetector()
        for seconds in range(6):
            self.assertFalse(detector.update(np.full((480,640,3),180,np.uint8), seconds)['success'])


class ImmediatePool:
    def __init__(self, **kwargs):
        pass
    def submit(self, function, *args):
        result = function(*args)
        return SimpleNamespace(done=lambda: True, result=lambda: result, cancel=lambda: None)
    def shutdown(self, **kwargs):
        pass


class LoopTests(unittest.TestCase):
    def simulate(self, completion=False):
        clock = SimpleNamespace(now=0.0)
        follower = fake_follower()
        follower.configure = Mock()
        client = SimpleNamespace(close=Mock())
        actions = np.repeat(np.arange(25)[:, None], 7, axis=1).astype(float) + 20
        report = {'physical_trial_started':False, 'request_ms':[], 'steps':[]}
        detector = SimpleNamespace(update=Mock(return_value={'success':completion}))
        images = {'observation.images.overhead':np.zeros((480,640,3), np.uint8)}
        def tick_sleep(seconds):
            clock.now += seconds
        with patch.object(policy,'perf_counter',side_effect=lambda:clock.now), \
             patch.object(policy,'sleep',side_effect=tick_sleep), \
             patch.object(policy,'ThreadPoolExecutor',ImmediatePool), \
             patch.object(policy,'PlacementDetector',return_value=detector), \
             patch.object(policy,'camera_images',return_value=images), \
             patch.object(policy,'stop_key',return_value=None), \
             patch.object(policy,'infer',return_value=(actions,200.0)), \
             patch.object(policy,'stop_follower',return_value={'completed_without_errors':True}) as stopper, \
             patch.object(policy,'send_motor_target',side_effect=lambda f,t,d:t.copy()), \
             redirect_stdout(StringIO()):
            policy.run_episode(follower,{},client,[1]*6+[15],.64,True,report)
        return report,stopper

    def test_control_clock_skips_elapsed_rows_and_preserves_delta(self):
        report,_=self.simulate()
        received=[r for r in report['requests'] if not r.get('initial')]
        self.assertGreater(len(received),0)
        self.assertEqual(received[0]['first_usable_index'],5)
        targets=np.array([s['target_motor_deg'] for s in report['steps']])
        self.assertLessEqual(np.abs(np.diff(targets,axis=0)).max(),.300001)
        for step in report['steps']:
            self.assertEqual(step['action_index'],step['controller_tick']-
                next(r['observation_tick'] for r in report['requests']
                     if r['request_id']==step['source_request_id']))

    def test_placement_ends_without_waiting_for_duration(self):
        report,stopper=self.simulate(completion=True)
        self.assertEqual(report['end_reason'],'placement_detected')
        self.assertEqual(stopper.call_args.kwargs['reason'],'task_completed')
        self.assertLess(len(report['steps']),5)  # Done does not wait for RTC's committed prefix.


if __name__ == '__main__':
    unittest.main()
