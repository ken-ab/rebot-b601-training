"""Offline ownership, RPC, freshness and paired-recording tests; no device is opened."""

from collections import deque
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import b601_correction as run
import rebot_capture as camera_io
from b601_correction_recording import CorrectionRecorder, RecordingError
from b601_handover import CorrectionKeys, Handover
from b601_leader_source import LeaderReader, LeaderSample, passive_leader
from b601_preflight import DeviceCommunicationError, JOINTS, official_config


DIRECTION = np.array([1, 1, -1, -1, -1, 1, 6.0])
POSE = np.array([5, 10, 20, 3, 0, 4, 6.0])


class CameraBindingTests(unittest.TestCase):
    def test_only_verified_usb_ports_resolve_and_conflicting_overhead_ports_are_rejected(self):
        def moniker(device_id):
            return ("@device_pnp_\\\\?\\" + device_id.replace("\\", "#")
                    + "#" + camera_io.VIDEO_INTERFACE + "\\global [UGREEN Camera 2K]")

        wrist, = camera_io.CAMERA_USB_IDS["wrist"]
        old, current = camera_io.CAMERA_USB_IDS["overhead"]
        unknown = old.replace("7&7736773", "7&1234567")
        with patch.object(camera_io, "FFMPEG"), patch.object(camera_io.subprocess, "run") as sources:
            for overhead in (old, current):
                sources.return_value.stdout = "\n".join(map(moniker, (unknown, overhead, wrist)))
                self.assertEqual(camera_io.require_camera_indices(), {"wrist": 2, "overhead": 1})
            sources.return_value.stdout = "\n".join(map(moniker, (wrist, unknown)))
            with self.assertRaisesRegex(camera_io.CameraFrameError, "俯视"):
                camera_io.require_camera_indices()
            sources.return_value.stdout = "\n".join(map(moniker, (current, wrist, old)))
            with self.assertRaisesRegex(camera_io.CameraFrameError, "多个"):
                camera_io.directshow_camera_indices()
            for devices, roles in (
                ((wrist, unknown, camera_io.EVAL_WRIST_USB_ID), {"overhead": 0, "wrist": 2}),
                ((camera_io.EVAL_WRIST_USB_ID, wrist), {"overhead": 1, "wrist": 0}),
            ):
                sources.return_value.stdout = "\n".join(map(moniker, devices))
                self.assertEqual(camera_io.require_camera_indices(), roles)

    def test_preflight_camera_failure_exposes_retry_without_connecting_the_follower(self):
        with patch.object(run.sys.stdin, "isatty", return_value=True), patch.object(run.policy, "check_health"), \
             patch.object(run, "connect_cameras", side_effect=camera_io.CameraFrameError("俯视暂时不可用")), \
             patch.object(run, "feedback_only") as can, redirect_stdout(StringIO()):
            self.assertEqual(run.main(["--execute", "--hand-guide"]), camera_io.CAMERA_PREFLIGHT_EXIT)
        can.assert_not_called()


class HandoverTests(unittest.TestCase):
    def test_alignment_is_a_clutch_then_first_target_is_exactly_current_command(self):
        h = Handover(DIRECTION)
        h.request_human()
        self.assertEqual(h.mode, "align")
        arbitrary_leader = np.array([15, 22, -35, 80, 12, 17, 3.0])
        h.confirm_alignment(arbitrary_leader, POSE)
        np.testing.assert_array_equal(h.human_target(arbitrary_leader), POSE)
        np.testing.assert_allclose(h.human_target(arbitrary_leader + 1), POSE + DIRECTION)

    def test_roll_wrap_is_shortest_motion_and_does_not_accumulate_a_full_turn(self):
        h = Handover(DIRECTION)
        h.request_human()
        h.confirm_alignment([179]*7, POSE)
        np.testing.assert_allclose(h.human_target([-179]*7), POSE + 2*DIRECTION)

    def test_old_epoch_cannot_reclaim_control_after_human_takeover(self):
        h = Handover(DIRECTION)
        h.request_human()
        h.confirm_alignment([0]*7, POSE)
        h.request_model()
        self.assertFalse(h.accept_model(0))
        self.assertEqual(h.mode, "resuming")
        self.assertTrue(h.accept_model(h.epoch))
        self.assertTrue(h.hold_first_model_tick)


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.buffer = deque()
        self.down = set()
        self.keys = CorrectionKeys(lambda: bool(self.buffer), self.buffer.popleft,
                                   lambda code: code in self.down)

    def test_held_h_and_release_tail_do_not_exit_alignment(self):
        self.down.add(72)
        self.buffer.extend('hhh')
        self.assertEqual(self.keys.poll(), 'h')
        self.buffer.extend('hh')
        self.assertIsNone(self.keys.poll())
        self.down.clear()
        self.buffer.extend('hh')
        self.assertIsNone(self.keys.poll())
        self.down.add(72)
        self.buffer.append('h')
        self.assertEqual(self.keys.poll(), 'h')

    def test_escape_has_priority_over_buffered_takeover_keys(self):
        self.buffer.extend('gh\rq\x1b')
        self.assertEqual(self.keys.poll(), '\x1b')

    def test_gripper_hold_toggles_once_per_g_press(self):
        self.down.add(ord('G'))
        self.buffer.extend('Ggg')
        self.assertEqual(self.keys.poll(), 'g')
        self.buffer.append('g')
        self.assertIsNone(self.keys.poll())
        self.down.clear()
        self.assertIsNone(self.keys.poll())
        self.down.add(ord('G'))
        self.buffer.append('G')
        self.assertEqual(self.keys.poll(), 'g')


class MemoryRecorder:
    def __init__(self, fail_at=None):
        self.submitted = 0
        self.rows = []
        self.journals = []
        self.fail_at = fail_at

    def check(self):
        if self.submitted == self.fail_at:
            raise RecordingError('injected writer failure')

    def journal(self, kind, row):
        self.journals.append((kind, row.copy()))

    def frame(self, observation, images, row):
        self.rows.append(row.copy())
        self.submitted += 1


class SessionTests(unittest.TestCase):
    def simulate(self, keys, *, delay=.12, failure=False, lose_leader=None, record_failure=None,
                 missing_camera=None, follower_fault_at=None, seconds=3, no_leader=False):
        clock = SimpleNamespace(now=0.0)
        pose = POSE.copy()
        follower = SimpleNamespace(config=official_config(), configure=Mock())
        client = SimpleNamespace(close=Mock())
        observations = 0
        def get_observation():
            nonlocal observations
            observations += 1
            if observations == follower_fault_at:
                raise DeviceCommunicationError('injected CAN failure')
            result = {}
            for j, value in zip(JOINTS, pose):
                result.update({f'{j}.pos':float(value), f'{j}.vel':0., f'{j}.torque':0.})
            return result
        follower.get_observation = get_observation
        def leader_sample():
            if lose_leader and lose_leader[0] <= clock.now < lose_leader[1]:
                return None, 'injected stale UART'
            # Leader moves during alignment; only post-confirmation displacement may be followed.
            angle = np.full(7, clock.now * 10)
            return LeaderSample(angle, angle.copy(), clock.now, [True]*7), None
        leader = None if no_leader else SimpleNamespace(latest=leader_sample)
        recorder = MemoryRecorder(record_failure)
        images = {name: np.zeros((48,64,3),np.uint8) for name in ('overhead','wrist')}
        initial = np.repeat((POSE / DIRECTION)[None,:],25,axis=0)
        next_actions = np.repeat(((POSE+10) / DIRECTION)[None,:],25,axis=0)
        pool_calls = []
        class Pool:
            def __init__(self,**kwargs): pass
            def submit(self, function, *args):
                call_time = clock.now
                pool_calls.append((call_time, args[2]))
                def result():
                    if failure and len(pool_calls)==1:
                        raise RuntimeError('injected obsolete network error')
                    return (next_actions.copy(),delay*1000), {'success':False}, {}
                return SimpleNamespace(done=lambda:clock.now>=call_time+delay-1e-7,
                                       result=result, cancel=lambda:False)
            def shutdown(self,**kwargs): pass
        input_events = dict(keys)
        def image_capture(cameras):
            stamps = dict.fromkeys(images, clock.now)
            if missing_camera:
                role, start, end = missing_camera
                if start <= clock.now < end:
                    stamps[role] = None
            return images, stamps
        def read_key():
            tick = round(clock.now*25)
            return input_events.pop(tick, None)
        worker = SimpleNamespace(client=client, close=Mock())
        def send(f, target, direction):
            pose[:] = target
            return target.copy()
        report = {'physical_trial_started':False}
        detector = Mock()
        detector.update.return_value = {'success':False}
        detector.new_window.return_value = detector
        with patch.object(run,'perf_counter',side_effect=lambda:clock.now), \
             patch.object(run,'sleep',side_effect=lambda duration:setattr(clock,'now',clock.now+duration)), \
             patch.object(run,'read_follower',side_effect=lambda f:dict(zip(JOINTS,POSE))), \
             patch.object(run,'capture',side_effect=image_capture), \
             patch.object(run,'PlacementDetector',return_value=detector), \
             patch.object(run,'ThreadPoolExecutor',Pool), \
             patch.object(run.policy,'infer',return_value=(initial,20.)), \
             patch.object(run.policy,'send_motor_target',side_effect=send), \
             patch.object(run.policy,'stop_follower',return_value={'completed_without_errors':True}) as stopper, \
             redirect_stdout(StringIO()):
            try:
                run.run_session(follower,{},worker,leader,recorder,POSE,seconds,True,report,
                                keys=SimpleNamespace(poll=read_key))
                error = None
            except Exception as caught:
                error = caught
        return report,recorder,stopper,pool_calls,error

    def test_alignment_human_and_resume_all_record_and_old_model_packet_is_discarded(self):
        report,r,stopper,calls,error = self.simulate({1:'h',5:'\r',10:'h',23:'q'},delay=.2)
        self.assertIsNone(error)
        self.assertTrue(report['assistance_marked'])
        modes={s['control_mode'] for s in r.rows}
        self.assertEqual(modes, {'model','align','human','resuming'})
        requests=[v for k,v in r.journals if k=='request' and v.get('request_id')==1]
        self.assertFalse(requests[0]['used'])
        self.assertEqual(requests[0]['discard_reason'],'control_epoch_changed')
        self.assertFalse(any(.04<=t<.4 for t,_ in calls[1:]))
        targets=np.array([POSE]+[row['target_motor_deg'] for row in r.rows])
        self.assertLessEqual(np.abs(np.diff(targets,axis=0)).max(),.300001)
        for mode in ['align','resuming']:
            for i,row in enumerate(r.rows):
                if row['control_mode']==mode:
                    np.testing.assert_allclose(row['target_motor_deg'],targets[i])
        events=[v for k,v in r.journals if k=='event']
        human_start=next(e for e in events if e['event']=='intervention_start')
        first=next(s for s in r.rows if s['control_mode']=='human')
        np.testing.assert_allclose(first['target_motor_deg'],human_start['command_anchor_motor_deg'])
        resumed=next(e for e in events if e['event']=='model_control_resumed')
        first_model=next(s for s in r.rows if s['control_mode']=='model' and s['control_epoch']==resumed['control_epoch'])
        self.assertEqual(first_model['command_source'],'hold')
        self.assertEqual(report['end_reason'],'operator_q')
        stopper.assert_called_once()

    def test_pending_old_rpc_error_cannot_abort_human_control(self):
        report,r,_,_,error=self.simulate({1:'h',3:'\r',12:'q'},failure=True,delay=.2)
        self.assertIsNone(error)
        self.assertEqual(report['end_reason'],'operator_q')
        self.assertTrue(any(row['command_source']=='leader' for row in r.rows))

    def test_no_leader_h_holds_then_resumes_without_ending_session(self):
        report, recorder, stopper, _, error = self.simulate(
            {1: 'h', 3: '\r', 5: 'h', 14: 'q'}, no_leader=True)
        self.assertIsNone(error)
        paused = [row for row in recorder.rows if row['control_mode'] == 'align']
        self.assertTrue(paused)
        self.assertTrue(all(row['command_source'] == 'hold' for row in paused))
        self.assertTrue(report['operator_pause_used'])
        self.assertFalse(report['assistance_marked'])
        self.assertTrue(any(e['event'] == 'model_control_resumed' for e in report['operator_events']))
        self.assertEqual(report['end_reason'], 'operator_q')
        stopper.assert_called_once()

    def test_operator_confirms_success_during_takeover_and_keeps_terminal_frame(self):
        report, recorder, stopper, _, error = self.simulate({1: 'h', 4: '\r', 10: 's'})
        self.assertIsNone(error)
        self.assertEqual(report['end_reason'], 'operator_confirmed_placement')
        self.assertEqual(report['outcome'], 'assisted_completion_requires_review')
        self.assertEqual(report['success_evidence']['terminal_frame_index'], len(recorder.rows) - 1)
        self.assertEqual(recorder.rows[-1]['command_source'], 'leader')
        stopper.assert_called_once()
        self.assertEqual(stopper.call_args.kwargs['reason'], 'task_completed')
        self.assertTrue(stopper.call_args.kwargs['return_to_zero'])

    def test_timeout_while_waiting_to_resume_holds_until_operator_exits(self):
        report, recorder, stopper, _, error = self.simulate(
            {1: 'h', 3: '\r', 5: 'h', 16: 'q'}, delay=.20, seconds=.16)
        self.assertIsNone(error)
        waiting = next(event for event in report['operator_events']
                       if event.get('reason') == 'model_duration_budget_exhausted')
        rows = recorder.rows[waiting['next_frame_index']:]
        self.assertTrue(rows)
        self.assertTrue(all(row['command_source'] == 'hold' for row in rows))
        self.assertEqual(report['end_reason'], 'operator_q')
        stopper.assert_called_once()

    def test_stale_leader_pauses_then_reanchors_without_catchup_motion(self):
        report,r,_,_,error=self.simulate({1:'h',2:'\r',9:'\r',14:'q'},lose_leader=(.16,.28))
        self.assertIsNone(error)
        stale=[x for x in r.rows if .16<=x['seconds']<.28]
        self.assertTrue(stale)
        self.assertTrue(all(x['control_mode']=='align' and x['command_source']=='hold' for x in stale))
        starts=[v for k,v in r.journals if k=='event' and v['event']=='intervention_start']
        self.assertEqual(len(starts),2)
        self.assertEqual(starts[1]['intervention_id'],2)

    def test_escape_disables_without_homing_from_human_mode(self):
        report,_,stopper,_,error=self.simulate({1:'h',2:'\r',5:'\x1b'})
        self.assertIsNone(error)
        self.assertEqual(report['end_reason'],'escape')
        self.assertFalse(stopper.call_args.kwargs['return_to_zero'])

    def test_writer_failure_runs_stop_and_preserves_prior_frames(self):
        report,r,stopper,_,error=self.simulate({1:'h',2:'\r',9:'q'},record_failure=4)
        self.assertIsInstance(error,RecordingError)
        self.assertEqual(len(r.rows),4)
        self.assertEqual(report['end_reason'],'recording_error')
        stopper.assert_called_once()

    def test_model_failure_pauses_for_manual_recovery_instead_of_exiting(self):
        report,r,_,_,error=self.simulate({7:'\r',12:'q'},failure=True)
        self.assertIsNone(error)
        self.assertEqual(report['end_reason'],'operator_q')
        self.assertTrue(any(row['control_mode']=='align' for row in r.rows))
        self.assertTrue(any(row['control_mode']=='human' for row in r.rows))

    def test_camera_loss_is_marked_and_model_holds(self):
        for role in ('overhead', 'wrist'):
            with self.subTest(role=role):
                report, r, _, _, error = self.simulate({8: 'h', 15: 'q'}, missing_camera=(role, .16, .28))
                self.assertIsNone(error)
                missing = [row for row in r.rows if not row['camera_valid'][role]]
                self.assertTrue(missing)
                self.assertTrue(all(row['control_mode'] == 'align' and row['command_source'] == 'hold' for row in missing))
                self.assertTrue(any(e['event'] == 'camera_feedback_unavailable' and e['cameras'] == [role]
                                    for e in report['operator_events']))

    def test_wrist_camera_loss_does_not_take_back_human_control(self):
        report, recorder, stopper, _, error = self.simulate(
            {1: 'h', 2: '\r', 8: 'h', 15: 'q'}, missing_camera=('wrist', .16, .28))
        self.assertIsNone(error)
        missing = [row for row in recorder.rows if not row['camera_valid']['wrist']]
        self.assertTrue(missing)
        self.assertTrue(all(row['command_source'] == 'leader' for row in missing))
        self.assertEqual(report['end_reason'], 'operator_q')
        stopper.assert_called_once()

    def test_model_duration_budget_does_not_end_a_human_correction_midway(self):
        report,r,_,_,error=self.simulate({1:'h',2:'\r',80:'h',100:'q'})
        self.assertIsNone(error)
        self.assertGreater(r.rows[-1]['seconds'],3.)
        self.assertLess(report['model_active_seconds'],3.)
        self.assertEqual(report['end_reason'],'operator_q')

    def test_can_failure_disables_without_automatic_return(self):
        report,r,stopper,_,error=self.simulate({1:'h',2:'\r'},follower_fault_at=5)
        self.assertIsInstance(error,DeviceCommunicationError)
        self.assertEqual(report['end_reason'],'device_fault')
        self.assertFalse(stopper.call_args.kwargs['return_to_zero'])


class FreshnessTests(unittest.TestCase):
    def test_correction_capture_marks_black_placeholder_for_either_missing_camera(self):
        for role in ('wrist', 'overhead'):
            with self.subTest(role=role), patch('rebot_capture.perf_counter', return_value=1.):
                cameras = {name: SimpleNamespace(frame_lock=Lock(), width=64, height=48,
                          latest_frame=np.ones((48, 64, 3), np.uint8), latest_timestamp=1.)
                           for name in ('wrist', 'overhead')}
                cameras[role].latest_timestamp = 0.
                images, stamps = run.capture(cameras)
                self.assertIsNone(stamps[role])
                self.assertEqual(images[role].shape, (48, 64, 3))
                self.assertFalse(np.any(images[role]))
                other = 'wrist' if role == 'overhead' else 'overhead'
                self.assertIs(images[other], cameras[other].latest_frame)

    def test_passive_connection_does_not_run_origin_reset_configuration(self):
        leader=SimpleNamespace(is_calibrated=True,is_connected=False,
            config=SimpleNamespace(joint_ids=dict(zip(JOINTS,range(7)))),configure=Mock())
        def connect(*,calibrate):
            self.assertFalse(calibrate)
            leader.configure()
            leader.is_connected=True
        leader.connect=Mock(side_effect=connect)
        leader.disconnect=Mock()
        reader=SimpleNamespace(start=Mock(),wait_ready=Mock(),close=Mock())
        with patch('lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader.RebotArm102Leader',return_value=leader), \
             patch('b601_leader_source.LeaderReader',return_value=reader):
            with passive_leader('COM_TEST') as active:
                self.assertIs(active,reader)
        leader.configure.assert_not_called()
        leader.connect.assert_called_once_with(calibrate=False)
        reader.close.assert_called_once()
        leader.disconnect.assert_called_once()

    def test_reliable_cached_sample_is_rejected_after_age_or_new_read_error(self):
        reader=LeaderReader(SimpleNamespace())
        reader._sample=LeaderSample(np.zeros(7),np.zeros(7),10.,[True]*7)
        reader._error=None
        with patch('b601_leader_source.perf_counter',return_value=10.1):
            self.assertIsNotNone(reader.latest()[0])
        with patch('b601_leader_source.perf_counter',return_value=11.):
            self.assertIsNone(reader.latest()[0])
        reader._error='unreliable UART'
        with patch('b601_leader_source.perf_counter',return_value=10.1):
            self.assertIsNone(reader.latest()[0])


class RecorderTests(unittest.TestCase):
    def test_real_recorder_through_model_align_human_resume_loop(self):
        import time
        from test_b601_official_stop import fake_follower
        follower = fake_follower(POSE)
        follower.configure = Mock()
        cameras = {name:SimpleNamespace(width=640,height=480) for name in ('wrist','overhead')}
        images = {name:np.zeros((480,640,3),np.uint8) for name in cameras}
        counter = SimpleNamespace(value=-1)
        def key():
            counter.value += 1
            return {3:'h',6:'\r',16:'h',30:'q'}.get(counter.value)
        def latest():
            angles = np.full(7,counter.value*.2)
            return LeaderSample(angles,angles.copy(),time.perf_counter(),[True]*7),None
        client = SimpleNamespace(infer=lambda request:{'actions':np.repeat((POSE/DIRECTION)[None,:],25,axis=0)},close=Mock())
        worker = run.InferenceWorker(client,'aligned')
        report = {'physical_trial_started':False}
        detector = Mock()
        detector.update.return_value = {'success':False}
        detector.new_window.return_value = detector
        with TemporaryDirectory() as folder:
            recorder=CorrectionRecorder(Path(folder)/'offline-loop',follower,cameras,'offline test',25,{'verification_only':True})
            with patch.object(run,'capture',side_effect=lambda c:(images,dict.fromkeys(cameras,time.perf_counter()))), \
                 patch.object(run,'PlacementDetector',return_value=detector), \
                 patch.object(run.policy,'stop_follower',return_value={'completed_without_errors':True}), \
                 redirect_stdout(StringIO()):
                try:
                    run.run_session(follower,cameras,worker,SimpleNamespace(latest=latest),recorder,
                                    POSE,5,True,report,keys=SimpleNamespace(poll=key))
                finally:
                    result=recorder.close('offline_verification',report)
            self.assertEqual(result['status'],'saved',result['errors'])
            self.assertEqual(set(result['frames_by_control_mode']),{'model','align','human','resuming'})
            self.assertEqual(result['media_validation']['decoded_video_frames']['wrist'],result['frames_written'])
            self.assertEqual(result['media_validation']['decoded_video_frames']['overhead'],result['frames_written'])
            rows=[json.loads(s) for s in (recorder.root/'telemetry-episode-000000.jsonl').read_text().splitlines()]
            for row in rows:
                np.testing.assert_allclose(row['action_adapter_deg'],np.array(row['target_motor_deg'])/DIRECTION)
            events=[json.loads(s) for s in (recorder.root/'events.jsonl').read_text().splitlines()]
            self.assertTrue({'intervention_start','intervention_end','model_control_resumed'} <= {e['event'] for e in events})

    def test_real_lerobot_encoder_keeps_both_videos_and_seven_axis_actions_aligned(self):
        import time
        from test_b601_official_stop import fake_follower
        follower = fake_follower(POSE)
        cameras = {name:SimpleNamespace(width=64,height=48) for name in ('wrist','overhead')}
        with TemporaryDirectory() as folder:
            root=Path(folder)/'offline-codec-check'
            r=CorrectionRecorder(root,follower,cameras,'offline verification only',25,{'verification_only':True})
            started=time.perf_counter()
            try:
                for i in range(25):
                    observation={f'{j}.pos':float(POSE[k]) for k,j in enumerate(JOINTS)}
                    images={name:np.full((48,64,3),i*8,np.uint8) for name in cameras}
                    mode=['model','align','human','resuming'][min(i//7,3)]
                    r.frame(observation,images,{'seconds':i*.04,'controller_tick':i,'control_mode':mode,'measured_motor_deg':POSE.tolist(),
                            'camera_valid':dict.fromkeys(cameras,True),'target_motor_deg':POSE.tolist()})
                    time.sleep(max(0,started+(i+1)*.04-time.perf_counter()))
            finally:
                result=r.close('offline_verification',{'end_reason':'operator_q','assistance_marked':True})
            self.assertEqual(result['status'],'saved',result['errors'])
            self.assertEqual(result['media_validation']['decoded_video_frames'],{'wrist':25,'overhead':25})
            self.assertEqual(result['media_validation']['state_action_rows'],25)
            self.assertEqual(result['skipped_control_ticks'],0)
            self.assertEqual(sum(result['frames_by_control_mode'].values()),25)
            self.assertFalse(result['ready_for_training_without_review'])

    def test_journal_and_action_coordinates_drain_before_finalizing(self):
        fake_dataset=SimpleNamespace(save_episode=Mock(),finalize=Mock())
        follower=SimpleNamespace(config=official_config())
        appended=[]
        with TemporaryDirectory() as folder, \
             patch('b601_correction_recording.open_dataset') as opened, \
             patch('b601_correction_recording.validate_saved_media',return_value={'offline_mock':True}), \
             patch('b601_correction_recording.append_frame',side_effect=lambda *args:appended.append(args)):
            root=Path(folder)/'trial'
            def create(*a): root.mkdir();return fake_dataset
            opened.side_effect=create
            r=CorrectionRecorder(root,follower,{},'task',25,{})
            r.journal('event',{'event':'intervention_start','seconds':0.})
            images={name:np.zeros((48,64,3),np.uint8) for name in ('overhead','wrist')}
            for i in range(3):
                r.frame({},images,{'seconds':i*.04,'target_motor_deg':POSE.tolist(),'control_mode':'human'})
            saved=r.close('assisted_completion_requires_review',{'end_reason':'operator_q','assistance_marked':True})
            self.assertEqual(saved['status'],'saved')
            self.assertEqual(saved['frames_written'],3)
            rows=[json.loads(s) for s in (root/'telemetry-episode-000000.jsonl').read_text().splitlines()]
            self.assertEqual([x['frame_index'] for x in rows],[0,1,2])
            np.testing.assert_allclose(rows[0]['action_adapter_deg'],POSE/DIRECTION)
            np.testing.assert_allclose(list(appended[0][3].values()),POSE/DIRECTION)
            self.assertTrue({'overhead','wrist'} <= set(appended[0][2]))
            self.assertEqual(len((root/'events.jsonl').read_text().splitlines()),1)
            fake_dataset.save_episode.assert_called_once()
            fake_dataset.finalize.assert_called_once()

    def test_writer_error_retains_partial_journal_and_never_marks_episode_complete(self):
        dataset=SimpleNamespace(save_episode=Mock(),finalize=Mock())
        follower=SimpleNamespace(config=official_config())
        with TemporaryDirectory() as folder, patch('b601_correction_recording.open_dataset') as opened, \
             patch('b601_correction_recording.append_frame',side_effect=OSError('injected full disk')):
            root=Path(folder)/'trial'
            def create(*a):root.mkdir();return dataset
            opened.side_effect=create
            r=CorrectionRecorder(root,follower,{},'task',25,{})
            r.frame({},dict.fromkeys(('overhead','wrist'),np.zeros((48,64,3),np.uint8)),
                    {'target_motor_deg':POSE.tolist()})
            r.finished.wait(2)
            result=r.close('not_completed',{'end_reason':'recording_error'})
            self.assertEqual(result['status'],'partial')
            self.assertTrue((root/'telemetry-episode-000000.partial.jsonl').exists())
            dataset.save_episode.assert_not_called()


class LauncherTests(unittest.TestCase):
    def test_describe_is_hardware_free_even_when_launcher_passes_execute(self):
        with patch.object(run,'passive_leader') as leader, patch.object(run.policy,'check_health') as health, \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(run.main(['--execute','--describe']),0)
        leader.assert_not_called()
        health.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())['leader_origin_writes'])


if __name__=='__main__':
    unittest.main()
