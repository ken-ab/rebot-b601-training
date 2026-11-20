"""Seeed RS host PD with measured dt instead of an assumed 0.02 seconds.

Based on Seeed-Projects/lerobot-robot-seeed-b601, revision
0da1a1a5575905d77823227003ffee7672059780, seeed_b601_rs_follower.py.
The gains, filter coefficient, velocity saturation, stall conditions and
torque limits below are the installed official implementation's values.
Position control uses the measured interval for both finite differences.
Free hand guidance uses the drive's MIT damping with a lighter project gain.
"""

import logging
from time import perf_counter

logger = logging.getLogger(__name__)


HAND_GUIDED_GRIPPER_KD = 1.0


class HandGuidedGripperMotor:
    """Forward SDK reads and commands, changing only the free J7 velocity gain."""

    def __init__(self, motor):
        self._motor = motor

    def __getattr__(self, name):
        return getattr(self._motor, name)

    def send_mit(self, pos, vel, kp, kd, tau):
        return self._motor.send_mit(pos, vel, kp, HAND_GUIDED_GRIPPER_KD, tau)


def measured_dt_gripper_torque(follower, motor, pos_target, *, hand_guided=False):
    if motor is None:
        return None
    follower.bus.poll_feedback_once()
    state = motor.get_state()
    if state is None:
        return None

    now = perf_counter()
    previous_time = getattr(follower, "_gripper_control_time", None)
    follower._gripper_control_time = now
    dt = None if previous_time is None else now - previous_time
    # On the first call, both finite differences are zero, so no guessed dt.
    follower._gripper_control_dt_s = dt
    if hand_guided:
        # The drive's MIT damping stays active. A hand on the jaws supplies the motion.
        # Discard derivative history so capturing the new aperture cannot kick the fingers.
        follower._gripper_prev_target_pos = None
        follower._gripper_prev_filtered_target_vel = None
        follower._gripper_prev_state_pos = state.pos
        follower._gripper_open_stall_count = 0
        motor.request_feedback()
        return 0.0
    prev_target = getattr(follower, "_gripper_prev_target_pos", None)
    target_vel = 0.0 if prev_target is None or dt is None or dt <= 0 else (pos_target - prev_target) / dt
    follower._gripper_prev_target_pos = pos_target
    prev_filtered = getattr(follower, "_gripper_prev_filtered_target_vel", None)
    filtered = target_vel if prev_filtered is None else .3 * target_vel + .7 * prev_filtered
    target_vel = max(-3.0, min(3.0, filtered))
    follower._gripper_prev_filtered_target_vel = target_vel

    prev_state = getattr(follower, "_gripper_prev_state_pos", None)
    state_vel = 0.0 if prev_state is None or dt is None or dt <= 0 else (state.pos - prev_state) / dt
    follower._gripper_prev_state_pos = state.pos
    torque = (float(follower.config.gripper_mit_kp) * (pos_target - state.pos)
              + float(follower.config.gripper_mit_kd) * (target_vel - state.vel))
    stalled = abs(state_vel) < .25
    follower._gripper_open_stall_count = (
        getattr(follower, "_gripper_open_stall_count", 0) + 1
        if pos_target - state.pos > .05 and stalled else 0)
    if follower._gripper_open_stall_count >= 5:
        logger.warning("gripper open-stall detected, cutting torque")
        motor.request_feedback()
        return 0.0
    max_torque = (follower.config.gripper_mit_torque_limit if not stalled
                  else follower.config.gripper_mit_hold_torque_limit)
    motor.request_feedback()
    return max(-max_torque, min(max_torque, torque))
