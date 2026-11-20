"""Seeed RS example 9 on the already-enabled LeRobot MIT connection.

reBotArm_control_py @ 3524ebc5316985d6d74fe30269ae771fea715507,
config/rebotarm_rs.yaml, gravity_compensation.profiles.basic.
The drive keeps running MIT throughout. One thread owns the CAN connection.
"""

import hashlib
from collections import deque
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from lerobot_robot_seeed_b601.gravity import _axis_angle_R, _rpy_to_R, compute_generalized_gravity, load_dynamics_model

from b601_gripper_control import HAND_GUIDED_GRIPPER_KD
from b601_preflight import JOINTS, read_follower


FADE_SECONDS = 0.5
TAU_SCALE = np.array([1., .98, .98, 1., 1., 1.])
OFFICIAL_REV = "3524ebc5316985d6d74fe30269ae771fea715507"


def smoothstep(elapsed):
    ratio = float(np.clip(elapsed / FADE_SECONDS, 0., 1.))
    return ratio * ratio * (3. - 2. * ratio)


def joint_pose(observation):
    return np.array([observation[f"{j}.pos"] for j in JOINTS], dtype=float)


class GravityGuide:
    def __init__(self, follower):
        self.follower = follower
        cfg = follower.config
        self.path = Path(cfg.gravity_urdf_path)
        self.model = load_dynamics_model(urdf_path=str(self.path))
        joints = [(j.name, j.q_idx) for j in self.model.joints if j.jtype in ("revolute", "continuous")]
        if joints != [(f"joint{i+1}", i) for i in range(6)]:
            raise ValueError(f"RS URDF 六轴与电机顺序不符：{joints}")
        self.direction = np.array([cfg.joint_directions[j] for j in JOINTS])
        self.lower, self.upper = np.array([cfg.joint_limits[j] for j in JOINTS]).T
        self.stiff_kp = np.array([cfg.mit_kp[j] for j in JOINTS[:-1]])
        self.stiff_kd = np.array([cfg.mit_kd[j] for j in JOINTS[:-1]])
        self.kp, self.kd = self.stiff_kp.copy(), self.stiff_kd.copy()
        self.scale = 0.
        self.offset = np.zeros(6)
        self.gripper_hold = None
        self.phase = "off"
        self.chain = [(j.origin_xyz, _rpy_to_R(*j.origin_rpy), j.axis) for j in self.model.joints
                      if j.jtype in ("revolute", "continuous")]

    def gravity(self, positions):
        # mechPos is already in the RS URDF frame. Adapter signs only apply to commands.
        q = np.zeros(self.model.nq)
        q[:6] = np.radians(positions[:6])
        tau = compute_generalized_gravity(model=self.model, q=q)[:6] * TAU_SCALE
        if not np.isfinite(tau).all():
            raise ValueError("RS 重力前馈返回非有限力矩")
        return tau

    def signature(self):
        return {"profile": "RS/basic + measured-load handoff", "revision": OFFICIAL_REV,
                "hand_guided_gripper_mit_kd": HAND_GUIDED_GRIPPER_KD,
                "urdf_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "calibration_sha256": hashlib.sha256(self.follower.calibration_fpath.read_bytes()).hexdigest(),
                "control_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def _switch(self, phase, now):
        self.phase, self.started = phase, now
        self.initial_kp, self.initial_kd, self.initial_scale = self.kp.copy(), self.kd.copy(), self.scale
        self.initial_offset = self.offset.copy()

    def begin(self, last, now):
        self.entry = last.copy()
        self.gripper_hold = None
        self.loads = deque()
        self.moving_since = None
        self._switch("capture_load", now)

    def _moved(self, positions, now):
        # Seeed example 10: WORLD spatial velocity, finite-difference encoders and LPF.
        q = np.radians(positions[:6])
        dt = now - self.previous_time
        if 0. < dt <= .1:
            self.qd += -np.expm1(-dt / .03) * ((q - self.previous_q) / dt - self.qd)
        else:
            self.qd.fill(0.)
        self.previous_q, self.previous_time = q, now
        rot, origin = np.eye(3), np.zeros(3)
        linear, angular = np.zeros(3), np.zeros(3)
        for (xyz, local_rot, axis), angle, speed in zip(self.chain, q, self.qd, strict=True):
            origin = origin + rot @ xyz
            rot = rot @ local_rot
            omega = (rot @ axis) * speed
            linear += np.cross(origin, omega)
            angular += omega
            rot = rot @ _axis_angle_R(axis, angle)
        # Also recognize a slow, sustained hand movement that never crosses an instantaneous speed.
        travelled = np.max(np.abs(q - self.entry_q))
        return np.linalg.norm(linear) > .04 or np.linalg.norm(angular) > .08 or travelled > .08 * FADE_SECONDS

    def anchor(self, last, now):
        # Capture the actual hand-guided aperture, or keep an explicitly latched grip under load.
        self.anchor_measured = np.array(list(read_follower(self.follower).values()))
        self.hold = np.clip(self.anchor_measured, self.lower, self.upper)
        if self.phase == "drag":
            if self.gripper_hold is not None:
                self.hold[-1] = self.gripper_hold
        else:
            self.hold[-1] = last[-1]
        self.previous_command = last.copy()
        self._switch("holding", now)
        return self.hold.copy()

    def toggle_gripper_hold(self, measured_deg):
        if self.phase != "drag":
            raise RuntimeError("只有手拖期间可以锁住夹爪开度")
        self.gripper_hold = (float(np.clip(measured_deg, self.lower[-1], self.upper[-1]))
                             if self.gripper_hold is None else None)
        return self.gripper_hold

    def release(self, now):
        if self.phase == "holding":
            self._switch("releasing", now)

    def settled(self, now):
        return self.phase != "holding" or now - self.started >= FADE_SECONDS

    def send(self, positions, planned, now, observation=None):
        if self.phase == "off":
            raise RuntimeError("重力切换尚未开始")
        gravity = self.gravity(positions)
        if self.phase == "capture_load":
            observation = observation if observation is not None else self.follower.get_observation()
            load = np.array([observation[f"{j}.torque"] for j in JOINTS[:-1]])
            if not np.isfinite(load).all():
                raise ValueError("切换时没有可靠的六轴实测保持力矩")
            self.loads.append((now, load - gravity))
            while self.loads[0][0] < now - FADE_SECONDS / 2:
                self.loads.popleft()
            if now - self.started >= FADE_SECONDS:
                self.load_offset = np.median([sample for _, sample in self.loads], axis=0)
                self.previous_q, self.previous_time = np.radians(positions[:6]), now
                self.entry_q = self.previous_q.copy()
                self.qd = np.zeros(6)
                self._switch("entering", now)
        b = smoothstep(now - self.started)
        if self.phase == "capture_load":
            # Identical MIT command while the existing position loop holds and measures the load.
            end_kp, end_kd, end_scale = self.initial_kp, self.initial_kd, self.initial_scale
            end_offset = self.initial_offset
            target = self.entry.copy()
        elif self.phase in ("entering", "drag"):
            end_kp, end_kd, end_scale = 15., 2.5, 1.
            end_offset = self.load_offset
            target = self.entry.copy()
            moved = self._moved(positions, now)
            if b == 1.:
                self.phase = "drag"
                if moved and self.moving_since is None:
                    self.moving_since = now
                if self.moving_since is not None:
                    # Release the entry load correction smoothly when the arm leaves its support.
                    end_offset = self.load_offset * (1. - smoothstep(now - self.moving_since))
            aperture = positions[-1] if self.gripper_hold is None else self.gripper_hold
            target[-1] += b * (aperture - self.entry[-1])
        else:
            end_kp, end_kd = self.stiff_kp, self.stiff_kd
            end_scale = self.initial_scale if self.phase == "holding" else 0.
            end_offset = self.initial_offset if self.phase == "holding" else np.zeros(6)
            target = self.hold.copy() if self.phase == "holding" else planned.copy()
        self.kp = self.initial_kp + b * (end_kp - self.initial_kp)
        self.kd = self.initial_kd + b * (end_kd - self.initial_kd)
        # Blend the position-loop *effort* once. Blending both its gains and its
        # error linearly would remove the old supporting torque twice under load.
        carry = (1. - b) * self.initial_kp / self.kp
        if self.phase in ("entering", "drag"):
            target[:6] = positions[:6] + carry * (self.entry[:6] - positions[:6])
        elif self.phase == "holding":
            target[:6] += carry * (self.previous_command[:6] - self.hold[:6])
        self.scale = self.initial_scale + b * (end_scale - self.initial_scale)
        self.offset = self.initial_offset + b * (end_offset - self.initial_offset)
        tau = gravity * self.scale + self.offset
        action = {f"{j}.pos": float(v) for j, v in zip(JOINTS, target / self.direction, strict=True)}
        accepted = self.follower.send_gravity_action(action, self.kp, self.kd, tau,
            hand_guided_gripper=self.phase == "drag" and self.gripper_hold is None)
        if self.phase == "releasing" and b == 1.:
            self.phase = "off"
        return np.array([accepted[f"{j}.pos"] for j in JOINTS])

    def prepare_return(self, last, poll):
        """Transfer the current load before normal SDK homing. Esc still interrupts."""
        if self.phase != "holding":
            last = self.anchor(last, perf_counter())
        while self.phase != "off":
            if poll() == "\x1b":
                raise InterruptedError("回零交接期间按下 Esc")
            obs = self.follower.get_observation()
            if self.phase == "holding" and self.settled(perf_counter()):
                self.release(perf_counter())
            last = self.send(joint_pose(obs), self.hold, perf_counter(), obs)
        return last

    def refresh_until(self, deadline, last):
        # Same CAN writer as the acquisition loop. Use spare time between video frames.
        # When returning to model control the model delta is applied only once per video tick.
        while self.phase != "off" and deadline - perf_counter() > .012:
            obs = self.follower.get_observation()
            last = self.send(joint_pose(obs), last, perf_counter(), obs)
        sleep(max(0., deadline - perf_counter()))
        return last
