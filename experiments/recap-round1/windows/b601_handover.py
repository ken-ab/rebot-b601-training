"""Control ownership and relative clutching; this module never opens hardware."""

import math
import numpy as np


MODES = ("model", "align", "human", "entering_drag", "drag", "resuming")


def vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("需要七轴有限角度")
    return result.copy()


class Handover:
    """A clutch captures the current command, never moves either arm to a zero."""

    def __init__(self, direction):
        self.direction = vector(direction)
        if np.any(self.direction == 0):
            raise ValueError("关节方向和倍率不能为零")
        self.mode = "model"
        self.epoch = 0
        self.intervention = 0
        self.human_used = False
        self.anchor_motor = None
        self.anchor_leader = None
        self.previous_leader = None
        self.displacement = np.zeros(7)
        self.hold_first_model_tick = False

    def request_human(self):
        self.epoch += 1
        self.mode = "align"
        self.previous_leader = None
        self.hold_first_model_tick = False

    def confirm_alignment(self, leader, last_command):
        if self.mode != "align":
            raise RuntimeError("只有对齐待命时可以开始主臂接管")
        self.anchor_leader = vector(leader)
        self.previous_leader = self.anchor_leader.copy()
        self.anchor_motor = vector(last_command)
        self.displacement = np.zeros(7)
        self.intervention += 1
        self.human_used = True
        self.mode = "human"

    def request_drag(self):
        self.request_human()
        self.mode = "entering_drag"

    def drag_ready(self):
        if self.mode != "entering_drag":
            raise RuntimeError("未开始手拖切换")
        self.intervention += 1
        self.mode = "drag"
        self.human_used = True

    def human_target(self, leader):
        if self.mode != "human":
            raise RuntimeError("主臂尚未取得控制权")
        current = vector(leader)
        delta = np.array([math.remainder(float(a - b), 360.0)
                          for a, b in zip(current, self.previous_leader, strict=True)])
        self.displacement += delta
        self.previous_leader = current
        return self.anchor_motor + self.direction * self.displacement

    def request_model(self):
        self.epoch += 1
        self.mode = "resuming"
        self.previous_leader = None
        self.hold_first_model_tick = True

    def accept_model(self, epoch):
        if self.mode != "resuming" or epoch != self.epoch:
            return False
        self.mode = "model"
        return True

    def alignment_reference(self, actual_motor):
        return vector(actual_motor) / self.direction


class CorrectionKeys:
    """H, G, Enter and S are edge triggered; Esc takes priority over other keys."""

    def __init__(self, available=None, read=None, held=None):
        if available is None:
            import ctypes
            import msvcrt
            available, read = msvcrt.kbhit, msvcrt.getwch
            held = lambda code: bool(ctypes.windll.user32.GetAsyncKeyState(code) & 0x8000)
        self.available, self.read, self.held = available, read, held
        self.virtual_keys = {"h": ord("H"), "g": ord("G"), "\r": 13, "s": ord("S")}
        self.latched = dict.fromkeys(self.virtual_keys, False)

    def poll(self):
        released = {key for key, latched in self.latched.items()
                    if latched and not self.held(self.virtual_keys[key])}
        seen, stop = set(), None
        while self.available():
            key = self.read()
            if key in ("\x00", "\xe0"):
                if self.available():
                    self.read()
                continue
            key = key.lower()
            if key == "\x1b":
                stop = key
            elif key == "q" and stop != "\x1b":
                stop = key
            elif key in self.latched and key not in released and not self.latched[key]:
                seen.add(key)
                self.latched[key] = True
        for key in released:
            self.latched[key] = False
        if stop:
            return stop
        return next((key for key in ("s", "h", "\r", "g") if key in seen), None)
