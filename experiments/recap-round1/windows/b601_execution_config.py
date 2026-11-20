"""Local deployment choices, independent of any training-server address."""

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).with_name("b601_execution.json")
ARM_DELTA_DEG = 0.3


def positive_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def load_execution_config(path=DEFAULT_CONFIG):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    mode = raw.get("policy_mode", "aligned")
    if mode not in ("aligned", "rtc"):
        raise ValueError("policy_mode must be aligned or rtc")
    return {
        **raw, "policy_mode": mode,
        "seconds": positive_number(raw.get("seconds", 180), "seconds"),
        "gripper_delta_motor_deg_per_step": positive_number(
            raw.get("gripper_delta_motor_deg_per_step", ARM_DELTA_DEG), "gripper_delta_motor_deg_per_step"),
    }


def motor_deltas(gripper_delta=ARM_DELTA_DEG):
    """Degrees in motor coordinates, after J7's x6 adapter conversion."""
    return np.array([ARM_DELTA_DEG] * 6 + [positive_number(gripper_delta, "gripper delta")])
