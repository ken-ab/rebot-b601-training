"""Shared dataset names and the B601 observation/action mapping for OpenPI."""

import dataclasses

import numpy as np


FPS = 25
ACTION_HORIZON = 25
JOINTS = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
    "wrist_yaw.pos", "wrist_roll.pos", "gripper.pos",
)
OVERHEAD = "observation.images.overhead"
WRIST = "observation.images.wrist"
TRAIN_ID = "rebot-b601-train"
VAL_ID = "rebot-b601-val"


def rgb_uint8(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or [0, 1] float RGB, got {image.dtype}")
    return image


@dataclasses.dataclass(frozen=True)
class B601Inputs:
    def __call__(self, sample):
        overhead = rgb_uint8(sample[OVERHEAD])
        wrist = rgb_uint8(sample[WRIST])
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        if state.shape != (len(JOINTS),):
            raise ValueError(f"Expected seven measured joints, got {state.shape}")
        result = {
            "state": state,
            "image": {"base_0_rgb": overhead, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(wrist)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "action" in sample:
            actions = np.asarray(sample["action"], dtype=np.float32)
            if actions.shape != (ACTION_HORIZON, len(JOINTS)):
                raise ValueError(f"Expected {ACTION_HORIZON} seven-joint actions, got {actions.shape}")
            # State is motor-side feedback. Action is already an absolute adapter input.
            result["actions"] = actions
        if "prompt" in sample:
            result["prompt"] = sample["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class B601Outputs:
    def __call__(self, result):
        return {"actions": np.asarray(result["actions"][..., :len(JOINTS)])}
