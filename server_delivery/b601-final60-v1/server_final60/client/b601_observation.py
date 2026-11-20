"""Prepare a small B601 network request with exactly the server's RGB mapping."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'openpi_client_src'))
from openpi_client.image_tools import resize_with_pad

CAMERAS = ('observation.images.overhead', 'observation.images.wrist')


def prepare_observation(observation):
    """Return224RGBuint8 images, raw seven-axis feedback and optional prompt.

    The float conversion uses B601Inputs' np.rint convention. Upstream
    convert_to_uint8 truncates floats instead and must not replace this step.
    """
    result = {}
    for key in CAMERAS:
        image = np.asarray(observation[key])
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Expected RGB HWC or CHW: ' + key)
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all():
                raise ValueError('RGB contains nonfinite values: ' + key)
            image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.dtype != np.uint8:
            raise ValueError('Expected RGB uint8 or float[0,1]: ' + key)
        result[key] = np.ascontiguousarray(resize_with_pad(image, 224, 224))
    state = np.asarray(observation['observation.state'], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('Expected seven finite raw motor-side feedback values')
    result['observation.state'] = state
    if 'prompt' in observation:
        if not isinstance(observation['prompt'], str):
            raise ValueError('Prompt must be a string')
        result['prompt'] = observation['prompt']
    return result
