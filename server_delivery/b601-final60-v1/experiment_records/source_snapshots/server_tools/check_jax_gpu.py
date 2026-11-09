"""Exercise both assigned GPUs and a real NCCL collective before model loading."""
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

devices = jax.devices()
assert len(devices) == 2 and all(device.platform == 'gpu' for device in devices), devices
@functools.partial(jax.pmap, axis_name='assigned_gpus')
def collective(x):
    return jax.lax.psum(x, 'assigned_gpus')
result = np.asarray(collective(jnp.ones(2)))
assert np.array_equal(result, np.array([2, 2])), result
matrix = jax.device_put(jnp.ones((64, 64), dtype=jnp.bfloat16), devices[0])
product = np.asarray(matrix @ matrix, dtype=np.float32)
assert np.isfinite(product).all() and np.all(product == 64)
receipt = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__,
    'devices': [str(device) for device in devices], 'kinds': [device.device_kind for device in devices],
    'all_reduce': result.tolist(), 'bfloat16_matmul_finite': True}
(Path(__file__).resolve().parent / 'JAX_GPU_ACCEPTANCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
