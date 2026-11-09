"""Construct the custom B601 policy without relying on OpenPI's config registry."""
import dataclasses
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

from b601_data import TRAIN_ID
from train_pi05 import ROOT, build_config


INFERENCE_XLA_FLAGS = ('--xla_gpu_autotune_level=0', '--xla_gpu_deterministic_ops=true')


def configure_inference():
    """Bind reproducible inference kernels before JAX loads; training is separate."""
    flags = shlex.split(os.environ.get('XLA_FLAGS', ''))
    missing = []
    for required in INFERENCE_XLA_FLAGS:
        name = required.split('=', 1)[0]
        present = [flag for flag in flags if flag.split('=', 1)[0] == name]
        if present and any(flag != required for flag in present):
            raise RuntimeError(f'B601 reproducible inference requires {required}; conflicting XLA_FLAGS={present}')
        if not present:
            missing.append(required)
    if missing and 'jax' in sys.modules:
        raise RuntimeError('Import b601_policy before JAX, or set both required INFERENCE_XLA_FLAGS before importing JAX')
    os.environ['XLA_FLAGS'] = ' '.join([*flags, *missing])


configure_inference()


def make_config(checkpoint: Path | None = None, *, base_model: bool = False):
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig
    args = SimpleNamespace(mode='train', steps=10000, batch_size=1, workers=0, base_params=None,
        exp_name='b601-inference', resume=False, fsdp_devices=1, lr_decay_steps=10000, save_interval=None)
    config = build_config(args, ROOT / 'prepared')
    if checkpoint is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data,
            assets=AssetsConfig(assets_dir=str(checkpoint / 'assets'), asset_id=TRAIN_ID)))
    if base_model:
        config = dataclasses.replace(config, model=Pi0Config(pi05=True, action_horizon=25,
            paligemma_variant='gemma_2b', action_expert_variant='gemma_300m'))
    return config


def load_policy(checkpoint: Path, *, normalization_checkpoint: Path | None = None,
                base_model: bool = False, denoise_steps: int = 10):
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.checkpoints import load_norm_stats
    checkpoint = checkpoint.resolve()
    stats_checkpoint = (normalization_checkpoint or checkpoint).resolve()
    config = make_config(stats_checkpoint, base_model=base_model)
    stats = load_norm_stats(stats_checkpoint / 'assets', TRAIN_ID)
    return create_trained_policy(config, checkpoint, norm_stats=stats,
        default_prompt='拿起蓝色餐巾纸，把它放进黑色框里', sample_kwargs={'num_steps': denoise_steps})
