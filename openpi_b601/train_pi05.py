"""Launch the pinned OpenPI trainer for the prepared B601 dataset."""

import argparse
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
from unittest.mock import patch

import numpy as np

from b601_data import ACTION_HORIZON, B601Inputs, B601Outputs, FPS, JOINTS, TRAIN_ID, VAL_ID


ROOT = Path(__file__).resolve().parent
CONFIG_NAME = "pi05_b601_lora"
BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"


def build_config(args, prepared):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, optimizer, weight_loaders

    model = Pi0Config(pi05=True, action_horizon=ACTION_HORIZON,
                      paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    smoke = args.mode == "smoke"
    steps = args.steps if args.steps is not None else (20 if smoke else 10_000)
    batch = args.batch_size if args.batch_size is not None else (1 if smoke else 8)
    workers = args.workers if args.workers is not None else (0 if smoke else 2)
    base_params = str(args.base_params.resolve()) if args.base_params else BASE_PARAMS

    return config.TrainConfig(
        name=CONFIG_NAME, exp_name=args.exp_name or datetime.now().strftime(f"{args.mode}-%Y%m%d-%H%M%S"),
        model=model, freeze_filter=model.get_freeze_filter(), ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(base_params),
        data=config.SimpleDataConfig(
            repo_id=TRAIN_ID,
            assets=config.AssetsConfig(assets_dir=str(prepared / "assets"), asset_id=TRAIN_ID),
            base_config=config.DataConfig(prompt_from_task=True),
            data_transforms=lambda _: transforms.Group(inputs=[B601Inputs()], outputs=[B601Outputs()]),
        ),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(500, max(1, steps // 10)), peak_lr=2.5e-5,
            decay_steps=max(steps, 2), decay_lr=2.5e-6,
        ),
        batch_size=batch, num_train_steps=steps, num_workers=workers,
        log_interval=1 if smoke else 100, save_interval=steps if smoke else 1000,
        checkpoint_base_dir=str(ROOT / "runs"), wandb_enabled=False, resume=args.resume,
        policy_metadata={"robot_type": "seeed_b601_rs_follower", "fps": FPS, "action_names": list(JOINTS),
                         "action_space": "absolute_adapter_input"},
    )


def check_inputs(config, manifest):
    from openpi.training import data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None or set(data_config.norm_stats) != {"state", "actions"}:
        raise ValueError("Missing training-only OpenPI normalization statistics")
    for split, repo_id in (("train", TRAIN_ID), ("val", VAL_ID)):
        # Reuse training normalization when reading held-out data.
        split_config = dataclasses.replace(data_config, repo_id=repo_id)
        raw = data_loader.create_torch_dataset(split_config, ACTION_HORIZON, config.model)
        if len(raw) != manifest[split]["frames"]:
            raise ValueError(f"{split} frame count does not match manifest")
        dataset = data_loader.transform_dataset(raw, split_config)
        for index in (0, len(dataset) - 1):
            sample = dataset[index]
            if (sample["state"].shape != (config.model.action_dim,) or
                    sample["actions"].shape != (ACTION_HORIZON, config.model.action_dim) or
                    not np.isfinite(sample["state"]).all() or not np.isfinite(sample["actions"]).all()):
                raise ValueError(f"Invalid OpenPI state or action chunk in {split} at {index}")
            expected_mask = {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
            if (sample["image_mask"] != expected_mask or sample["tokenized_prompt"].size == 0 or
                    any(rgb.shape != (224, 224, 3) for rgb in sample["image"].values())):
                raise ValueError(f"Invalid image or prompt mapping in {split} at {index}")
        print(f"{split}: {len(raw)} frames, images/prompt and [{ACTION_HORIZON}, {config.model.action_dim}] actions OK",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "train"))
    parser.add_argument("--base-params", type=Path, help="Local JAX pi05_base params directory")
    parser.add_argument("--exp-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.batch_size is not None and args.batch_size < 1 or
            args.steps is not None and args.steps < 1 or
            args.workers is not None and args.workers < 0):
        parser.error("batch size and steps must be positive, workers must be nonnegative")
    if args.resume and not args.exp_name:
        parser.error("--resume requires --exp-name")
    if args.exp_name and (Path(args.exp_name).name != args.exp_name or args.exp_name in (".", "..")):
        parser.error("--exp-name must be a single directory name")
    if args.base_params and not args.base_params.is_dir():
        parser.error(f"No local params directory at {args.base_params}")

    prepared = ROOT / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["output_version"], manifest["fps"], manifest["action_horizon"]) != ("v2.1", FPS, ACTION_HORIZON):
        parser.error("Dataset version, FPS, or action horizon does not match the B601 config")
    splits = json.loads((ROOT.parent / "splits.json").read_text(encoding="utf-8"))
    for split, repo_id in (("train", TRAIN_ID), ("val", VAL_ID)):
        if manifest[split]["source_episode_indices"] != splits[split]:
            parser.error(f"Prepared {split} differs from splits.json. Move {prepared} aside and rerun.")
        if not (prepared / "datasets" / repo_id / "meta/info.json").is_file():
            parser.error(f"Missing prepared dataset {repo_id}")
    os.environ["HF_LEROBOT_HOME"] = str(prepared / "datasets")

    import jax
    from lerobot.common.datasets import lerobot_dataset

    if lerobot_dataset.CODEBASE_VERSION != "v2.1":
        parser.error("Use the pinned OpenPI environment via run_pi05.sh")
    if args.mode != "check" and not any(device.platform == "gpu" for device in jax.devices()):
        parser.error("OpenPI JAX cannot see an NVIDIA GPU. Check nvidia-smi and CUDA before training.")

    config = build_config(args, prepared)
    # OpenPI calls the legacy loader without a video_backend argument. Pick PyAV
    # here so Linux does not default to a TorchCodec build needing system FFmpeg.
    with patch.object(lerobot_dataset, "get_safe_default_codec", return_value="pyav"):
        if args.mode != "train":
            check_inputs(config, manifest)
        if args.mode != "check":
            print(f"{config.name}: {config.num_train_steps} steps, batch {config.batch_size}, {FPS} Hz", flush=True)
            print(f"Checkpoints: {config.checkpoint_dir}", flush=True)
            runpy.run_path("scripts/train.py")["main"](config)


if __name__ == "__main__":
    main()
