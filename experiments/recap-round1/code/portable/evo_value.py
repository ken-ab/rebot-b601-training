"""Transfer the public Evo-RL pistar06 checkpoint to the B601 capture, offline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import Pistar06Policy
from lerobot.values.pistar06.processor_pistar06 import (
    PISTAR06_IMAGES_KEY, PISTAR06_IMAGE_MASK_KEY, Pistar06PrepareTaskPromptProcessorStep,
)


TASK = "拿起蓝色餐巾纸，把它放进黑色框里"
PARENT_SHA = "58377dfa2a5463cbe7228fb02398fa419f024c8b81ded391bc3ace7258aa4ef9"
SOURCE_REV = "b1009248c270515946b96ca90ae5a156f103c637"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, data):
    part = p.with_name(p.name + ".partial")
    part.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    part.replace(p)


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def materialize_backbones(assets, dest, full_weights, expected_sha):
    """Reassemble native HF backbones from a complete Evo-RL value state."""
    details = read(assets / "sources.json")
    for name, expected in details["sha256"].items():
        if digest(assets / name) != expected:
            raise ValueError(f"Changed model asset {name}")
    if (dest / "ready.json").exists():
        if read(dest / "ready.json")["source_sha256"] != expected_sha:
            raise ValueError(f"Different value checkpoint already cached under {dest}")
        return dest
    if digest(full_weights) != expected_sha:
        raise ValueError("Complete value checkpoint SHA-256 mismatch")
    dest.mkdir(parents=True, exist_ok=True)
    for group, prefix in (("vision", "model.vision_encoder."), ("language", "model.language_model.")):
        target = dest / group
        target.mkdir(exist_ok=True)
        native_config = read(assets / group / "config.json")
        if group == "language":
            # Evo-RL stores the bare transformer, without a generative output head.
            native_config["architectures"] = ["Gemma3TextModel"]
        write(target / "config.json", native_config)
        if group == "vision":
            shutil.copy2(assets / "vision/preprocessor_config.json", target)
        with safe_open(str(full_weights), framework="pt", device="cpu") as handle:
            tensors = {k.removeprefix(prefix): handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix)}
            if len(tensors) < 100:
                raise ValueError(f"Incomplete {group} encoder")
            part = target / "model.partial.safetensors"
            save_file(tensors, str(part), metadata={"format": "pt"})
            part.replace(target / "model.safetensors")
    write(dest / "ready.json", {"source_sha256": expected_sha, "upstream_revision": SOURCE_REV})
    return dest


def load_native(root, device, export=None):
    source = root / "inputs/stacking"
    backbones = materialize_backbones(root/"inputs/model-assets", root/"inputs/backbones",
                                      source/"model.safetensors", PARENT_SHA)
    weights = export if export is not None else source/"model.safetensors"
    return native_from_file(source, backbones, weights, device)


def native_from_file(source, backbones, weights, device):
    cfg = PreTrainedConfig.from_pretrained(str(source))
    if not isinstance(cfg, Pistar06Config):
        raise ValueError(f"Expected the upstream pistar06 critic, got {type(cfg).__name__}")
    cfg.vision_repo_id, cfg.language_repo_id = (str(backbones / k) for k in ("vision", "language"))
    cfg.vision_revision = cfg.language_revision = None
    cfg.device = str(device)
    cfg.dtype = "bfloat16" if device.type == "cuda" else "float32"
    cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    cfg.camera_features = ["observation.images.top", "observation.images.wrist"]
    cfg.freeze_language_model = True
    cfg.freeze_vision_encoder = False
    model = Pistar06Policy(cfg)
    model.requires_grad_(False)
    vision = model.model.vision_encoder.vision_model
    modules = [model.model.image_projector, model.model.language_projector, model.model.final_norm,
               model.model.value_head, vision.encoder.layers[-1], vision.post_layernorm, vision.head]
    for module in modules:
        module.float().requires_grad_(True)
    # Enforce all backbone and value-head keys: mismatches must not start random training.
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(device)


class Episode:
    def __init__(self, root, record):
        self.record = record
        folder = root / "episodes" / record["id"]
        with np.load(folder / "rows.npz") as z:
            self.t, self.state, self.offsets = (z[k] for k in ("time", "state", "offsets"))
        if len(self.t) != len(self.state) or self.state.shape[1] != 7:
            raise ValueError(f"Invalid joint observations in {record['id']}")
        self.video = np.memmap(folder / "views.jpgs", mode="r", dtype=np.uint8)
        if int(self.offsets[-1, -1]) != self.video.size:
            raise ValueError(f"Truncated image pairs in {record['id']}")

    def images(self, i):
        a, b, c = self.offsets[i]
        images = []
        for left, right in ((a, b), (b, c)):
            image = cv2.imdecode(self.video[left:right], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image {self.record['id']} frame {i}")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(images).transpose(0, 3, 1, 2)


def observation_batch(measured, pictures, low, high, prompt, tokenizer, device):
    scale = high - low
    scale = np.where(scale == 0, 1e-8, scale)
    normalized = torch.from_numpy(2 * (measured - low) / scale - 1)
    transition = prompt({TransitionKey.OBSERVATION: {OBS_STATE: normalized},
                         TransitionKey.COMPLEMENTARY_DATA: {"task": [TASK]*len(measured)}})
    texts = transition[TransitionKey.COMPLEMENTARY_DATA]["task"]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=200, return_tensors="pt")
    batch = {PISTAR06_IMAGES_KEY: torch.from_numpy(np.ascontiguousarray(pictures)),
             OBS_LANGUAGE_TOKENS: tokens.input_ids, OBS_LANGUAGE_ATTENTION_MASK: tokens.attention_mask,
             PISTAR06_IMAGE_MASK_KEY: torch.ones(len(measured), 2, dtype=torch.bool)}
    return {k: v.to(device) for k, v in batch.items()}


class Corpus:
    def __init__(self, root, device):
        self.device = device
        rows = read(root / "manifest.json")["episodes"]
        self.episodes = [Episode(root, row) for row in rows]
        train = [ep for ep in self.episodes if ep.record["split"] == "train"]
        self.groups = [[ep for ep in train if ep.record["success"] == ok] for ok in (False, True)]
        self.failure_fraction = len(self.groups[0]) / len(train)
        states = np.concatenate([ep.state for ep in train], axis=0)
        quantiles = np.quantile(states, [.01, .99], axis=0).astype(np.float32)
        self.low, self.high = quantiles
        self.penalty = max(120., max(ep.record["seconds"] for ep in train))
        self.tokenizer = AutoTokenizer.from_pretrained(str(root / "inputs/model-assets/language"), local_files_only=True)
        self.prompt = Pistar06PrepareTaskPromptProcessorStep()
        self.norm = {"q01": self.low.tolist(), "q99": self.high.tolist(), "penalty_seconds": self.penalty,
                     "valid_tasks": [TASK], "camera_order": ["overhead", "wrist"], "joint_count": 7}
        for ep in self.episodes:
            ep.target = (-(ep.t[-1] - ep.t) - (0. if ep.record["success"] else self.penalty)) / (2*self.penalty)
            if ep.target.min() < -1.00001 or ep.target.max() > .00001:
                raise ValueError(f"Value target outside native bins: {ep.record['id']}")

    def batch(self, points):
        measured = np.stack([ep.state[i] for ep, i in points]).astype(np.float32)
        pictures = np.stack([ep.images(i) for ep, i in points])
        batch = observation_batch(measured,pictures,self.low,self.high,self.prompt,self.tokenizer,self.device)
        batch["observation.value_target"] = torch.tensor([ep.target[i] for ep, i in points],
                                                        dtype=torch.float32, device=self.device)
        return batch

    def sample(self, rng, size):
        # More failure examples per update, with inverse-probability weights to
        # preserve the original episode-uniform Monte Carlo objective.
        pts, weights = [], []
        for _ in range(size):
            fail = rng.random() < .35
            group = self.groups[int(not fail)]
            ep = group[int(rng.integers(len(group)))]
            pts.append((ep, int(rng.integers(len(ep.t)))))
            weights.append(self.failure_fraction/.35 if fail else (1-self.failure_fraction)/.65)
        batch = self.batch(pts)
        batch["observation.value_loss_weight"] = torch.tensor(weights, device=self.device, dtype=torch.float32)
        return batch


@torch.inference_mode()
def predictions(model, corpus, points):
    with torch.autocast(device_type=corpus.device.type, dtype=torch.bfloat16, enabled=corpus.device.type=="cuda"):
        pred = model.predict_value(corpus.batch(points)).float().cpu().numpy().reshape(-1)
    if pred.shape != (len(points),) or not np.isfinite(pred).all():
        raise ValueError("Invalid native critic predictions")
    return pred


@torch.inference_mode()
def evaluate(model, corpus, split, samples, batch):
    model.eval()
    rows = []
    for ep in corpus.episodes:
        if ep.record["split"] != split:
            continue
        grid = np.linspace(ep.t[0], ep.t[-1], min(len(ep.t), samples))
        ix = np.unique(np.searchsorted(ep.t, grid).clip(0, len(ep.t)-1))
        pts = [(ep, int(i)) for i in ix]
        pred = np.concatenate([predictions(model, corpus, pts[i:i+batch]) for i in range(0, len(pts), batch)])
        rows.append({"episode": ep.record["id"], "success": ep.record["success"], "frames": len(pts),
                     "mae": float(np.abs(pred - ep.target[ix]).mean()), "first": float(pred[0]),
                     "last": float(pred[-1]), "mean_value": float(pred.mean())})
    a, b = [float(np.mean([r["mae"] for r in rows if r["success"] == ok])) for ok in (True, False)]
    return {"macro_mae": float(np.mean([r["mae"] for r in rows])), "successful_episode_mae": a,
            "failed_episode_mae": b, "outcome_balanced_mae": (a+b)/2, "episodes": rows}


def learned(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def save_fit_state(path, model, optimizer, step):
    partial = path.with_suffix(".part")
    torch.save({"learned": {k: v.detach().cpu() for k, v in learned(model).items()},
                "optimizer": optimizer.state_dict(), "step": step, "cuda_rng": torch.cuda.get_rng_state(),
                "torch_rng": torch.get_rng_state()}, partial)
    partial.replace(path)


def apply_selected(model, weights):
    trainable = learned(model)
    if set(trainable) != set(weights):
        raise ValueError("Wrong Evo-RL trainable parameter set")
    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(weights[name])


def select_weights(root, model, step, result):
    path = root / "value/best.json"
    if path.exists() and read(path)["outcome_balanced_mae"] <= result["outcome_balanced_mae"]:
        return
    temp = root / "value/best.partial.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in learned(model).items()}, str(temp))
    temp.replace(root / "value/best.safetensors")
    write(path, {"step": step, **result})


def train(root, size, eval_batch):
    device = torch.device("cuda")
    torch.manual_seed(601)
    model = load_native(root, device)
    corpus = Corpus(root, device)
    params = learned(model)
    groups = [{"params": [v for k, v in params.items() if "vision_encoder." in k], "base_lr": 1e-5},
              {"params": [v for k, v in params.items() if "vision_encoder." not in k], "base_lr": 5e-5}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    contract = read(root / "contract.json")
    steps = contract["value_budget_steps"]
    details = {"steps": steps, "batch_size": size, "trainable": sum(v.numel() for v in params.values()),
               "trainable_tensors": list(params), "base_sha256": PARENT_SHA, "source_revision": SOURCE_REV,
               "normalization": corpus.norm, "selection": "lowest outcome-balanced validation episode MAE",
               "validation_anchors_per_episode": 48, "failure_sampling": .35, "importance_correction": True}
    checkpoint = root / "value/latest.pt"
    if checkpoint.exists() and read(root / "value/config.json") != details:
        raise ValueError("Cannot resume with changed value configuration")
    write(root / "value/config.json", details)
    print("Loaded native critic", details["trainable"], "trainable parameters; batch", size, flush=True)
    first = 0
    if checkpoint.exists():
        last = torch.load(checkpoint, map_location="cpu", weights_only=True)
        apply_selected(model, last["learned"])
        optimizer.load_state_dict(last["optimizer"])
        torch.cuda.set_rng_state(last["cuda_rng"])
        torch.set_rng_state(last["torch_rng"])
        first = last["step"]
        del last
        print("Resumed value at", first, flush=True)
    if not (root / "value/baseline.json").exists():
        result = evaluate(model, corpus, "val", 48, eval_batch)
        write(root / "value/baseline.json", result)
        print("Public parent validation", result["macro_mae"], result["outcome_balanced_mae"], flush=True)
    interval, start = [], time.monotonic()
    for step in range(first+1, steps+1):
        if time.time() >= contract["budget_end_epoch"]:
            save_fit_state(checkpoint, model, optimizer, step-1)
            raise TimeoutError("Global experiment deadline reached during value training")
        model.train()
        model.model.language_model.eval()
        batch = corpus.sample(np.random.default_rng(6010000+step), size)
        warmup = 500
        if step <= warmup:
            factor = step/warmup
        else:
            factor = .02 + .98*(1+math.cos(math.pi*(step-warmup)/(steps-warmup)))/2
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"]*factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(list(params.values()), 10., error_if_nonfinite=True)
        optimizer.step()
        interval.append(float(loss.detach()))
        if step in (1, 5, 20) or step % 50 == 0:
            row = {"step": step, "train_ce": float(np.mean(interval)), "grad_norm": float(grad),
                   "seconds": round(time.monotonic()-start, 1), "peak_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 3)}
            with (root / "value/training.jsonl").open("a") as f:
                print(json.dumps(row), file=f)
            print(row, flush=True)
            interval.clear()
        if step == 100 or step % 500 == 0 or step == steps:
            result = evaluate(model, corpus, "val", 48, eval_batch)
            write(root / f"value/validation-{step:05d}.json", result)
            select_weights(root, model, step, result)
            save_fit_state(checkpoint, model, optimizer, step)
            print("Value validation", step, result["macro_mae"], result["outcome_balanced_mae"], flush=True)
    apply_selected(model, load_file(str(root / "value/best.safetensors")))
    best = read(root / "value/best.json")
    metrics = {"completed_steps": steps, "selected_step": best["step"],
               "validation": evaluate(model, corpus, "val", 320, eval_batch),
               "test": evaluate(model, corpus, "test", 320, eval_batch)}
    export = root / "value/model.safetensors"
    temp = export.with_name("model.partial.safetensors")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp), metadata={"format": "pt"})
    temp.replace(export)
    metrics["full_sha256"] = digest(export)
    write(root / "value/metrics.json", metrics)
    print("Selected full critic", best["step"], metrics["full_sha256"], flush=True)


def score(root, size):
    metrics = read(root / "value/metrics.json")
    export = root / "value/model.safetensors"
    if digest(export) != metrics["full_sha256"]:
        raise ValueError("Selected critic changed")
    device = torch.device("cuda")
    model = load_native(root, device, export=export).eval()
    corpus = Corpus(root, device)
    result_dir = root / "value/predictions"
    result_dir.mkdir(exist_ok=True)
    for ep in corpus.episodes:
        path = result_dir / (ep.record["id"]+".npy")
        stamp = path.with_suffix(".json")
        if path.exists() and stamp.exists() and read(stamp).get("critic") == metrics["full_sha256"]:
            if np.load(path, mmap_mode="r").shape == ep.t.shape:
                continue
        values = np.empty(len(ep.t), dtype=np.float32)
        for i in range(0, len(ep.t), size):
            pts = [(ep, j) for j in range(i, min(i+size, len(ep.t)))]
            values[i:i+len(pts)] = predictions(model, corpus, pts)
        temp = path.with_suffix(".partial.npy")
        np.save(temp, values)
        temp.replace(path)
        write(stamp, {"critic": metrics["full_sha256"], "frames": len(values)})
        print("Scored", ep.record["id"], len(values), flush=True)
    write(result_dir / "complete.json", {"critic": metrics["full_sha256"], "episodes": len(corpus.episodes),
                                       "frames": sum(len(ep.t) for ep in corpus.episodes)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "score", "probe"])
    ap.add_argument("--root", type=Path, default=Path("/data/b601-recap-round1/evo"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=12)
    opt = ap.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(0)
    torch.set_float32_matmul_precision("high")
    if opt.stage == "train":
        train(opt.root, opt.batch, opt.eval_batch)
    elif opt.stage == "score":
        score(opt.root, opt.eval_batch)
    else:
        dev = torch.device("cuda")
        policy = load_native(opt.root, dev)
        dataset = Corpus(opt.root, dev)
        batch = dataset.sample(np.random.default_rng(77), opt.batch)
        before = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(list(learned(policy).values()), 10., error_if_nonfinite=True)
        print("Native forward/backward", float(loss), float(grad), "wall", time.monotonic()-before,
              "peak GiB", torch.cuda.max_memory_allocated()/2**30, flush=True)
