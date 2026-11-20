"""Export and read the selected B601 Evo-RL critic without its upstream training data."""
import argparse
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.values.pistar06.processor_pistar06 import Pistar06PrepareTaskPromptProcessorStep
from evo_value import (Episode, SOURCE_REV, TASK, digest, materialize_backbones, native_from_file,
                       observation_batch, read, write)


class B601Value:
    def __init__(self, folder, device="cpu", cache=None):
        folder = Path(folder).resolve()
        self.device = torch.device(device)
        self.meta = read(folder/"b601.json")
        full = folder/"model.safetensors"
        assets = folder/"model-assets"
        sha = self.meta["model_sha256"]
        if digest(full) != sha:
            raise ValueError("The portable B601 value model failed its checksum")
        cache = Path(cache or folder/"runtime-cache") / sha[:20]
        backbones = materialize_backbones(assets, cache, full, sha)
        self.model = native_from_file(folder, backbones, full, self.device).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(str(assets/"language"), local_files_only=True)
        self.prompt = Pistar06PrepareTaskPromptProcessorStep()
        self.low = np.asarray(self.meta["normalization"]["q01"], dtype=np.float32)
        self.high = np.asarray(self.meta["normalization"]["q99"], dtype=np.float32)

    @torch.inference_mode()
    def predict(self, overhead_rgb, wrist_rgb, motor_degrees):
        pair = []
        for image in (overhead_rgb, wrist_rgb):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("The two cameras must be RGB uint8 [height,width,3]")
            if image.shape[:2] != (168,224):
                image = cv2.resize(image,(224,168),interpolation=cv2.INTER_AREA)
            pair.append(image)
        state = np.asarray(motor_degrees,dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Expected seven finite motor-side degree observations in capture order")
        pictures = np.stack(pair)[None].transpose(0,1,4,2,3)
        batch = observation_batch(state[None],pictures,self.low,self.high,self.prompt,self.tokenizer,self.device)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
            v = float(self.model.predict_value(batch).float().item())
        if not np.isfinite(v) or not -1.0001 <= v <= .0001:
            raise ValueError("Critic produced an invalid normalized return")
        return v


def export(root, folder):
    folder.mkdir(parents=True, exist_ok=True)
    metrics = read(root/"value/metrics.json")
    selected = root/"value/model.safetensors"
    if digest(selected) != metrics["full_sha256"]:
        raise ValueError("Selected value weights changed")
    target = folder/"model.safetensors"
    if not target.exists():
        try:
            os.link(selected,target)
        except OSError:
            temp = target.with_suffix(".partial")
            shutil.copy2(selected,temp)
            temp.replace(target)
    if digest(target) != metrics["full_sha256"]:
        raise ValueError("Existing delivery folder contains a different critic")
    cfg = read(root/"inputs/stacking/config.json")
    cfg["input_features"]["observation.state"]["shape"] = [7]
    cfg["camera_features"] = ["observation.images.top","observation.images.wrist"]
    cfg["mode"] = "vision_language"
    cfg["freeze_language_model"] = True
    cfg["repo_id"] = None
    cfg["push_to_hub"] = False
    write(folder/"config.json",cfg)
    shutil.copytree(root/"inputs/model-assets",folder/"model-assets",dirs_exist_ok=True)
    for filename in ("metrics.json","best.json","training.jsonl","baseline.json"):
        shutil.copy2(root/"value"/filename,folder/filename)
    meta = {"format": "B601 Evo-RL value round 1", "upstream_revision": SOURCE_REV,
            "model_sha256": metrics["full_sha256"], "completed_steps": metrics["completed_steps"],
            "selected_step": metrics["selected_step"], "task": TASK,
            "normalization": read(root/"value/config.json")["normalization"],
            "inputs": "overhead RGB, wrist RGB, seven motor-side degrees",
            "output": "expected normalized return, in [-1,0]; higher is better"}
    write(folder/"b601.json", meta)
    # References were independently produced during evaluation of the selected checkpoint.
    records = {r["id"]:r for r in read(root/"manifest.json")["episodes"]}
    test = metrics["test"]["episodes"]
    examples = [next(r for r in test if r["success"]), next(r for r in test if not r["success"]),
                metrics["validation"]["episodes"][0]]
    frames, states, targets, names = [], [], [], []
    for row in examples:
        ep = Episode(root,records[row["episode"]])
        for i, key in ((0,"first"),(len(ep.t)-1,"last")):
            frames.append(ep.images(i).transpose(0,2,3,1))
            states.append(ep.state[i]); targets.append(row[key]); names.append(f"{row['episode']}:{i}")
    np.savez_compressed(folder/"reference.npz", rgb=np.stack(frames), state=np.stack(states),
                        expected=np.array(targets), names=np.array(names))


def verify(folder, device, cache, output):
    model = B601Value(folder,device,cache)
    with np.load(folder/"reference.npz") as ref:
        rows = []
        for images, state, expected, name in zip(ref["rgb"],ref["state"],ref["expected"],ref["names"],strict=True):
            actual = model.predict(*images,state)
            rows.append({"frame":str(name),"expected":float(expected),"actual":actual,
                         "absolute_difference":abs(actual-float(expected))})
    report = {"device":device, "model_sha256":model.meta["model_sha256"], "samples":rows,
              "max_absolute_difference":max(r["absolute_difference"] for r in rows)}
    if output:
        write(output, report)
    print(json.dumps(report), flush=True)
    if report["max_absolute_difference"] > .01:
        raise ValueError("Portable readback differs from the training evaluation")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export","verify","predict"])
    ap.add_argument("--root",type=Path,default=Path("/data/b601-recap-round1/evo"))
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    ap.add_argument("--cache",type=Path)
    ap.add_argument("--report",type=Path)
    ap.add_argument("--overhead",type=Path)
    ap.add_argument("--wrist",type=Path)
    ap.add_argument("--state",nargs=7,type=float)
    opt=ap.parse_args()
    torch.set_num_threads(3)
    if opt.stage=="export":
        export(opt.root,opt.bundle)
    elif opt.stage=="verify":
        verify(opt.bundle,opt.device,opt.cache,opt.report)
    else:
        if not all([opt.overhead,opt.wrist,opt.state]):
            ap.error("predict requires both image files and seven state values")
        def load_rgb(path):
            image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable RGB image {path}")
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        print(B601Value(opt.bundle,opt.device,opt.cache).predict(load_rgb(opt.overhead),load_rgb(opt.wrist),opt.state))
