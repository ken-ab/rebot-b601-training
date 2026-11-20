"""Extract dense two-camera features on Windows using a private DirectML runtime."""
import argparse
import hashlib
import mmap
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from prepare import OUTPUT, read, write_json


BATCH = 32
DIM = 1536
WEIGHT_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def torch_encoder(weights):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    class Encoder(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.early = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
            self.late = net.layer4

        def forward(self, x):
            early = self.early(x)
            spatial = nn.functional.avg_pool2d(early, 4).flatten(1)
            semantic = self.late(early).mean((-1, -2))
            return torch.cat((spatial, semantic), dim=1)

    if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHT_SHA:
        raise ValueError("Unexpected torchvision ResNet-18 weights")
    net = resnet18()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return Encoder(net).eval()


def create_encoder(output):
    import onnxruntime as ort
    import torch
    torch.set_num_threads(2)
    assets = output / "assets"
    model = torch_encoder(assets / "resnet18-f37072fd.pth")
    probe = torch.from_numpy(np.random.default_rng(601).normal(size=(BATCH,3,128,128)).astype("f"))
    path = assets / "value-encoder-v1.onnx"
    if not path.is_file():
        tmp = path.with_suffix(".partial.onnx")
        torch.onnx.export(model, probe, str(tmp), input_names=["rgb"], output_names=["embedding"],
                          opset_version=17, dynamo=False, external_data=False)
        tmp.replace(path)
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    provider = [("DmlExecutionProvider", {"device_id":int(os.environ.get("B601_DML_DEVICE", "1"))}),
                "CPUExecutionProvider"] if "DmlExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), sess_options=options, providers=provider)
    with torch.no_grad():
        reference = model(probe).numpy()
    actual = sess.run(None, {"rgb":probe.numpy()})[0]
    if actual.shape != (BATCH,DIM) or not np.allclose(reference, actual, atol=.03, rtol=.015):
        raise ValueError(f"GPU encoder disagrees with PyTorch: max abs {np.abs(reference-actual).max()}")
    write_json(assets/"encoder.json", {"model":"torchvision/resnet18 ImageNet1K V1","weights_sha256":WEIGHT_SHA,
               "onnx_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"providers":sess.get_providers(),
               "reference_max_abs":float(np.abs(reference-actual).max()), "provider_options":sess.get_provider_options(),
               "view_order":["overhead","wrist"],
               "input":"RGB 128x96 with 16px vertical black padding; ImageNet channel normalization",
               "features_per_view":DIM})
    print("Visual encoder", sess.get_providers(), "verified", flush=True)
    return sess


def extract(output, ep, session):
    meta = read(ep / "episode.json")
    final = output / "features" / (ep.name + ".npy")
    if final.exists():
        if np.load(final, mmap_mode="r").shape != (meta["n"],2*DIM):
            raise ValueError(f"Inconsistent cached features in {ep.name}")
        return
    offsets = np.load(ep/"rows.npz")["offsets"]
    temp = final.with_suffix(".partial.npy")
    features = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16, shape=(meta["n"],2*DIM))
    # Each batch contains both views of consecutive observations.
    with (ep/"views.jpgs").open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        rgb = np.zeros((BATCH, 128, 128, 3), np.uint8)
        for start in range(0, meta["n"], BATCH//2):
            end = min(meta["n"], start+BATCH//2)
            for row in range(start, end):
                for v in (0,1):
                    a,b = offsets[row, v:v+2]
                    im = cv2.imdecode(np.frombuffer(data[a:b], np.uint8), cv2.IMREAD_COLOR)
                    rgb[2*(row-start)+v,16:112] = cv2.resize(im,(128,96),interpolation=cv2.INTER_AREA)[:,:,::-1]
            x = ((rgb.astype(np.float32)/255 - MEAN)/STD).transpose(0,3,1,2).copy()
            arr = session.run(None, {"rgb":x})[0][:2*(end-start)].reshape(end-start,2*DIM)
            if not np.isfinite(arr).all():
                raise ValueError(f"Nonfinite image embeddings in {ep.name}")
            features[start:end] = arr
    features.flush()
    del features
    temp.replace(final)
    print("Encoded", ep.name, meta["n"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=OUTPUT)
    ap.add_argument("--follow",action="store_true")
    args = ap.parse_args()
    sys.path.insert(0,str(args.output/"runtime"))
    cv2.setNumThreads(1)
    (args.output/"features").mkdir(exist_ok=True)
    session = create_encoder(args.output)
    while True:
        freeze = args.output/"manifest.json"
        wanted = {r["id"] for r in read(freeze)["episodes"]} if freeze.exists() else None
        for ep in sorted((args.output/"episodes").iterdir()):
            if (ep/"episode.json").exists() and (wanted is None or ep.name in wanted):
                extract(args.output, ep, session)
        if not args.follow or (wanted is not None and all((args.output/"features"/(k+".npy")).exists() for k in wanted)):
            break
        time.sleep(4)


if __name__ == "__main__":
    main()
