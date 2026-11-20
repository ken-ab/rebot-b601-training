"""Export the complete trained value model as one portable, CPU-loadable TorchScript file."""
import argparse
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from prepare import OUTPUT, read, write_json
from value import ValueHead
from visual_features import MEAN, STD, torch_encoder


class CompleteCritic(nn.Module):
    def __init__(self,encoder,head):
        super().__init__()
        self.encoder,self.head=encoder,head
        self.register_buffer("mean",torch.from_numpy(MEAN).view(1,3,1,1))
        self.register_buffer("std",torch.from_numpy(STD).view(1,3,1,1))

    def forward(self,views,measured_degrees):
        # Input is uint8 RGB [batch, 2, 96, 128, 3] in overhead, wrist order.
        batch=views.size(0)
        rgb=views.reshape(-1,96,128,3).permute(0,3,1,2).float()/255
        obs=(F.pad(rgb,(0,0,16,16))-self.mean)/self.std
        features=self.encoder(obs).half().float().reshape(batch,-1)
        return self.head.expected(self.head(features,measured_degrees))


def export(root):
    torch.set_num_threads(4)
    bundle=torch.load(root/"value/best.pt",weights_only=True,map_location="cpu")
    value=ValueHead(bundle["visual_dim"])
    value.load_state_dict(bundle["state_dict"])
    net=CompleteCritic(torch_encoder(root/"assets/resnet18-f37072fd.pth"),value).eval()
    views=[]; states=[]; targets=[]
    for r in read(root/"manifest.json")["episodes"]:
        if r["split"]!="test":
            continue
        ep=root/"episodes"/r["id"]
        i=r["n"]//2
        with np.load(ep/"rows.npz") as z,(ep/"views.jpgs").open("rb") as f:
            a,b,c=z["offsets"][i]
            f.seek(a)
            pair=[]
            for length in (b-a,c-b):
                im=cv2.imdecode(np.frombuffer(f.read(length),np.uint8),cv2.IMREAD_COLOR)
                pair.append(cv2.resize(im,(128,96),interpolation=cv2.INTER_AREA)[:,:,::-1].copy())
            views.append(pair);states.append(z["state"][i])
        targets.append(pq.read_table(root/"labels"/(r["id"]+".parquet"),columns=["value"])["value"][i].as_py())
    inp=torch.from_numpy(np.asarray(views)); state=torch.from_numpy(np.array(states))
    with torch.inference_mode():
        traced=torch.jit.freeze(torch.jit.trace(net,(inp[:1],state[:1])).eval())
        target=root/"value/b601_value.ts"
        temp=target.with_suffix(".part")
        with temp.open("wb") as stream:
            torch.jit.save(traced,stream)
        with temp.open("rb") as stream:
            loaded=torch.jit.load(stream,map_location="cpu")
        predicted=loaded(inp,state).numpy()
        err=float(np.max(np.abs(predicted-np.array(targets))))
        if predicted.shape!=(len(states),) or err>2e-4:
            raise ValueError(f"Portable critic readback differs by {err}")
        temp.replace(target)
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(root/"value/portable.json",{"file":target.name,"sha256":sha,"test_episodes":len(targets),
               "max_absolute_difference":err,"inputs":["uint8 RGB [B,2,96,128,3], overhead then wrist",
               "float32 [B,7] measured motor degrees"],"output":"expected normalized return [B]"})
    print("Exported and reopened",target,sha,"max abs difference",err,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=OUTPUT)
    export(ap.parse_args().output)
