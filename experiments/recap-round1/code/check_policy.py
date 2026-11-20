"""Verify a saved B601 policy offline on held-out observations, without robot I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from policy import Episode, PARENT, TASK, save_json


def load_positive_policy(checkpoint, bundle=PARENT):
    """Return a callable for ordinary B601 observation dictionaries with positive conditioning."""
    sys.path.insert(0,str(bundle/"code"))
    os.environ.setdefault("OPENPI_DATA_HOME",str(bundle/"cache/openpi"))
    from b601_policy import load_policy
    policy = load_policy(checkpoint)
    def predict(observation, **kwargs):
        obs=dict(observation)
        base=obs.get("prompt") or TASK
        obs["prompt"]=base if base.endswith(" Advantage: positive") else base+" Advantage: positive"
        return policy.infer(obs,**kwargs)
    return predict


def main(root, parent):
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    eligible=[Episode(root,r) for r in manifest["episodes"] if r["split"]=="test" and r["kind"]=="new"]
    if parent:
        from b601_policy import load_policy
        predict=load_policy(PARENT/"checkpoint").infer
    else:
        predict=load_positive_policy(root/"policy/checkpoint")
    rows=[]
    start=time.monotonic()
    for ep in eligible:
        valid=ep.starts[ep.positive[ep.starts]]
        if not len(valid):
            continue
        for j in np.unique(np.linspace(0,len(valid)-1,3,dtype=int)):
            i=int(valid[j])
            raw=ep.sample(i,None)
            truth=raw.pop("action")
            seed=int.from_bytes(hashlib.sha256(f"{ep.meta['id']}:{i}".encode()).digest()[:4],"little")
            noise=np.random.default_rng(seed).standard_normal((25,32)).astype(np.float32)
            actions=np.asarray(predict(raw,noise=noise)["actions"])
            if actions.shape!=(25,7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid readback at {ep.meta['id']} {i}")
            err=np.abs(actions-truth).mean(axis=0)
            rows.append({"episode":ep.meta["id"],"frame":i,"mae":err.tolist()})
    grouped=[]
    for rid in sorted({r["episode"] for r in rows}):
        grouped.append(np.mean([r["mae"] for r in rows if r["episode"]==rid],axis=0))
    avg=np.mean(grouped,axis=0)
    title="parent" if parent else "round1"
    metrics={"model":title,"episodes":len(grouped),"chunks":len(rows),"chunk_shape":[25,7],"finite":True,
             "macro_episode_per_axis_mae":avg.tolist(),"mean_axes_1_6_mae":float(avg[:6].mean()),
             "gripper_mae":float(avg[6]),"units":"absolute adapter-input degrees","elapsed_seconds":time.monotonic()-start,
             "scope":"positive-labeled held-out action chunks; fixed noise per anchor; offline","observations":rows}
    save_json(root/"policy"/(title+"-action-readback.json"),metrics)
    print(title,len(grouped),len(rows),avg,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("/data/b601-recap-round1"))
    ap.add_argument("--parent",action="store_true")
    opts=ap.parse_args()
    os.environ.setdefault("OPENPI_DATA_HOME",str(PARENT/"cache/openpi"))
    main(opts.root,opts.parent)
