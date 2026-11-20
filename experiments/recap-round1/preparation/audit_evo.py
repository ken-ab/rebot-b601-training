"""Check the delivered advantages against source frames and compare the two fitted critics."""
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outcome_errors(rows):
    pairs = []
    for outcome in (True,False):
        group = [r["mae"] for r in rows if r["success"]==outcome]
        pairs.append(float(np.mean(group)))
    return {"episodes":len(rows),"successful":sum(r["success"] for r in rows),
            "failed":sum(not r["success"] for r in rows),"macro_mae":float(np.mean([r["mae"] for r in rows])),
            "success_mae":pairs[0],"failure_mae":pairs[1],"balanced_mae":.5*sum(pairs)}


def preference(rows):
    success = np.array([r["progress"] for r in rows if r["success"]])
    failure = np.array([r["progress"] for r in rows if not r["success"]])
    d = success[:,None,:]-failure[None,:,:]
    return {"pairs":len(success)*len(failure),"fractions":[.25,.5,.75,.9,1.0],
            "fraction_of_success_above_failure":((d>0)+.5*(d==0)).mean(axis=(0,1)).tolist()}


def audit(root):
    out = root/"evo"
    bundle = out/"value-model"
    labeled = out/"tagged-data"
    meta, summary = read(bundle/"b601.json"), read(labeled/"label_summary.json")
    if meta["model_sha256"]!=summary["checkpoint_sha256"]:
        raise ValueError("Tags and delivered model have different hashes")
    rows = read(labeled/"manifest.json")["episodes"]
    if (root/"manifest.json").read_bytes()!=(labeled/"manifest.json").read_bytes():
        raise ValueError("The server and local episode manifests differ")
    if len({r["signature"] for r in rows})!=len(rows):
        raise ValueError("Duplicate episode signatures in splits")
    penalty = meta["normalization"]["penalty_seconds"]
    observed = {s:[] for s in ("train","val","test")}
    lightweight = {s:[] for s in observed}
    total = {s:dict(frames=0,ranked=0,ranked_positive=0,manual=0,excluded=0,starts=0,changed_ranked_tags=0)
             for s in observed}
    max_diff = 0.
    for row in rows:
        name, split = row["id"],row["split"]
        with np.load(root/"episodes"/name/"rows.npz") as source:
            t,mode,epoch,frame = (source[k] for k in ("time","mode","epoch","source_frame"))
        labels = pq.read_table(labeled/"labels"/(name+".parquet"))
        old = pq.read_table(root/"labels"/(name+".parquet"),columns=["value","positive"])
        if not np.array_equal(frame,labels["source_frame"].to_numpy()) or not np.array_equal(t,labels["task_seconds"].to_numpy()):
            raise ValueError(f"Frame misalignment in {name}")
        v = labels["value"].to_numpy()
        manual,eligible = mode==2,mode==1
        positive = labels["positive"].fill_null(False).to_numpy()
        if (not np.array_equal(labels["force_positive"].to_numpy(),manual) or
                not np.array_equal(labels["positive"].is_null().to_numpy(),mode==0) or
                not np.array_equal(labels["ranking_pool"].to_numpy(),eligible) or not positive[manual].all()):
            raise ValueError(f"Invalid intervention tags in {name}")
        valid_start = np.zeros(len(t),dtype=bool)
        ends = np.r_[np.where((mode[1:]!=mode[:-1]) | (epoch[1:]!=epoch[:-1]))[0]+1,len(t)]
        expected = np.empty(len(t))
        low = 0
        for high in ends:
            query = t[low:high]+2.
            j = np.minimum(np.searchsorted(t,query),min(high,len(t)-1))
            end_value = np.where(j==len(t)-1,0. if row["success"] else -.5,v[j])
            expected[low:high] = end_value-v[low:high]-(t[j]-t[low:high])/(2*penalty)
            if mode[low]>0 and high-low>=25:
                valid_start[low:high-24]=True
            low=high
        if not np.array_equal(valid_start,labels["policy_chunk_start"].to_numpy()):
            raise ValueError(f"Action chunk crosses a control handoff in {name}")
        a = labels["advantage"].to_numpy()
        mask = mode!=0
        if mask.any():
            diff = float(np.max(np.abs(expected[mask]-a[mask])))
            max_diff=max(diff,max_diff)
            if not np.isfinite(diff) or diff>1e-6:
                raise ValueError(f"Advantage mismatch in {name}")
        ix=np.unique(np.searchsorted(t,np.linspace(t[0],t[-1],min(320,len(t)))).clip(0,len(t)-1))
        pi=np.searchsorted(t,t[0]+(t[-1]-t[0])*np.array([.25,.5,.75,.9,1.])).clip(0,len(t)-1)
        truth=(-(t[-1]-t)-(0. if row["success"] else penalty))/(2*penalty)
        for dest,pred in ((observed,v),(lightweight,old["value"].to_numpy())):
            dest[split].append({"success":row["success"],"mae":float(np.abs(pred[ix]-truth[ix]).mean()),
                                "progress":pred[pi].tolist()})
        c=total[split]
        for k,n in {"frames":len(t),"ranked":eligible.sum(),"ranked_positive":positive[eligible].sum(),
                    "manual":manual.sum(),"excluded":(~mask).sum(),"starts":valid_start.sum(),
                    "changed_ranked_tags":np.count_nonzero(positive[eligible]!=old["positive"].fill_null(False).to_numpy()[eligible])}.items():
            c[k]+=int(n)
    if total["train"]["ranked_positive"]!=round(total["train"]["ranked"]*.4):
        raise ValueError("Automatic training positives do not match 40 percent")
    report={"episodes":len(rows),"max_recomputed_advantage_difference":max_diff,"frames":total,
            "value_checkpoint_sha256":meta["model_sha256"],
            "value_comparison_same_320_temporal_anchors":{s:{"evo":outcome_errors(observed[s]),
                "lightweight":outcome_errors(lightweight[s]),"evo_at_equal_fractional_time":preference(observed[s]),
                "lightweight_at_equal_fractional_time":preference(lightweight[s])} for s in ("val","test")}}
    file=out/"frame-and-value-audit.json"
    file.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(file, sum(t["frames"] for t in total.values()), "frames checked")


if __name__=="__main__":
    audit(Path(sys.argv[1]))
