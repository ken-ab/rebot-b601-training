"""Fit the B601 distributional critic and create per-frame advantage sidecars."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from prepare import OUTPUT, read, write_json


BINS = 201
LOOKAHEAD_SECONDS = 50 / 25


class ValueHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.18))
        self.joints = nn.Sequential(nn.Linear(7, 48), nn.LayerNorm(48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(240, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.12),
                                  nn.Linear(128, BINS))
        self.register_buffer("bins", torch.linspace(-1., 0., BINS))
        self.register_buffer("visual_mean", torch.zeros(dim))
        self.register_buffer("visual_std", torch.ones(dim))
        self.register_buffer("state_mean", torch.zeros(7))
        self.register_buffer("state_std", torch.ones(7))

    def forward(self, features, joints):
        visual = self.visual((features.float() - self.visual_mean) / self.visual_std)
        low = self.joints((joints - self.state_mean) / self.state_std)
        return self.head(torch.cat((visual, low), dim=1))

    def expected(self, logits):
        return (logits.softmax(-1) * self.bins).sum(-1)


def returns(t, success, penalty):
    return (-(t[-1]-t) - (0. if success else penalty)) / (2*penalty)


def distribution_loss(logits, target):
    pos = ((target.clamp(-1., 0.) + 1) * (BINS-1)).clamp(0., BINS-1)
    lo = pos.long()
    hi = (lo + 1).clamp(max=BINS-1)
    mix = pos-lo
    ll = logits.log_softmax(-1)
    return -((1-mix)*ll.gather(1,lo[:,None]).squeeze(1) + mix*ll.gather(1,hi[:,None]).squeeze(1))


def sample_split(root, records, split, penalty):
    visual, joints, target, groups, success = [], [], [], [], []
    for i, r in enumerate(records):
        if r["split"] != split:
            continue
        with np.load(root/"episodes"/r["id"]/"rows.npz") as z:
            t = z["time"]
            # Equal temporal coverage per episode, regardless of source camera FPS.
            grid = np.linspace(t[0], t[-1], min(320, len(t)))
            ix = np.unique(np.searchsorted(t, grid).clip(0,len(t)-1))
            joints.append(z["state"][ix])
            target.append(returns(t,r["success"],penalty)[ix].astype("f"))
        feat = np.load(root/"features"/(r["id"]+".npy"), mmap_mode="r")
        visual.append(np.array(feat[ix]))
        groups.extend([i]*len(ix))
        success.extend([r["success"]]*len(ix))
    return (torch.from_numpy(np.concatenate(visual)),torch.from_numpy(np.concatenate(joints)),
            torch.from_numpy(np.concatenate(target)),np.array(groups),np.array(success))


@torch.inference_mode()
def assess(model, data, baseline):
    model.eval()
    feature,state,target,groups,success = data
    preds, losses = [], []
    for f,s,y in zip(feature.split(512),state.split(512),target.split(512),strict=True):
        logits = model(f,s)
        preds.append(model.expected(logits))
        losses.append(distribution_loss(logits,y))
    pred = torch.cat(preds).numpy()
    truth = target.numpy()
    def macro(a, mask):
        return float(np.mean([a[(groups==g)&mask].mean() for g in np.unique(groups[mask])]))
    all_rows = np.ones(len(groups), bool)
    err = np.abs(pred-truth)
    return {"episodes":len(np.unique(groups)), "observations":len(truth),
            "macro_mae":macro(err,all_rows), "constant_baseline_macro_mae":macro(np.abs(truth-baseline),all_rows),
            "successful_episode_mae":macro(err,success), "failed_episode_mae":macro(err,~success),
            "distribution_ce":float(torch.cat(losses).mean()),"targets_outside_bins":int(((truth < -1)|(truth>0)).sum())}


def fit(root):
    torch.set_num_threads(5)
    torch.manual_seed(601)
    manifest = root/"manifest.json"
    records = read(manifest)["episodes"]
    train = [r for r in records if r["split"]=="train"]
    penalty = max(120., max(r["seconds"] for r in train))
    data = sample_split(root,records,"train",penalty)
    val = sample_split(root,records,"val",penalty)
    f,s,y = data[:3]
    model = ValueHead(f.shape[1])
    stat = f[::3].float()
    model.visual_mean.copy_(stat.mean(0))
    model.visual_std.copy_(stat.std(0).clamp(min=.05))
    model.state_mean.copy_(s.mean(0))
    model.state_std.copy_(s.std(0).clamp(min=.1))
    del stat
    baseline = float(y.median())
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.02)
    gen = torch.Generator().manual_seed(601)
    run = root/"value"
    run.mkdir(exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, 31):
        model.train()
        loss_sum = 0.
        for idx in torch.randperm(len(y), generator=gen).split(192):
            logits = model(f[idx],s[idx])
            loss = distribution_loss(logits,y[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())*len(idx)
        result = assess(model,val,baseline)
        history.append({"epoch":epoch,"train_ce":loss_sum/len(y),"validation":result})
        write_json(run/"history.json",history)
        print(f"Value epoch {epoch} train CE={loss_sum/len(y):.4f} validation MAE={result['macro_mae']:.5f}",flush=True)
        if result["macro_mae"] < best:
            best = result["macro_mae"]
            best_epoch = epoch
            checkpoint = {"state_dict":model.state_dict(),"visual_dim":f.shape[1],"fail_penalty_seconds":penalty,
                          "return_scale_seconds":2*penalty,"horizon_seconds":LOOKAHEAD_SECONDS,
                          "epoch":epoch,"train_median":baseline,"value_bins":BINS,"encoder":read(root/"assets/encoder.json"),
                          "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}
            tmp = run/"best.partial.pt"
            torch.save(checkpoint,tmp)
            tmp.replace(run/"best.pt")
        if epoch-best_epoch >= 6:
            break
    c = torch.load(run/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(c["state_dict"])
    test = sample_split(root,records,"test",penalty)
    report = {"architecture":"frozen ImageNet ResNet18, dual view + seven measured joints, 201-bin critic; single fixed task",
              "learned_parameters":sum(p.numel() for p in model.parameters()), "best_epoch":best_epoch,
              "epochs_run":epoch,"penalty_seconds":penalty,"train_observations":len(y),
              "validation":assess(model,val,baseline),"test":assess(model,test,baseline),
              "manifest_sha256":c["manifest_sha256"], "encoder_sha256":c["encoder"]["onnx_sha256"]}
    write_json(run/"metrics.json",report)
    print("Value held-out results:", json.dumps(report), flush=True)


def segments_right(mode,epoch):
    # First observation outside the segment has not yet executed its own action.
    n = len(mode)
    ends = np.r_[np.flatnonzero((mode[1:]!=mode[:-1]) | (epoch[1:]!=epoch[:-1]))+1,n]
    return np.repeat(ends,np.diff(np.r_[0,ends]))


def advantages(t, values, mode, epoch, success, penalty):
    n = len(t)
    boundary = segments_right(mode, epoch)
    j = np.minimum(np.searchsorted(t,t+LOOKAHEAD_SECONDS),np.minimum(boundary,n-1))
    boot = values[j].copy()
    boot[j==n-1] = 0. if success else -.5
    advantage = -(t[j]-t)/(2*penalty) + boot-values
    # Every action in the training chunk must be observed and from one control epoch.
    can_start = (mode>0) & (np.arange(n)+25 <= boundary)
    return advantage, can_start


def tie_ids(key,n):
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
    return np.random.default_rng(seed).integers(0,np.iinfo("uint64").max,size=n,dtype=np.uint64)


@torch.inference_mode()
def label(root):
    torch.set_num_threads(5)
    records = read(root/"manifest.json")["episodes"]
    ckpt = torch.load(root/"value/best.pt",map_location="cpu",weights_only=True)
    if ckpt["manifest_sha256"] != hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest():
        raise ValueError("Value checkpoint and episode manifest differ")
    model = ValueHead(ckpt["visual_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    predictions = {}
    for r in records:
        rid = r["id"]
        feature = np.load(root/"features"/(rid+".npy"), mmap_mode="r")
        with np.load(root/"episodes"/rid/"rows.npz") as z:
            state = z["state"]
        predictions[rid] = np.concatenate([
            model.expected(model(torch.tensor(feature[i:i+384]), torch.from_numpy(state[i:i+384]))).numpy()
            for i in range(0, len(state), 384)])
    checkpoint_sha = hashlib.sha256((root/"value/best.pt").read_bytes()).hexdigest()
    write_labels(root, records, predictions, ckpt["fail_penalty_seconds"], checkpoint_sha)


def write_labels(root, records, predictions, penalty, checkpoint_sha):
    stored = {}
    rank_a,rank_k = [],[]
    for r in records:
        rid = r["id"]
        with np.load(root/"episodes"/rid/"rows.npz") as z:
            t, mode, epoch, src = (z[k] for k in ("time","mode","epoch","source_frame"))
        v = predictions[rid]
        if v.shape != t.shape or not np.isfinite(v).all():
            raise ValueError(f"Missing critic observations in {rid}")
        a, valid_start = advantages(t,v,mode,epoch,r["success"],penalty)
        key = tie_ids(rid,len(t))
        stored[rid] = t,src,v,a,mode,valid_start,key
        if r["split"] == "train" and (mode==1).any():
            rank_a.append(a[mode==1])
            rank_k.append(key[mode==1])
    values, keys = np.concatenate(rank_a),np.concatenate(rank_k)
    k = round(.4*len(values))
    ix = np.lexsort((keys,values))[-k]
    cutoff, tie = values[ix], keys[ix]
    folder = root/"labels"
    folder.mkdir(exist_ok=True)
    totals = {s:Counter() for s in ("train","val","test")}
    for r in records:
        rid = r["id"]
        t,src,v,a,mode,can,key = stored[rid]
        ranked = (a>cutoff) | ((a==cutoff) & (key>=tie))
        positive = (mode==2) | ((mode==1) & ranked)
        dest = folder/(rid+".parquet")
        table = pa.table({"source_frame":src,"task_seconds":t,"value":v,
                          "advantage":pa.array(a,mask=mode==0),
                          "positive":pa.array(positive,mask=mode==0),"force_positive":mode==2,
                          "ranking_pool":mode==1,"policy_chunk_start":can})
        tmp = dest.with_suffix(".part")
        pq.write_table(table,tmp,compression="zstd")
        tmp.replace(dest)
        group = totals[r["split"]]
        group.update({"episodes":1,"observations":len(v),"pool_frames":int((mode==1).sum()),
                      "ranked_positive":int(((mode==1)&positive).sum()),"forced_positive":int((mode==2).sum()),
                      "excluded_from_policy":int((mode==0).sum()),"policy_chunk_starts":int(can.sum())})
    if totals["train"]["ranked_positive"] != k:
        raise ValueError("Ranking cut does not match requested 40 percent")
    report = {"lookahead_steps_at_25hz":50,"threshold":float(cutoff),"tie_key":int(tie),
              "ranking_scope":"eligible demonstration and autonomous training frames; correction frames excluded",
              "validation_and_test":"same frozen training cutoff","totals":totals,
              "checkpoint_sha256":checkpoint_sha}
    write_json(root/"label_summary.json", report)
    print("Labeled:",json.dumps(report),flush=True)


def label_evo(root):
    pred_dir = root/"value/predictions"
    sha = read(root/"value/metrics.json")["full_sha256"]
    records = read(root/"manifest.json")["episodes"]
    complete = read(pred_dir/"complete.json")
    if complete["critic"] != sha or complete["episodes"] != len(records):
        raise ValueError("Incomplete Evo-RL scoring pass")
    predictions = {}
    for r in records:
        name = r["id"]
        if read(pred_dir/(name+".json"))["critic"] != sha:
            raise ValueError(f"Stale critic predictions in {name}")
        predictions[name] = np.load(pred_dir/(name+".npy"), mmap_mode="r")
    write_labels(root,records,predictions,read(root/"value/config.json")["normalization"]["penalty_seconds"],sha)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("train","label","both","evo-label"))
    ap.add_argument("--output",type=Path,default=OUTPUT)
    options=ap.parse_args()
    if options.stage in ("train","both"):
        fit(options.output)
    if options.stage in ("label","both"):
        label(options.output)
    if options.stage == "evo-label":
        label_evo(options.output)
