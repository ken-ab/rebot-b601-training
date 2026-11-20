"""Offline advantage-conditioned OpenPI LoRA training for the deployed B601 checkpoint."""
import argparse
import dataclasses
import functools
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as pq


TASK = "拿起蓝色餐巾纸，把它放进黑色框里"
PARENT = Path("/root/b601-deploy/B601_PI05_DEPLOYMENT")
NORMALIZATION_SHA = "73adb6290f10854524740cc9229a15a29431a0ec68e7da39f2f41c76ba3b7493"
PINNED_TRAIN_SHA = "22f98a06aaa6e68e0f725d727ab95398238d1cae8a46bb94e829f330758a9baf"
sys.path.insert(0,str(PARENT/"code"))


def save_json(path, value):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class Episode:
    def __init__(self, root, meta):
        self.meta = meta
        self.path = root/"episodes"/meta["id"]
        with np.load(self.path/"rows.npz") as rows:
            self.state, self.action, self.offsets = (rows[k] for k in ("state","action","offsets"))
        labels = pq.read_table(root/"labels"/(meta["id"]+".parquet"))
        self.starts = np.flatnonzero(labels["policy_chunk_start"].to_numpy())
        self.positive = labels["positive"].fill_null(False).to_numpy()
        self.file = None

    def sample(self, i, condition):
        if self.file is None:
            self.file = (self.path/"views.jpgs").open("rb")
        a,b,c = self.offsets[i]
        self.file.seek(a)
        data = self.file.read(c-a)
        images = []
        for piece in (data[:b-a],data[b-a:]):
            image = cv2.imdecode(np.frombuffer(piece,np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable view at {self.meta['id']}#{i}")
            images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        prompt = TASK if condition is None else TASK + (" Advantage: positive" if condition else " Advantage: negative")
        return {"observation.state":self.state[i],"action":self.action[i:i+25],
                "observation.images.overhead":images[0],"observation.images.wrist":images[1],"prompt":prompt}


def get_config(args):
    from flax import nnx
    from openpi.shared import nnx_utils
    from openpi.training import config as tc, optimizer
    from train_pi05 import build_config
    parent = args.bundle/"checkpoint"
    cfg = build_config(SimpleNamespace(mode="train",steps=args.steps,batch_size=1,workers=0,
                       lr_decay_steps=args.steps,base_params=parent/"params",exp_name="b601-recap-round1",
                       fsdp_devices=1,resume=False,save_interval=500),parent)
    # The upstream default also trains some vision weights. This single-card run updates only existing LoRA leaves.
    freeze = nnx.Not(nnx_utils.PathRegex(".*lora.*"))
    schedule_steps = 30_000 if getattr(args, "until_epoch", None) else args.steps
    rate = optimizer.CosineDecaySchedule(warmup_steps=min(150,max(1,schedule_steps//10)),
            peak_lr=2.0e-5,decay_steps=max(2,schedule_steps),decay_lr=2.e-6)
    return dataclasses.replace(cfg,freeze_filter=freeze,lr_schedule=rate,
            data=dataclasses.replace(cfg.data,assets=tc.AssetsConfig(assets_dir=str(parent/"assets"),asset_id="rebot-b601-train")))


def transform_for(cfg):
    from openpi import transforms
    ds = cfg.data.create(cfg.assets_dirs,cfg.model)
    if ds.norm_stats is None or set(ds.norm_stats)!={"state","actions"}:
        raise ValueError("Missing parent normalization")
    return transforms.compose([*ds.repack_transforms.inputs,*ds.data_transforms.inputs,
                               transforms.Normalize(ds.norm_stats,use_quantiles=ds.use_quantile_norm),
                               *ds.model_transforms.inputs])


def pack(sample,transform):
    import jax
    from openpi.models import model as om
    s = transform(sample)
    if s["actions"].shape != (25,32) or s["state"].shape != (32,) or not np.isfinite(s["actions"]).all():
        raise ValueError("OpenPI transform changed B601 action or state format")
    s["state"]=s["state"].astype(np.float32)
    s["actions"]=s["actions"].astype(np.float32)
    batch = jax.tree.map(lambda x:np.expand_dims(np.asarray(x),0),s)
    return om.Observation.from_dict(batch), batch["actions"]


def sample_points(episodes, split):
    return [(ep,int(ep.starts[j])) for ep in episodes if ep.meta["split"]==split
            for j in np.unique(np.linspace(0,len(ep.starts)-1,4,dtype=int))]


def gripper_validation_points(episodes):
    points = []
    for ep in episodes:
        if ep.meta["split"] != "val":
            continue
        valid = ep.starts[ep.positive[ep.starts]]
        if not len(valid):
            continue
        delta = ep.action[valid+24,6] - ep.action[valid,6]
        # Use the same mid-trajectory, opening and closing examples at every checkpoint.
        indices = [valid[len(valid)//2], valid[np.argmax(delta)], valid[np.argmin(delta)]]
        points.extend((ep, int(i)) for i in np.unique(indices))
    return points


def adapter_dict(params,cfg):
    from flax import traverse_util
    return traverse_util.flatten_dict(params.filter(cfg.trainable_filter).to_pure_dict(),sep="/")


def install_adapter(state,cfg,path):
    from flax import nnx,traverse_util
    import jax
    adapters = state.params.filter(cfg.trainable_filter)
    original = adapter_dict(state.params,cfg)
    with np.load(path) as file:
        if set(file.files) != set(original):
            raise ValueError("LoRA tree differs from parent")
        payload = {}
        for name,old in original.items():
            val = file[name]
            if val.shape != old.shape or not np.isfinite(val).all():
                raise ValueError(f"Invalid LoRA tensor {name}")
            payload[name] = jax.device_put(val.astype(old.dtype),old.sharding)
    adapters.replace_by_pure_dict(traverse_util.unflatten_dict(payload,sep="/"))
    model = nnx.merge(state.model_def,state.params)
    nnx.update(model,adapters)
    return dataclasses.replace(state,params=nnx.state(model))


def snapshot(state,cfg,step,run,metrics):
    import jax
    target = run/"snapshots"/f"step-{step:05d}"
    target.parent.mkdir(exist_ok=True)
    temp = target.with_name(target.name+".partial")
    temp.mkdir(exist_ok=True)
    arrays = {k:np.asarray(v,dtype=np.float32) for k,v in adapter_dict(state.params,cfg).items()}
    if not arrays or any("lora" not in k for k in arrays):
        raise ValueError("Unexpected trainable tensors")
    np.savez_compressed(temp/"adapter.npz",**arrays)
    np.savez(temp/"optimizer.npz",*[np.asarray(a) for a in jax.tree.leaves(state.opt_state)])
    save_json(temp/"metrics.json",metrics)
    temp.rename(target)
    return target


def restore_optimizer(state,path,step):
    import jax
    leaves, tree = jax.tree.flatten(state.opt_state)
    with np.load(path) as f:
        if len(leaves)!=len(f.files):
            raise ValueError("Optimizer structure changed")
        loaded = [jax.device_put(f[f"arr_{i}"].astype(x.dtype),x.sharding) for i,x in enumerate(leaves)]
    return dataclasses.replace(state,opt_state=jax.tree.unflatten(tree,loaded),
                               step=jax.device_put(np.asarray(step,dtype=state.step.dtype), state.step.sharding))


def main(args):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import nnx
    from openpi.training import sharding
    import openpi_pinned_train as trainer
    if hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest()!=PINNED_TRAIN_SHA:
        raise ValueError("Unexpected OpenPI training implementation")
    stats_path = args.bundle/"checkpoint/assets/rebot-b601-train/norm_stats.json"
    if hashlib.sha256(stats_path.read_bytes()).hexdigest()!=NORMALIZATION_SHA:
        raise ValueError("Unexpected parent normalization")
    if not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("No GPU training device")
    jax.config.update("jax_compilation_cache_dir",str(args.root/"jax-cache"))
    cv2.setNumThreads(1)
    cfg = get_config(args)
    transform = transform_for(cfg)
    from openpi import transforms
    dataset_cfg = cfg.data.create(cfg.assets_dirs,cfg.model)
    to_robot = transforms.compose([*dataset_cfg.model_transforms.outputs,
                    transforms.Unnormalize(dataset_cfg.norm_stats,use_quantiles=dataset_cfg.use_quantile_norm),
                    *dataset_cfg.data_transforms.outputs,*dataset_cfg.repack_transforms.outputs])
    manifest = json.loads((args.root/"manifest.json").read_text(encoding="utf-8"))
    if sum(m["kind"]=="new" for m in manifest["episodes"])!=100:
        raise ValueError("The 100-episode cohort is incomplete")
    episodes = [Episode(args.root,m) for m in manifest["episodes"] if m["kind"]!="old"]
    episodes = [ep for ep in episodes if len(ep.starts)]
    train = [ep for ep in episodes if ep.meta["split"]=="train"]
    validation = sample_points(episodes,"val")
    action_points = gripper_validation_points(episodes)
    held_out = sample_points(episodes,"test")
    example = train[0]
    a = transform(example.sample(example.starts[0],True))
    b = transform(example.sample(example.starts[0],False))
    if np.array_equal(a["tokenized_prompt"],b["tokenized_prompt"]):
        raise ValueError("The positive/negative indicator was lost in tokenization")
    if args.inputs_only:
        print("Input validation OK",len(train),len(validation),len(held_out),flush=True)
        return
    run = args.root/"policy"
    run.mkdir(exist_ok=True)
    mesh = sharding.make_mesh(1)
    seeds = jax.random.split(jax.random.key(601))
    print(f"Loading parent, {jax.devices()}, config steps={args.steps}",flush=True)
    state, ss = trainer.init_train_state(cfg,seeds[0],mesh,resume=False)
    arrays = adapter_dict(state.params,cfg)
    if not arrays or not all("lora" in k for k in arrays):
        raise ValueError("Only LoRA weights may be updated")
    count = sum(v.size for v in arrays.values())
    initial = {k:np.asarray(v,dtype=np.float32).copy() for k,v in arrays.items()}
    fingerprint = hashlib.sha256((args.root/"manifest.json").read_bytes()+(args.root/"label_summary.json").read_bytes()).hexdigest()
    meta = dict(minimum_or_fixed_optimizer_steps=args.steps,until_epoch=args.until_epoch,
                learning_rate_decay_steps=30_000 if args.until_epoch else args.steps,
                selection="average of validation joint and gripper MAE ratios to the deployed parent; flow loss breaks ties",
                validation_action_anchors=len(action_points),batch_size=1,seed=601,parent_checkpoint="b601-final60 step 43000",
                trainable_parameters=count,leaves=len(arrays),manifest_and_labels_sha256=fingerprint,
                norm_stats_sha256=NORMALIZATION_SHA,train_episodes=len(train),validation_anchors=len(validation),
                test_anchors=len(held_out),conditioning_dropout=.3,task=TASK,positive_prompt=TASK+" Advantage: positive",
                normalization="parent", action_horizon=25,optimizer_peak_lr=2e-5)
    del arrays
    if (run/"config.json").exists() and json.loads((run/"config.json").read_text()) != meta:
        raise ValueError("Run parameters or inputs changed")
    save_json(run/"config.json",meta)
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    train_step=jax.jit(functools.partial(trainer.train_step,cfg),in_shardings=(replicated,ss,data_shard),
                      out_shardings=(ss,replicated),donate_argnums=(1,))
    @jax.jit
    def eval_step(params,key,batch):
        net = nnx.merge(state.model_def,params)
        net.eval()
        return jnp.mean(net.compute_loss(key,batch[0],batch[1],train=False))

    @jax.jit
    def predict_actions(params,key,observation,noise):
        model = nnx.merge(state.model_def,params)
        model.eval()
        return model.sample_actions(key,observation,noise=noise,num_steps=10)

    def action_metrics(condition):
        by_episode = {}
        for ep,i in action_points:
            raw = ep.sample(i,condition)
            obs,_ = pack(raw,transform)
            salt=int.from_bytes(hashlib.sha256(f"val:{ep.meta['id']}:{i}".encode()).digest()[:4],"little")
            noise=np.random.default_rng(salt).standard_normal((1,25,32)).astype(np.float32)
            normalized=np.asarray(predict_actions(state.params,jax.random.key(salt),obs,noise))[0]
            robot=to_robot({"actions":normalized,"state":np.asarray(obs.state)[0]})["actions"]
            if robot.shape != (25,7) or not np.isfinite(robot).all():
                raise ValueError(f"Invalid validation action at {ep.meta['id']}:{i}")
            by_episode.setdefault(ep.meta["id"],[]).append(np.abs(robot-raw["action"]).mean(axis=0))
        avg=np.mean([np.mean(rows,axis=0) for rows in by_episode.values()],axis=0)
        return {"joint_mae":float(avg[:6].mean()),"gripper_mae":float(avg[6]),
                "per_axis_mae":avg.tolist(),"episodes":len(by_episode),"anchors":len(action_points)}

    def evaluate(points, condition="annotated"):
        sums = {}
        for ix,(ep,i) in enumerate(points):
            cond = bool(ep.positive[i]) if condition=="annotated" else None
            batch = pack(ep.sample(i,cond),transform)
            loss = float(eval_step(state.params,jax.random.fold_in(seeds[1],100000+ix),batch))
            sums.setdefault(ep.meta["id"],[]).append(loss)
        return float(np.mean([np.mean(v) for v in sums.values()]))

    with sharding.set_mesh(mesh):
        baseline = run/"baseline.json"
        if not baseline.exists():
            ref = {f"validation_{k}":evaluate(validation,k) for k in ("neutral","annotated")}
            ref["action_metrics"] = action_metrics(None)
            save_json(baseline,ref)
            print("Parent validation",ref,flush=True)
        parent_actions = json.loads(baseline.read_text())["action_metrics"]
        snaps = sorted(x for x in (run/"snapshots").glob("step-*") if x.is_dir() and not x.name.endswith(".partial"))
        first = 0
        if snaps:
            snap = snaps[-1]
            first = int(snap.name.split("-")[1])
            state = install_adapter(state,cfg,snap/"adapter.npz")
            state = restore_optimizer(state,snap/"optimizer.npz",first)
            print("Resumed",first,flush=True)
        start_clock=time.monotonic()
        recent=[]
        step=first
        last_update_epoch=None
        cutoff=args.until_epoch-args.closeout_seconds if args.until_epoch else None
        updates=itertools.count(first) if cutoff else range(first,args.steps)
        for i in updates:
            if cutoff and time.time() >= cutoff:
                break
            rng = np.random.default_rng(601000+i)
            ep = train[rng.integers(len(train))]
            frame = int(rng.choice(ep.starts))
            cond = bool(ep.positive[frame]) if rng.random() >= .30 else None
            sample = pack(ep.sample(frame,cond),transform)
            state,info = train_step(seeds[1],state,sample)
            loss,grad = map(float,(info["loss"],info["grad_norm"]))
            last_update_epoch=int(time.time())
            if not np.isfinite([loss,grad]).all():
                raise ValueError(f"Nonfinite update {i+1}")
            recent.append(loss)
            step=i+1
            if step in (1,5,20) or step%50==0:
                row={"step":step,"training_flow_loss_interval":float(np.mean(recent)),
                     "last_grad_norm":grad,"elapsed_this_process_s":round(time.monotonic()-start_clock,2)}
                with (run/"training.jsonl").open("a") as log:
                    print(json.dumps(row),file=log)
                print(row,flush=True)
                recent.clear()
            last_update = bool(cutoff and time.time() >= cutoff)
            if step==100 or step%500==0 or step==args.steps or last_update:
                measure={"step":step,"validation_flow_loss":evaluate(validation)}
                action = action_metrics(True)
                measure["action_metrics"] = action
                measure["selection_score"] = .5 * (action["joint_mae"]/max(parent_actions["joint_mae"],1e-9)
                                                 +action["gripper_mae"]/max(parent_actions["gripper_mae"],1e-9))
                snap=snapshot(state,cfg,step,run,measure)
                best=run/"best.json"
                previous = json.loads(best.read_text()) if best.exists() else None
                def rank(row):
                    return row["selection_score"], row["validation_flow_loss"]
                if previous is None or rank(measure) < rank(previous):
                    save_json(best,{**measure,"snapshot":snap.name})
                print("Saved",measure,flush=True)
                if last_update:
                    break
        if not (run/"best.json").exists():
            raise RuntimeError("Time budget ended before a validated policy checkpoint was available")
        last_saved=sorted(x for x in (run/"snapshots").glob("step-*") if x.is_dir() and not x.name.endswith(".partial"))[-1]
        if int(last_saved.name.split("-")[1]) != step:
            # Preserve the last update even when the deadline occurred during validation or between batches.
            snapshot(state,cfg,step,run,{"step":step,"end_of_training":True})
        best=json.loads((run/"best.json").read_text())
        state=install_adapter(state,cfg,run/"snapshots"/best["snapshot"]/"adapter.npz")
        changed = {k:float(np.linalg.norm(np.asarray(v,dtype=np.float32)-initial[k]))
                   for k,v in adapter_dict(state.params,cfg).items()}
        if not any(x>0 for x in changed.values()):
            raise ValueError("The selected adapter has no learned updates")
        test_loss=evaluate(held_out)
        dest=run/"checkpoint"
        if not dest.exists():
            out=run/"export-work"
            out.mkdir(exist_ok=True)
            # Preserve the deployed B601 parent's native Orbax leaf paths.
            # OpenPI's restore_params unwraps this single 'value' level.
            params=jax.tree.map(lambda value: {"value":value},state.params.to_pure_dict())
            with ocp.PyTreeCheckpointer() as ck:
                ck.save(str((out/"params").resolve()),args=ocp.args.PyTreeSave({"params":params}),force=True)
            shutil.copytree(args.bundle/"checkpoint/assets",out/"assets",dirs_exist_ok=True)
            save_json(out/"conditioning.json",{"prompt":meta["positive_prompt"],"checkpoint":best})
            out.rename(dest)
        save_json(run/"result.json",{"completed_optimizer_steps":step,"minimum_steps_met":step>=args.steps,
                  "gradient_updates_ended_epoch":last_update_epoch,
                  "selected":best,"test_flow_loss":test_loss,
                  "baseline":json.loads(baseline.read_text()),"scope":"offline flow-matching, same frozen validation anchors",
                  "changed_lora_leaves":sum(v>0 for v in changed.values()),
                  "lora_update_l2":float(np.linalg.norm(list(changed.values()))),
                  "checkpoint":"checkpoint","adapter":"snapshots/"+best["snapshot"]+"/adapter.npz"})
        print("Training complete",best,"test",test_loss,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/data/b601-recap-round1"))
    parser.add_argument("--bundle",type=Path,default=PARENT)
    parser.add_argument("--steps",type=int,default=3000)
    parser.add_argument("--until-epoch",type=int,help="Absolute Unix time when this experiment must finish")
    parser.add_argument("--closeout-seconds",type=int,default=300)
    parser.add_argument("--inputs-only",action="store_true")
    options=parser.parse_args()
    if not options.inputs_only and not (options.root/"policy/parent-action-readback.json").exists():
        (options.root/"policy").mkdir(exist_ok=True)
        print("Checking the deployed parent on the fixed test observations",flush=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name("check_policy.py")),
                        "--root",str(options.root),"--parent"],check=True)
    main(options)
