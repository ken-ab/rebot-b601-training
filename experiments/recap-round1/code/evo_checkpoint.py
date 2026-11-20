"""Materialize the selected OpenPI checkpoint locally from the exact parent and its LoRA update."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import ml_dtypes
import numpy as np
import tensorstore as ts


def write(path, obj):
    temp = path.with_name(path.name+".partial")
    temp.write_text(json.dumps(obj,sort_keys=True,indent=2),encoding="utf-8")
    temp.replace(path)


def open_store(params):
    return ts.KvStore.open({"driver":"ocdbt","base":{"driver":"file","path":params.resolve().as_posix()+"/"}}).result()


def leaves(kv):
    return sorted(k.decode()[:-len("/.zarray")] for k in kv.list().result() if k.endswith(b"/.zarray"))


def array(kv, key):
    return ts.open({"driver":"zarr","kvstore":kv.spec().to_json(),"path":key},open=True).result()


def block_slices(shape, bytes_per):
    if not shape:
        yield ()
        return
    rows=max(1,48*1024**2//max(1,math.prod(shape[1:])*bytes_per))
    for start in range(0,shape[0],rows):
        yield slice(start,min(start+rows,shape[0]))


def store_record(source, dtype, dest=None):
    sha=hashlib.sha256()
    for ix in block_slices(source.shape, max(4,np.dtype(dtype).itemsize)):
        raw=source[ix].read().result()
        values=np.ascontiguousarray(raw,dtype=dtype)
        if dest is not None:
            dest[ix].write(values).result()
        sha.update(values.tobytes(order="C"))
    return {"shape":list(source.shape),"dtype":np.dtype(dtype).name,"sha256":sha.hexdigest()}


def created_leaf(target, key, source_kv, dtype):
    m=json.loads(source_kv.read(key+"/.zarray").result().value)
    m.pop("dtype")
    if m.get("fill_value") is None:
        m.pop("fill_value", None)
    return ts.open({"driver":"zarr","kvstore":target.spec().to_json(),"path":key,
                    "dtype":np.dtype(dtype).name,"metadata":m},open=True,create=True).result()


def prepare(parent, dest, state_dir):
    if parent.resolve()==dest.resolve():
        raise ValueError("Local export requires a different destination from the parent")
    state_dir.mkdir(parents=True,exist_ok=True)
    target_params=dest/"params"
    target_params.mkdir(parents=True,exist_ok=True)
    source_kv=open_store(parent/"params")
    new_kv=open_store(target_params)
    keys=leaves(source_kv)
    if not keys or not any("lora_" in k for k in keys):
        raise ValueError("Original complete LoRA checkpoint required")
    for k in keys:
        receipt=state_dir/(k+".json")
        if receipt.exists() and new_kv.read(k+"/.zarray").result().state=="value":
            continue
        source=array(source_kv,k)
        # This is the exact OpenPI init_train_state cast used during this run.
        dtype=source.dtype.numpy_dtype if "lora" in k else ml_dtypes.bfloat16
        target=created_leaf(new_kv,k,source_kv,dtype)
        record=store_record(source,dtype,target)
        write(receipt,record)
        print("Prepared",k,record["dtype"],flush=True)
    for item in (parent/"params").iterdir():
        if item.is_file() and item.name.startswith("_"):
            shutil.copy2(item,target_params/item.name)
        if item.name=="array_metadatas" and item.is_dir():
            shutil.copytree(item,target_params/item.name,dirs_exist_ok=True)
    shutil.copytree(parent/"assets",dest/"assets",dirs_exist_ok=True)
    write(state_dir/"core-complete.json",{"leaves":len(keys),"parent":str(parent.resolve())})


def overlay(dest, adapter, state_dir):
    kv=open_store(dest/"params")
    target_keys={key for key in leaves(kv) if "lora" in key}
    with np.load(adapter) as updated:
        mapping={}
        for name in updated.files:
            key="params."+name.replace("/",".")
            if key not in target_keys and key+".value" in target_keys:
                key+=".value"
            mapping[name]=key
        if len(set(mapping.values()))!=len(mapping) or set(mapping.values())!=target_keys:
            raise ValueError("Selected LoRA file does not match every native LoRA leaf")
        # Only publish the merged checkpoint after all leaves and their hashes match.
        for name,key in mapping.items():
            target=array(kv,key)
            data=updated[name]
            if tuple(data.shape)!=tuple(target.shape) or not np.isfinite(data).all():
                raise ValueError(f"Invalid local LoRA update {name}")
            target.write(data.astype(target.dtype.numpy_dtype,copy=False)).result()
            verified=store_record(target,target.dtype.numpy_dtype)
            expected=hashlib.sha256(data.astype(target.dtype.numpy_dtype).tobytes(order="C")).hexdigest()
            if verified["sha256"]!=expected:
                raise ValueError(f"LoRA readback differs at {name}")
    write(state_dir/"last-overlay.json",{"snapshot":str(adapter),"updated_leaves":len(mapping)})


def fingerprint(checkpoint, file):
    kv=open_store(checkpoint/"params")
    out={}
    for key in leaves(kv):
        source=array(kv,key)
        out[key]=store_record(source,source.dtype.numpy_dtype)
    if not out:
        raise ValueError("Empty native parameter store")
    write(file,out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","overlay","fingerprint","verify"])
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--parent",type=Path)
    p.add_argument("--state-dir",type=Path)
    p.add_argument("--adapter",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--reference",type=Path)
    a=p.parse_args()
    if a.stage=="prepare":
        prepare(a.parent,a.checkpoint,a.state_dir)
    elif a.stage=="overlay":
        overlay(a.checkpoint,a.adapter,a.state_dir)
    else:
        result=fingerprint(a.checkpoint,a.output)
        if a.stage=="verify":
            reference=json.loads(a.reference.read_text(encoding="utf-8"))
            differences=[k for k in sorted(set(result)|set(reference)) if result.get(k)!=reference.get(k)]
            if differences:
                raise ValueError(f"Local and cloud native arrays differ: {differences}")
            print("Verified",len(result),"native arrays against cloud Orbax output")
