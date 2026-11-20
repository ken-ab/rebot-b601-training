"""Create checksummed, independently retryable SSH transfers of policy-training inputs."""
import argparse
import hashlib
from pathlib import Path
import tarfile

from prepare import OUTPUT, read, write_json


def archive(root, name, members):
    out = root/"transfer"/(name+".tar")
    receipt = out.with_suffix(".json")
    stamp=hashlib.sha256()
    for member in members:
        stat=(root/member).stat()
        stamp.update(f"{member} {stat.st_size} {stat.st_mtime_ns}\n".encode())
    source_signature=stamp.hexdigest()
    if receipt.exists() and out.is_file():
        old = read(receipt)
        if old["bytes"] == out.stat().st_size and old.get("source_signature") == source_signature:
            return old
    tmp = out.with_suffix(".part")
    with tarfile.open(tmp, "w") as tar:
        for member in members:
            tar.add(root/member,arcname=member,recursive=False)
    tmp.replace(out)
    with out.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    row={"file":out.name,"bytes":out.stat().st_size,"sha256":digest,"members":members,"source_signature":source_signature}
    write_json(receipt,row)
    return row


def build(root, partial=False):
    (root/"transfer").mkdir(exist_ok=True)
    records=[]
    for kind,total,start in (("demo",60,0),("new",100,1)):
        for first in range(start,start+total,10):
            ids = [f"{kind}-{j:03d}" for j in range(first,first+10)]
            if any(not (root/"episodes"/k/"episode.json").exists() for k in ids):
                if partial:
                    continue
                raise ValueError(f"Missing observations for {ids}")
            members = [f"episodes/{k}/{f}" for k in ids for f in ("views.jpgs","rows.npz","episode.json")]
            records.append(archive(root,f"{kind}-{first:03d}",members))
    if records:
        write_json(root/"transfer"/"image_parts.json",records)
    if (root/"label_summary.json").exists():
        manifest=read(root/"manifest.json")
        members=["manifest.json","label_summary.json"]+[f"labels/{r['id']}.parquet" for r in manifest["episodes"]]
        entry=archive(root,"annotations",members)
        write_json(root/"transfer"/"all_parts.json",[*records,entry])
    print("Image parts",len(records),"bytes",sum(r["bytes"] for r in records),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=OUTPUT)
    ap.add_argument("--partial",action="store_true")
    a=ap.parse_args()
    build(a.output,a.partial)
