"""Prepare independently addressable offline observations from the three B601 data sources."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import cv2
import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
LERO = PROJECT / "lerobot-windows"
OUTPUT = LERO / "training/recap-round1"
DEMO = PROJECT / "openpi-b601-final-60"
RAW = LERO / "corrections/recap-100-v1"
LEGACY = LERO / "inference-evals/rtc-comparison-20261111"
OLD = (3, 5, 6, 7, 8, 9, 12, 14)
VIEWS = ("overhead", "wrist")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
NORMAL_END = {"placement_detected", "operator_confirmed_placement", "duration_completed", "operator_q"}
SCHEMA = 1


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def one(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one file, got {paths}")
    return paths[0]


def movie_frames(path):
    with av.open(str(path)) as stream:
        video = stream.streams.video[0]
        video.codec_context.thread_count = 2
        for frame in stream.decode(video):
            if (frame.width, frame.height) != (640, 480):
                raise ValueError(f"Unexpected source geometry in {path}")
            yield frame.reformat(width=224, height=168, format="bgr24", interpolation="AREA").to_ndarray()


def legacy_images(folder, frames, view):
    for row in frames:
        src = folder / row["images"][f"observation.images.{view}"]
        im = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape != (480, 640, 3):
            raise ValueError(f"Invalid legacy frame {src}")
        yield cv2.resize(im, (224, 168), interpolation=cv2.INTER_AREA)


def inputs():
    splits = read(DEMO / "splits.json")
    for split in ("train", "val"):
        ds = DEMO / "prepared/datasets" / f"rebot-b601-{split}"
        for index, orig in enumerate(splits[split]):
            stem = f"episode_{index:06d}"
            data = one((ds / "data").glob(f"*/{stem}.parquet"))
            videos = [one((ds / "videos").glob(f"*/observation.images.{v}/{stem}.mp4")) for v in VIEWS]
            yield dict(id=f"demo-{orig:03d}", kind="demo", success=True, human=False, split=split,
                       data=data, videos=videos, path=ds, origin=orig)
    reviews = {r["trial"]: r for r in read(LEGACY / "reviewed-results.json")}
    for number in OLD:
        row = reviews[number]
        if not row["valid_start"] or row["manual_confirmed"] or row["infrastructure_interrupted"]:
            raise ValueError(f"Unexpected historical review for {number}")
        yield dict(id=f"old-{number:02d}", kind="old", success=row["autonomous_success"], human=False,
                   path=LERO / row["frames"], log=LERO / row["log"], row=row, origin=number)
    seen = set()
    for path in sorted(RAW.glob("rollout-*")):
        if not (path / "outcome.json").is_file() or not (path / "session.json").is_file():
            continue
        outcome, session = read(path / "outcome.json"), read(path / "session.json")
        if outcome["result"] not in ("success", "failure"):
            continue
        if (session["status"] != "saved" or session["end_reason"] not in NORMAL_END
                or session["fps"] != 25 or not session["metadata"]["execute"]
                or session["metadata"]["intervention_control"] != "black_follower_gravity"
                or session["joint_order"] != JOINTS or not outcome.get("reviewed_at")
                or not outcome.get("review_source")):
            raise ValueError(f"Inconsistent adjudication or configuration in {path}")
        index = int(path.name.split("-")[1])
        if index in seen:
            raise ValueError(f"Multiple confirmed attempts for rollout {index}")
        seen.add(index)
        yield dict(id=f"new-{index:03d}", kind="new", success=outcome["result"] == "success",
                   human=outcome["human_intervention"], origin=index, path=path, session=session,
                   data=one((path / "data").rglob("*.parquet")),
                   videos=[one((path / f"videos/observation.images.{v}").rglob("*.mp4")) for v in VIEWS])


def source_signature(r):
    files = [r["data"], *r["videos"]] if "data" in r else [r["log"], r["path"] / "frames.jsonl",
        LEGACY / f"evidence/trial-{r['origin']:02d}-frame-index.json", LEGACY / "reviewed-results.json"]
    if r["kind"] == "new":
        files.extend(r["path"] / key for key in ("session.json", "outcome.json", r["session"]["telemetry"]))
    data = [(str(p.relative_to(PROJECT)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps([SCHEMA, data]).encode()).hexdigest()


def material(r):
    if r["kind"] == "old":
        raw = r["log"].read_bytes()
        if hashlib.sha256(raw).hexdigest() != r["row"]["sha256"]:
            raise ValueError(f"Archived run changed: {r['id']}")
        trace = json.loads(raw)["steps"]
        t = np.array([x["seconds"] for x in trace])
        frames = read(LEGACY / f"evidence/trial-{r['origin']:02d}-frame-index.json")
        frames = [f for f in frames if f["model_seconds"] >= 0]
        if r["origin"] in (3, 6):
            stop = next(i for i, f in enumerate(frames) if f["model_seconds"] >= 86)
            frames = frames[:stop+1]
        elif not r["success"]:
            frames = [f for f in frames if f["model_seconds"] <= 120]
        clock = np.array([f["model_seconds"] for f in frames])
        # First live request is the historical clock origin. Never read a future motor sample.
        ix = np.maximum(0, np.searchsorted(t, clock, side="right") - 1)
        state = np.array([trace[i]["measured_motor_deg"] for i in ix], dtype=np.float32)
        n = len(frames)
        arrays = dict(time=clock, source_frame=np.array([f["sequence"] for f in frames]),
                      state=state, mode=np.zeros(n, np.int8), epoch=np.zeros(n, np.int32))
        return arrays, [legacy_images(r["path"], frames, v) for v in VIEWS]
    tab = pq.read_table(r["data"])
    n = tab.num_rows
    def column(key, dtype):
        return np.array(tab[key].to_pylist(), dtype=dtype)
    state = column("observation.state", np.float32)
    action = column("action", np.float32)
    ix = column("frame_index", np.int32)
    if not np.array_equal(ix, np.arange(n)) or state.shape != (n, 7) or action.shape != (n, 7):
        raise ValueError(f"Bad columns in {r['id']}")
    clock = column("timestamp", np.float64)
    mode = np.ones(n, np.int8)  # 0 excluded, 1 rank by advantage, 2 forced human correction
    epoch = np.zeros(n, np.int32)
    if r["kind"] == "new":
        sess = r["session"]
        with (r["path"] / sess["telemetry"]).open(encoding="utf-8") as fh:
            trace = [json.loads(line) for line in fh]
        if len(trace) != n or n != sess["frames_written"]:
            raise ValueError(f"Missing control rows in {r['id']}")
        for key, actual in (("measured_motor_deg", state), ("action_adapter_deg", action)):
            if not np.allclose(actual, [x[key] for x in trace], atol=1e-4, rtol=1e-6):
                raise ValueError(f"Motor or adapter coordinates differ in {r['id']}")
        pool = column("recap.advantage_pool", bool).reshape(-1)
        allowed = column("recap.policy_eligible", bool).reshape(-1)
        forced = column("recap.advantage_tag", int).reshape(-1) == 1
        src = np.array([x["command_source"] for x in trace])
        if (not np.array_equal(allowed, forced | pool) or np.any(forced & pool)
                or np.any(forced & (src != "hand_drag")) or np.any(pool & (src != "model"))
                or bool((src == "hand_drag").any()) != r["human"]
                or any(not all(row["camera_valid"][v] for v in VIEWS) for row in trace)
                or any(flag and not row.get("next_state_observed") for flag, row in zip(forced, trace))):
            raise ValueError(f"Invalid camera or correction annotation in {r['id']}")
        mode = np.where(forced, 2, pool).astype(np.int8)
        epoch = np.array([x["control_epoch"] for x in trace], dtype=np.int32)
        clock = np.array([x["seconds"] for x in trace])
    arrays = dict(time=clock-clock[0], source_frame=ix, state=state, action=action, mode=mode, epoch=epoch)
    return arrays, [movie_frames(path) for path in r["videos"]]


def prepare_one(output, row):
    final = output / "episodes" / row["id"]
    signature = source_signature(row)
    if final.is_dir():
        meta = read(final / "episode.json")
        if meta["signature"] != signature:
            raise ValueError(f"Source changed since preparation: {row['id']}")
        return meta
    arrays, streams = material(row)
    n = len(arrays["time"])
    if (n < 2 or np.any(np.diff(arrays["time"]) <= 0)
            or any(not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError(f"Invalid numeric trace in {row['id']}")
    with TemporaryDirectory(dir=final.parent, prefix=".building-") as tmp:
        work = Path(tmp) / "data"
        work.mkdir()
        offsets = np.zeros((n, 3), dtype=np.int64)
        with (work / "views.jpgs").open("wb") as out:
            count = 0
            for i, pair in enumerate(zip(*streams, strict=True)):
                if i >= n:
                    raise ValueError(f"Too many frames in {row['id']}")
                offsets[i, 0] = out.tell()
                for j, image in enumerate(pair, 1):
                    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 93])
                    if not ok:
                        raise ValueError(f"Image encoder failed in {row['id']}")
                    out.write(data)
                    offsets[i, j] = out.tell()
                count += 1
        if count != n:
            raise ValueError(f"Only {count}/{n} paired images in {row['id']}")
        np.savez(work / "rows.npz", **arrays, offsets=offsets)
        meta = {k:row[k] for k in ("id","kind","origin","success","human")}
        meta.update(n=n, seconds=float(arrays["time"][-1]), signature=signature,
                    source=str(row["path"].relative_to(PROJECT).as_posix()), demo_split=row.get("split"),
                    scored_pool=int((arrays["mode"] == 1).sum()), corrected=int((arrays["mode"] == 2).sum()))
        write_json(work / "episode.json", meta)
        work.rename(final)
    print(f"Prepared {row['id']}  {n} paired frames", flush=True)
    return meta


def holdout(rows, count, salt):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["human"], r["success"])].append(r["id"])
    exact = {key:len(v)*count/len(rows) for key,v in groups.items()}
    seats = {key:int(v) for key,v in exact.items()}
    order = sorted(groups, key=lambda k: (-(exact[k]-seats[k]), k))
    for key in order[:count-sum(seats.values())]:
        seats[key] += 1
    return {rid for k,group in groups.items() for rid in sorted(group,
            key=lambda rid:hashlib.sha256(f"{salt}:{rid}".encode()).digest())[:seats[k]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--freeze", action="store_true", help="Require all 100 confirmed episodes and save fixed splits")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    (args.output / "episodes").mkdir(parents=True, exist_ok=True)
    records = sorted(inputs(), key=lambda r: (r["kind"] != "old", r["kind"], r["origin"]))
    if args.limit:
        records = records[:args.limit]
    metas = [prepare_one(args.output, r) for r in records]
    kinds = Counter(r["kind"] for r in metas)
    print("Prepared", dict(kinds), flush=True)
    if args.freeze:
        if kinds != {"demo":60,"old":8,"new":100} or len({r['id'] for r in metas}) != 168:
            raise ValueError(f"Cannot freeze the requested cohort yet: {kinds}")
        new = [r for r in metas if r["kind"]=="new"]
        test = holdout(new,10,"b601-round1-test")
        val = holdout([r for r in new if r["id"] not in test],10,"b601-round1-val")
        val |= holdout([r for r in metas if r["kind"]=="old"],2,"b601-round1-old")
        for r in metas:
            r["split"] = r["demo_split"] or ("test" if r["id"] in test else "val" if r["id"] in val else "train")
        manifest = {"schema":SCHEMA,"views":list(VIEWS),"joints":JOINTS,"episodes":metas}
        path = args.output / "manifest.json"
        if path.exists() and read(path) != manifest:
            raise ValueError("A different frozen cohort already occupies this output directory")
        write_json(path, manifest)
        print("Frozen split", dict(Counter(r["split"] for r in metas)), flush=True)


if __name__ == "__main__":
    main()
