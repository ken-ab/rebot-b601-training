"""Export a B601 LeRobot v3 dataset with per-episode MP4s to OpenPI's v2.1 layout."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from b601_data import ACTION_HORIZON, FPS, JOINTS, OVERHEAD, TRAIN_ID, VAL_ID, WRIST


CAMERAS = (WRIST, OVERHEAD)
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000
STAT_KEYS = ("min", "max", "mean", "std", "count")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_video(path, length, shape):
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, f"Expected one video stream in {path}")
        video = container.streams.video[0]
        require((video.height, video.width, 3) == tuple(shape), f"Unexpected RGB size in {path}")
        require(video.frames == length and video.average_rate == FPS, f"Frame count or FPS mismatch in {path}")
        require(video.codec_context.name == "h264", f"Expected H.264 in {path}")
        packets = [packet for packet in container.demux(video) if packet.size]
        require(len(packets) == length and packets[0].is_keyframe, f"Incomplete episode video: {path}")
        times = np.array([float(packet.pts * packet.time_base) for packet in packets])
        require(np.allclose(times, np.arange(length) / FPS, atol=1e-6, rtol=0), f"Unexpected video PTS in {path}")


def episode_table(root, info, episode):
    path = root / info["data_path"].format(chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
    episode_id = episode["episode_index"]
    table = pq.read_table(path, filters=[("episode_index", "=", episode_id)])
    size = episode["length"]
    frames = np.arange(size)
    require(size > 0 and table.num_rows == size, f"Episode {episode_id} row count mismatch")
    require(np.array_equal(table["frame_index"].to_numpy(), frames), f"Episode {episode_id} frame order mismatch")
    require(np.allclose(table["timestamp"].to_numpy(), frames / FPS, atol=1e-5, rtol=0),
            f"Episode {episode_id} timestamps are not on the 25 Hz grid")
    require(np.array_equal(table["index"].to_numpy(), frames + episode["dataset_from_index"]),
            f"Episode {episode_id} dataset index mismatch")
    return table


def episode_stats(episode, features, episode_index, offset):
    stats = {
        feature: {key: episode[f"stats/{feature}/{key}"] for key in STAT_KEYS}
        for feature in features
    }
    length = episode["length"]
    indices = np.arange(offset, offset + length)
    stats["index"] = {"min": [offset], "max": [offset + length - 1], "mean": [float(indices.mean())],
                      "std": [float(indices.std())], "count": [length]}
    stats["episode_index"] = {"min": [episode_index], "max": [episode_index], "mean": [episode_index],
                              "std": [0.0], "count": [length]}
    return {"episode_index": episode_index, "stats": stats}


def summarize(values):
    values = values.reshape(-1, len(JOINTS)).astype(np.float64)
    q01, q99 = np.quantile(values, (0.01, 0.99), axis=0)
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "q01": q01.tolist(), "q99": q99.tolist()}


def export_split(source, target, info, episodes, tasks, indices, train):
    offset = 0
    metadata, stats, states, windows = [], [], [], []
    task_by_id = {row["task_index"]: row["task"] for row in tasks}
    for new_id, source_id in enumerate(indices):
        episode = episodes[source_id]
        table = episode_table(source, info, episode)
        size = table.num_rows
        require(set(episode["tasks"]) == {task_by_id[task] for task in table["task_index"].to_pylist()},
                f"Episode {source_id} task mismatch")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        for key, values in (("state", state), ("action", action)):
            require(values.shape == (size, len(JOINTS)) and np.isfinite(values).all(),
                    f"Episode {source_id} has invalid {key} values")
        if train:
            # LeRobot clamps requested future indices to the last frame of this episode.
            future = np.minimum(np.arange(size)[:, None] + np.arange(ACTION_HORIZON), size - 1)
            states.append(state)
            windows.append(action[future])
        for key, values in (("episode_index", np.full(size, new_id)), ("index", np.arange(offset, offset + size))):
            column = table.schema.get_field_index(key)
            table = table.set_column(column, key, pa.array(values, type=table.schema.field(key).type))
        path_args = {"episode_chunk": new_id // CHUNK_SIZE, "episode_index": new_id}
        data_path = target / DATA_PATH.format(**path_args)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        # Let the older Hugging Face Datasets reader infer features from Arrow types.
        pq.write_table(table.replace_schema_metadata(None), data_path)

        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            require(episode[f"{prefix}/from_timestamp"] == 0 and
                    abs(episode[f"{prefix}/to_timestamp"] - size / FPS) < 1e-6,
                    f"Episode {source_id} {camera} needs its own complete video starting at t=0")
            video_source = source / info["video_path"].format(
                video_key=camera, chunk_index=episode[f"{prefix}/chunk_index"], file_index=episode[f"{prefix}/file_index"],
            )
            validate_video(video_source, size, info["features"][camera]["shape"])
            video_target = target / VIDEO_PATH.format(**path_args, video_key=camera)
            video_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video_source, video_target)

        metadata.append({"episode_index": new_id, "tasks": episode["tasks"], "length": size})
        stats.append(episode_stats(episode, info["features"], new_id, offset))
        offset += size

    count = len(indices)
    write_json(target / "meta/info.json", {
        "codebase_version": "v2.1", "robot_type": info["robot_type"], "fps": FPS,
        "total_episodes": count, "total_frames": offset, "total_tasks": len(tasks),
        "total_videos": count * len(CAMERAS), "total_chunks": (count + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE, "splits": {"train": f"0:{count}"},
        "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": info["features"],
    })
    write_jsonl(target / "meta/tasks.jsonl", tasks)
    write_jsonl(target / "meta/episodes.jsonl", metadata)
    write_jsonl(target / "meta/episodes_stats.jsonl", stats)
    report = {"repo_id": target.name, "episodes": count, "frames": offset, "source_episode_indices": indices}
    norm_stats = None
    if train:
        norm_stats = {"state": summarize(np.concatenate(states)), "actions": summarize(np.concatenate(windows))}
    return report, norm_stats


def prepare(source, split_path, output):
    source, output = source.resolve(), output.resolve()
    require(not output.exists() and not output.is_relative_to(source), f"Choose a new output outside {source}")
    info = read_json(source / "meta/info.json")
    require(info["codebase_version"] == "v3.0" and info["fps"] == FPS, "Expected LeRobot v3.0 at 25 fps")
    for feature in ("action", "observation.state"):
        spec = info["features"][feature]
        require(spec["dtype"] == "float32" and tuple(spec["shape"]) == (len(JOINTS),) and tuple(spec["names"]) == JOINTS,
                f"Unexpected B601 feature specification for {feature}")
    require({key for key, spec in info["features"].items() if spec["dtype"] == "video"} == set(CAMERAS),
            "Expected the B601 wrist and overhead streams")
    task_table = pq.read_table(source / "meta/tasks.parquet")
    task_text = "task" if "task" in task_table.column_names else "__index_level_0__"
    tasks = [{"task_index": row["task_index"], "task": row[task_text]} for row in task_table.to_pylist()]
    rows = pq.read_table(source / "meta/episodes").to_pylist()
    episodes = {row["episode_index"]: row for row in rows}
    require(len(episodes) == len(rows) == info["total_episodes"] and sum(r["length"] for r in rows) == info["total_frames"],
            "Source metadata counts do not match")
    split = read_json(split_path)
    require(set(split) == {"train", "val"} and split["train"] and split["val"] and
            Counter(split["train"] + split["val"]) == Counter(episodes.keys()),
            "Train/val must partition all source episodes exactly once")

    output.mkdir(parents=True, exist_ok=False)
    report = {"source": source.name, "source_version": "v3.0", "output_version": "v2.1",
              "fps": FPS, "action_horizon": ACTION_HORIZON}
    for name, repo_id in (("train", TRAIN_ID), ("val", VAL_ID)):
        report[name], stats = export_split(source, output / "datasets" / repo_id, info, episodes,
                                           tasks, split[name], train=name == "train")
        if stats is not None:
            write_json(output / "assets" / repo_id / "norm_stats.json", {"norm_stats": stats})
    # The launcher requires this file. An interrupted export cannot start training.
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", type=Path, default=Path(__file__).resolve().parents[1] / "splits.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.splits, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
