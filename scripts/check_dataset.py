"""Check the transferred LeRobot dataset with its real video decoder."""

import argparse
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK = "拿起蓝色餐巾纸，把它放进黑色框里"
CAMERAS = ("observation.images.overhead", "observation.images.wrist")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", default="local/blue-tissue-center-20261109-final-53")
    parser.add_argument("--episodes", type=int, default=53)
    parser.add_argument("--frames", type=int, default=30915)
    args = parser.parse_args()
    require((args.root / "meta/info.json").is_file(), f"数据目录缺少 meta/info.json，{args.root}")
    dataset = LeRobotDataset(
        args.repo_id, root=args.root, video_backend="pyav",
        delta_timestamps={"action": [i / 25 for i in range(8)]},
    )
    require(dataset.num_episodes == args.episodes and len(dataset) == args.frames, "数据条数或帧数与清单不符。")
    require(dataset.meta.fps == 25 and list(dataset.meta.tasks.index) == [TASK], "帧率或任务文本不符。")
    for feature in ("observation.state", "action"):
        require(tuple(dataset.meta.features[feature]["shape"]) == (7,), f"{feature} 需要 7 维。")
        require({"q01", "q99"} <= dataset.meta.stats[feature].keys(), f"{feature} 缺少分位数统计。")
    for row in dataset.meta.episodes:
        start, end = row["dataset_from_index"], row["dataset_to_index"]
        for index in (start, (start + end) // 2, end - 1):
            sample = dataset[index]
            require(sample["episode_index"].item() == row["episode_index"] and sample["task"] == TASK,
                    f"第 {index} 帧的条号或任务不匹配。")
            require(tuple(sample["action"].shape) == (8, 7) and
                    bool(torch.isfinite(sample["action"]).all()) and
                    bool(torch.isfinite(sample["observation.state"]).all()), f"第 {index} 帧的状态或动作无效。")
            for camera in CAMERAS:
                require(tuple(sample[camera].shape) == (3, 480, 640), f"第 {index} 帧的 {camera} 尺寸无效。")
    print(f"通过，共 {dataset.num_episodes} 条 episode、{len(dataset)} 帧，两路相机可读取。")


if __name__ == "__main__":
    main()
