"""Collect 100 separately reviewed, durable real rollouts across desktop restarts."""

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

from b601_correction_recording import atomic_json
from rebot_capture import CAMERA_PREFLIGHT_EXIT, CameraFrameError, require_camera_indices


HERE = Path(__file__).resolve().parent
ROOT = HERE / "corrections" / "recap-100-v1"
GOAL = 100
TASK_END = {"placement_detected", "operator_confirmed_placement", "duration_completed", "operator_q"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reviewable(path):
    info = path / "session.json"
    if not info.is_file():
        return None
    manifest = read(info)
    metadata = manifest.get("metadata", {})
    if (manifest.get("status") != "saved" or not manifest.get("frames_written") or
            manifest.get("end_reason") not in TASK_END or not metadata.get("execute") or
            metadata.get("intervention_control") != "black_follower_gravity"):
        return None
    return manifest


def scan(root):
    counted, pending = [], []
    for path in sorted(root.glob("rollout-*")):
        manifest = reviewable(path)
        if not manifest:
            continue
        outcome = path / "outcome.json"
        if outcome.is_file():
            if read(outcome)["result"] in ("success", "failure"):
                counted.append(path)
        else:
            pending.append((path, manifest))
    return counted, pending


def review(path, manifest, ask=None):
    if ask is None:
        ask = input
    while True:
        choice = ask(f"{path.name} 已存 {manifest['frames_written']} 帧，结束原因 {manifest['end_reason']}。"
                     "按实际放置结果输入 1 成功、0 失败、X 技术无效或回车稍后标记：").strip().lower()
        if choice in ("1", "0", "x", ""):
            break
    if not choice:
        return False
    human = manifest.get("frames_by_control_mode", {}).get("drag", 0) > 0
    result = {"1":"success", "0":"failure", "x":"invalid"}[choice]
    atomic_json(path / "outcome.json", {"result": result, "review_source":"operator",
        "human_intervention": human, "end_reason": manifest["end_reason"],
        "reviewed_at":datetime.now().astimezone().isoformat()})
    return True


def recoverable_recording_stop(path):
    """An episode file failed after a verified normal stop. No motor retry is issued here."""
    if not (path / "session.json").is_file() or not (path / "report.json").is_file():
        return False
    if read(path / "session.json").get("status") != "partial":
        return False
    report = read(path / "report.json")
    stop = report.get("stop", {})
    return (report.get("end_reason") in TASK_END | {"recording_error"}
            and report.get("physical_trial_started") is True
            and report.get("stop_failed") is False
            and stop.get("completed_without_errors") is True
            and stop.get("disable", {}).get("all_sdk_calls_completed") is True
            and stop.get("close", {}).get("all_calls_completed") is True)


@contextmanager
def exclusive(root):
    import msvcrt
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".collector.lock").open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("另一个窗口正在采集这一批。") from error
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def run(script, *args):
    return subprocess.run([sys.executable, "-X", "utf8", "-B", "-u", str(HERE / script), *args],
                          cwd=HERE).returncode


def main(args):
    if args == ["--count"]:
        complete, pending = scan(ROOT)
        print(f"{len(complete)}/{GOAL} 条已确认，{len(pending)} 条待确认。")
        return 0
    config = args[args.index("--config") + 1]
    delta = args[args.index("--gripper-delta-deg") + 1]
    evaluation = "--evaluation" in args
    root, goal = ROOT, GOAL
    if evaluation:
        import b601_eval_archive
        args = [arg for arg in args if arg != "--evaluation"]
        root = Path(args[args.index("--output") + 1]).resolve()
        if root == ROOT.resolve():
            raise ValueError("评测需要独立输出目录。")
        goal = 10
        settings = read(Path(config))
        b601_eval_archive.prepare(settings, int(args[args.index("--port") + 1]))
    with exclusive(root):
        deferred = set()
        while True:
            if evaluation:
                b601_eval_archive.archive_finished(root, settings)
            complete, pending = scan(root)
            for path, manifest in pending:
                if path not in deferred and not review(path, manifest):
                    deferred.add(path)
                    print("本条已保存待判，可继续采集。")
            if evaluation and pending:
                b601_eval_archive.archive_finished(root, settings)
            complete, pending = scan(root)
            if len(complete) >= goal:
                print(f"本批已完成 {goal}/{goal} 条。")
                return 0
            print(f"本批已确认 {len(complete)}/{goal} 条，已录待判 {len(pending)} 条。", flush=True)
            number = len(complete) + len(pending) + 1
            if number > goal:
                prompt = "已录满，请输入 R 补判结果，或 E 结束本批："
            else:
                prompt = f"下一条目标 {number}。纸巾复位、手已离开后输入 N 开始，R 补判，E 退出："
            choice = input(prompt).strip().lower()
            if choice == "e":
                return 0
            if choice == "r":
                deferred.clear()
                continue
            if choice != "n" or number > goal:
                continue
            try:
                require_camera_indices()
            except CameraFrameError as error:
                print(f"第 {number} 条等待相机。{error} 恢复后输入 N 重试。", flush=True)
                continue
            if run("b601_restore_start.py", "--execute", "--config", config, "--gripper-delta-deg", delta):
                return 1
            dest = root / f"rollout-{number:03d}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
            code = run("b601_correction.py", *args, "--session-dir", str(dest))
            if code == CAMERA_PREFLIGHT_EXIT:
                print(f"第 {number} 条尚未开录。相机恢复后输入 N 重试本条。", flush=True)
                continue
            if recoverable_recording_stop(dest):
                errors = read(dest / "session.json").get("errors") or ["记录未通过完整性校验"]
                print(f"本条原始数据已单独保留待技术复核，暂不计数。{'；'.join(errors)}。可复位后继续。\n{dest}", flush=True)
                continue
            if not reviewable(dest) or code:
                if evaluation:
                    b601_eval_archive.archive_finished(root, settings)
                print(f"本批暂停，真机或采集未正常结束。已确认 {len(scan(root)[0])}/{goal} 条，详见 {dest}。")
                return code or 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
