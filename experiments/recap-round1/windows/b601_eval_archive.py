"""Archive completed evaluation episodes on the inference server after motion stops."""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from b601_correction_recording import atomic_json


REMOTE_DIR = "/data/b601-evaluations/recap-round1-step-47000"
METADATA = {"session.json", "report.json", "outcome.json", "remote-archive.json"}


def transport(settings, mode, item, destination=None):
    ssh = settings["inference_ssh"]
    host = ssh["host"]
    user = ssh["user"]
    port = str(int(ssh["port"]))
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", host) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", user):
        raise ValueError("Invalid SSH destination")
    helper = os.environ.get("B601_EVAL_SSH_HELPER")
    if helper:
        args = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned",
                "-File", helper, mode, str(item)]
        if destination is not None:
            args.append(destination)
    elif mode == "put":
        args = ["scp", "-q", "-o", "BatchMode=yes", "-P", port, str(item), f"{user}@{host}:{destination}"]
    else:
        args = ["ssh", "-o", "BatchMode=yes", "-p", port, f"{user}@{host}", item]
    subprocess.run(args, check=True, timeout=600)


def prepare(settings, port):
    from rebot_pi05 import B601WebsocketClientPolicy, HOST

    client = B601WebsocketClientPolicy(HOST, port)
    try:
        metadata = client.get_server_metadata()
    finally:
        client.close()
    if (metadata.get("b601_checkpoint_snapshot") != "step-47000"
            or metadata.get("b601_advantage_condition") != "positive" or metadata.get("rtc_enabled")):
        raise RuntimeError("服务端模型或 RTC 设置与本批评测不匹配。")
    transport(settings, "run", f"umask 077 && mkdir -p {REMOTE_DIR} && test -w {REMOTE_DIR}")
    print(f"3090 新策略 step-47000 已连接。每条保存到服务器 {REMOTE_DIR}。", flush=True)


def publish(settings, file, name):
    with file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # Only generated episode names reach these paths. An interrupted upload cannot expose a partial result.
    target = shlex.quote(f"{REMOTE_DIR}/{name}")
    incoming = f"{REMOTE_DIR}/.{name}.{digest}.upload"
    transport(settings, "put", file, incoming)
    preserve = (f"if test -e {target}; then printf '%s  %s\\n' {digest} {target} | sha256sum -c --status; fi; "
                if name.endswith(".tar") else "")
    transport(settings, "run",
              f"set -e; cd {shlex.quote(REMOTE_DIR)}; "
              f"printf '%s  %s\\n' {digest} {shlex.quote(incoming)} | sha256sum -c --status; "
              f"{preserve}mv -f -- {shlex.quote(incoming)} {target}")
    return digest


def archive_finished(root, settings):
    root = root.resolve()
    for episode in sorted(root.glob("rollout-*")):
        if (not episode.is_dir() or episode.is_symlink() or episode.resolve().parent != root or
                not re.fullmatch(r"rollout-\d{3}-\d{8}-\d{6}-[a-f0-9]{8}", episode.name)):
            continue
        info = episode / "session.json"
        if not info.is_file() or json.loads(info.read_text(encoding="utf-8")).get("status") == "in_progress":
            continue
        receipt_file = episode / "remote-archive.json"
        receipt = json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.is_file() else {}
        archive = root / (episode.name + ".tar")
        if not receipt.get("archive_sha256"):
            files = sorted(p for p in episode.rglob("*") if p.is_file() and p.name not in {"outcome.json","remote-archive.json"})
            with tarfile.open(archive, "w") as bundle:
                for file in files:
                    if file.is_symlink() or not file.resolve().is_relative_to(episode):
                        raise ValueError("Unexpected recording path")
                    bundle.add(file, arcname=file.relative_to(episode).as_posix(), recursive=False)
            receipt["archive_sha256"] = publish(settings, archive, archive.name)
            receipt["remote_path"] = f"{REMOTE_DIR}/{archive.name}"
            atomic_json(receipt_file, receipt)
            print(f"{episode.name} 的录像和逐帧记录已经上传，服务器 SHA-256 一致。", flush=True)
        outcome = episode / "outcome.json"
        if outcome.is_file():
            reviewed = hashlib.sha256(outcome.read_bytes()).hexdigest()
            if receipt.get("outcome_sha256") != reviewed:
                receipt["outcome_sha256"] = publish(settings, outcome, episode.name + ".outcome.json")
                atomic_json(receipt_file, receipt)
        # Keep only the review metadata used by the batch counter. Preserve local media on any transfer error.
        if receipt.get("remote_path") != f"{REMOTE_DIR}/{archive.name}":
            raise ValueError("Unexpected archive receipt")
        if archive.is_file():
            with tarfile.open(archive, "r") as bundle:
                for member in bundle:
                    if not member.isfile() or member.name in METADATA:
                        continue
                    file = episode / member.name
                    if file.is_symlink() or not file.resolve().is_relative_to(episode):
                        raise ValueError("Unexpected recording path")
                    file.unlink(missing_ok=True)
            archive.unlink()
