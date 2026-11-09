"""Run a durable stage and write its exit receipt outside the training repository."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
label = sys.argv[1]
command = sys.argv[2:]
if not command or Path(label).name != label:
    raise SystemExit("run_stage.py LABEL COMMAND [ARGS...]")
log = ROOT / "logs" / f"{label}.log"
receipt = ROOT / "receipts" / f"{label}.json"
if log.exists() or receipt.exists():
    raise SystemExit(f"Refusing to overwrite stage evidence: {label}")
start = time.time()
record = {"label": label, "command": command, "started_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
          "start_unix": start, "log": str(log), "runner_pid": os.getpid(), "status": "running"}
receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
with log.open("w", buffering=1) as output:
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                             cwd="/home/yujiegao/桌面/zzk/rebot-b601-training")
    record["child_pid"] = child.pid
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    code = child.wait()
record.update(returncode=code, status="passed" if code == 0 else "failed", elapsed_seconds=time.time() - start,
              finished_at=datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False))
raise SystemExit(code)
