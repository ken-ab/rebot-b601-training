"""Read-only current status for the B601 long run."""
from collections import deque
import datetime
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
MONITOR = Path('/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1_monitor')

def read(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {'error': 'write in progress'}

def latest_line(path):
    if not path.exists():
        return None
    with path.open() as source:
        lines = deque(source, maxlen=1)
    return json.loads(lines[0]) if lines else None

status = {'observed_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'pipeline': read(ROOT / 'PIPELINE_STATUS.json')}
for name in ('SESSION_START.json', 'LATEST.json', 'BEST.json', 'BASELINE_FLOW.json', 'LONGRUN_COMPLETE.json'):
    value = read(MONITOR / name)
    if value is not None:
        status[name.removesuffix('.json').lower()] = value
metric = latest_line(MONITOR / 'TRAIN_METRICS.jsonl')
if metric is None:
    metric = latest_line(ROOT / 'CONSOLE_TRAIN_METRICS.jsonl')
    if metric is not None:
        metric = {**metric, 'at': metric['observed_at']}
if metric:
    status['last_logged_training_metrics'] = metric
    status['seconds_since_last_metric_observation'] = time.time() - datetime.datetime.fromisoformat(metric['at']).timestamp()
session = status.get('session_start')
if session and not status.get('longrun_complete'):
    start = datetime.datetime.fromisoformat(session['first_dispatch_completed_at']).timestamp()
    status['live_active_hours_estimate'] = (session['previous_active_seconds'] + time.time() - start) / 3600
    earliest = start + session['target_active_seconds'] - session['previous_active_seconds']
    status['earliest_time_gate'] = datetime.datetime.fromtimestamp(earliest, ZoneInfo('Asia/Shanghai')).isoformat()
result = subprocess.run(['systemctl', '--user', 'show', 'b601-pipeline-20261109', '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ExecMainStatus'],
                        capture_output=True, text=True, timeout=10)
status['service'] = result.stdout.strip()
result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=10)
status['gpu_snapshot'] = result.stdout.strip().splitlines()
print(json.dumps(status, ensure_ascii=False, indent=2))
