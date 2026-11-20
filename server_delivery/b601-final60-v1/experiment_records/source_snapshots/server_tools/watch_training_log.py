"""Persist console interval metrics and stop only our job on nonfinite values."""
import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
LOG = ROOT / 'logs/formal-longrun-v1.log'
OUT = ROOT / 'CONSOLE_TRAIN_METRICS.jsonl'
OFFSET = ROOT / 'CONSOLE_WATCH_OFFSET.json'
seen = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        value = json.loads(line)
        seen.add(value['step_label'])
position = json.loads(OFFSET.read_text())['offset'] if OFFSET.exists() else 0
pattern = re.compile(r'Step\s+(\d+):\s*([^\r\n]+)')

def timestamp():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

while True:
    if LOG.exists():
        with LOG.open('rb') as source:
            source.seek(position)
            while True:
                before = source.tell()
                line = source.readline()
                if not line or not line.endswith(b'\n'):
                    source.seek(before)
                    break
                text = line.decode('utf-8', errors='replace')
                for match in pattern.finditer(text):
                    step = int(match.group(1))
                    if step in seen:
                        continue
                    fields = dict(re.findall(r'(loss|grad_norm|param_norm)=([^,\s]+)', match.group(2)))
                    if set(fields) != {'loss', 'grad_norm', 'param_norm'}:
                        continue
                    numeric = {key: float(value) for key, value in fields.items()}
                    if not all(math.isfinite(value) for value in numeric.values()):
                        alert = {'observed_at': timestamp(), 'type': 'nonfinite_training_metric',
                                 'step_label': step, 'raw_line': text, 'offset': before,
                                 'action': 'stop only b601-pipeline-20261109.service; preserve saved checkpoints'}
                        (ROOT / 'HEALTH_ALERT.json').write_text(json.dumps(alert, indent=2) + '\n')
                        subprocess.run(['systemctl', '--user', 'stop', 'b601-pipeline-20261109'], check=False)
                        raise SystemExit(2)
                    row = {'observed_at': timestamp(), 'step_label': step, 'optimizer_step_lower_bound': step + 1,
                           'metric_source': 'console interval mean, rounded to 4 decimal places',
                           'log_offset': before, **numeric}
                    with OUT.open('a', buffering=1) as output:
                        output.write(json.dumps(row, allow_nan=False) + '\n')
                    seen.add(step)
                position = source.tell()
        temporary = OFFSET.with_suffix('.tmp')
        temporary.write_text(json.dumps({'offset': position, 'checked_at': timestamp()}) + '\n')
        temporary.replace(OFFSET)
    status_file = ROOT / 'PIPELINE_STATUS.json'
    if status_file.exists() and json.loads(status_file.read_text()).get('stage') == 'complete':
        break
    time.sleep(1)
