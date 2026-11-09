"""Compact read-only status for the continuing Pi05 run and independent ACT arm."""
import datetime
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
ACT = ROOT.parent/'B601_ACT_20261109'
MONITOR = Path('/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1_monitor')
ACT_RUN = Path('/data2/yujiegao/b601-act-final60/runs/act-b601-final60-v1')


def read(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {'write_in_progress': True}


def last_line(path):
    if path.exists():
        for line in reversed(path.read_text().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue


now = datetime.datetime.now(ZoneInfo('Asia/Shanghai'))
session = read(MONITOR/'SESSION_START.json')
result = {'observed_at': now.isoformat(),
    'pi05_stage': read(ROOT/'PIPELINE_STATUS.json'),
    'pi05_latest': read(MONITOR/'LATEST.json'), 'pi05_best': read(MONITOR/'BEST.json'),
    'pi05_last_log': last_line(ROOT/'CONSOLE_TRAIN_METRICS.jsonl'),
    'pi05_complete': read(MONITOR/'LONGRUN_COMPLETE.json'),
    'final_acceptance': read(ROOT/'FINAL_ACCEPTANCE_STATUS.json'),
    'act_stage': read(ACT/'PIPELINE_STATUS.json'), 'act_latest': read(ACT_RUN/'LATEST.json'),
    'act_best': read(ACT_RUN/'BEST.json'), 'act_last_log': last_line(ACT_RUN/'TRAIN_METRICS.jsonl'),
    'act_delivery': read(ACT/'DELIVERY_STATUS.json')}
act_log = ACT_RUN/'TRAIN_METRICS.jsonl'
if act_log.exists():
    result['act_training_log_age_seconds'] = time.time()-act_log.stat().st_mtime
if result['pi05_complete']:
    result['pi05_live_active_hours_estimate'] = result['pi05_complete']['active_seconds']/3600
elif session:
    result['pi05_live_active_hours_estimate'] = (session['previous_active_seconds']+
        now.timestamp()-datetime.datetime.fromisoformat(session['first_dispatch_completed_at']).timestamp())/3600
result['services'] = subprocess.check_output(['systemctl','--user','show',
    'b601-pipeline-20261109','b601-act-pipeline-20261109','b601-act-delivery-20261109',
    'b601-final-acceptance-20261109',
    '-p','Id','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True).strip()
if session:
    process = subprocess.run(['ps','-p',str(session['pid']),'-o','pid,etime,pcpu,rss'],
        capture_output=True,text=True)
    result['pi05_training_process'] = {'pid':session['pid'], 'live':process.returncode==0,
        'observation':process.stdout.strip()}
result['gpus'] = subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip().splitlines()
print(json.dumps(result,ensure_ascii=False,indent=2))
