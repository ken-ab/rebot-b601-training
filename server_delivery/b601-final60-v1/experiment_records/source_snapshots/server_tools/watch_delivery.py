"""Read-only, bounded observations of evaluation, export and final acceptance."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--polls',type=int,default=50)
parser.add_argument('--interval',type=float,default=55)
args = parser.parse_args()
assert args.polls>0 and 1<=args.interval<=60


def read(path):
    return json.loads(Path(path).read_text())


def worker(stage):
    path=ROOT/'receipts'/(stage+'.json')
    if not path.exists():return None
    value=read(path)
    pid=value.get('child_pid')
    process=subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,pcpu=,rss='],capture_output=True,text=True) if pid else None
    return {'pid':pid,'receipt_status':value['status'],
            'live':process.returncode==0 if process else None,
            'ps':process.stdout.strip() if process else None}


for tick in range(args.polls):
    main=read(ROOT/'PIPELINE_STATUS.json')
    final=read(ROOT/'FINAL_ACCEPTANCE_STATUS.json')
    services=subprocess.check_output(['systemctl','--user','show','b601-pipeline-20261109',
        'b601-final-acceptance-20261109','--property=Id,ActiveState,MainPID,ExecMainStatus'],text=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu',
                                  '--format=csv,noheader'],text=True).strip().splitlines()
    archive=ROOT/'B601_PI05_DEPLOYMENT.tar.gz'
    row={'at':datetime.datetime.now().astimezone().isoformat(),
         'main_stage':main['stage'],'main_worker':worker(main['stage']),
         'final_stage':final['stage'],'final_worker':worker(final['stage']),
         'services':services,'gpu2_3':gpu[2:],
         'archive_bytes':archive.stat().st_size if archive.exists() else None}
    with (ROOT/'DELIVERY_OBSERVATIONS.jsonl').open('a') as output:output.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if 'failed' in main['stage'] or 'failed' in final['stage'] or final['stage']=='complete_pending_root_completion_audit':
        print('Terminal state observed; return control for root inspection.',flush=True)
        break
    if tick+1<args.polls:time.sleep(args.interval)
