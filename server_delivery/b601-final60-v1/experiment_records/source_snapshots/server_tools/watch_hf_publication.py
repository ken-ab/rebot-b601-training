"""Bounded read-only publication observations, excluding all credential contents."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--polls',type=int,default=50)
parser.add_argument('--interval',type=float,default=55)
args=parser.parse_args()
assert args.polls>0 and 1<=args.interval<=60
last_progress=None
for tick in range(args.polls):
    receipt=json.loads((ROOT/'receipts/hf-publication-v1.json').read_text())
    pid=receipt.get('child_pid')
    process=subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,pcpu=,rss='],capture_output=True,text=True)
    io={}
    try:
        io={key:int(value) for key,value in (line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines()) if key in ('rchar','wchar')}
    except (FileNotFoundError,ProcessLookupError):pass
    log=Path(receipt['log'])
    if log.exists():
        with log.open('rb') as source:
            source.seek(max(0,log.stat().st_size-65536));tail=source.read().decode(errors='replace')
        for line in tail.splitlines():
            try:value=json.loads(line)
            except ValueError:continue
            if isinstance(value,dict) and value.get('stage')=='upload_batch':
                last_progress={k:value[k] for k in ('stage','repository','first','files','total_files')}
    service=subprocess.check_output(['systemctl','--user','show','b601-hf-publication-20261110',
                                     '--property=ActiveState,MainPID,ExecMainStatus'],text=True).strip()
    row={'at':datetime.datetime.now().astimezone().isoformat(),'receipt_status':receipt['status'],
         'pid':pid,'process_live':process.returncode==0,'process':process.stdout.strip(),
         'process_io_counters':io,'log_bytes':log.stat().st_size if log.exists() else 0,
         'last_batch':last_progress,'service':service,
         'verified_publication_receipt_exists':(ROOT/'HF_PUBLICATION_RECEIPT.json').exists()}
    with (ROOT/'HF_PUBLICATION_OBSERVATIONS.jsonl').open('a') as output:output.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if receipt['status']!='running' or 'ActiveState=active' not in service:
        print('Publication terminal/service state observed; return control for inspection.',flush=True)
        break
    if tick+1<args.polls:time.sleep(args.interval)
