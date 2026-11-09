"""Wait for real ACT training, export selected weights, verify, archive."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
LOGROOT = ROOT.parent/'B601_LONGRUN_20261109'
PYTHON = '/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/.venv/bin/python'
TRAIN = Path('/data2/yujiegao/b601-act-final60/runs/act-b601-final60-v1')
BUNDLE = ROOT/'B601_ACT_DEPLOYMENT'


def status(stage, **extra):
    row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage, **extra}
    temporary = ROOT/'DELIVERY_STATUS.tmp'
    temporary.write_text(json.dumps(row,indent=2)+'\n')
    temporary.replace(ROOT/'DELIVERY_STATUS.json')
    print(json.dumps(row),flush=True)


def run(label, command, gpu=False):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=(
        'GPU-3630f58a-b268-bb35-863e-7565f2e026b9' if gpu else ''), PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    subprocess.run([sys.executable,str(LOGROOT/'run_stage.py'),label,*command],env=environment,check=True)
    receipt = json.loads((LOGROOT/'receipts'/(label+'.json')).read_text())
    assert receipt['status']=='passed'
    return receipt


try:
    status('waiting_real_act_training_completion')
    while True:
        file = ROOT/'PIPELINE_STATUS.json'
        if file.exists():
            pipeline = json.loads(file.read_text())
            if pipeline['stage']=='failed_needs_agent_analysis':
                raise RuntimeError(pipeline)
            if pipeline['stage']=='training_complete_needs_deployment_export':
                break
        time.sleep(5)
    completion = json.loads((TRAIN/'COMPLETE.json').read_text())
    best = json.loads((TRAIN/'BEST.json').read_text())
    assert completion['status']=='complete' and completion['step']>=20000
    checkpoint = Path(best['checkpoint'])
    status('exporting_selected_act', selected=best, completion=completion)
    run('act-final-export-v1',[PYTHON,str(ROOT/'build_runtime.py'),'--checkpoint',str(checkpoint),
        '--bundle',str(BUNDLE),'--scope','final'])
    receipts = []
    for device in ('cpu','cuda'):
        status('verifying_final_act_'+device)
        receipts.append(run('act-final-selftest-'+device+'-v1',
            [PYTHON,str(BUNDLE/'self_test.py'),'--device',device],gpu=(device=='cuda')))
        receipts.append(run('act-final-full-'+device+'-v1',
            [PYTHON,str(ROOT/'verify_runtime_full.py'),'--bundle',str(BUNDLE),'--device',device,
             '--output',str(ROOT/'diagnostics'/('runtime_final_'+device+'_v1'))],gpu=(device=='cuda')))
    verification = BUNDLE/'verification'
    verification.mkdir()
    for receipt in receipts:
        label = Path(receipt['log']).stem
        shutil.copy2(LOGROOT/'receipts'/(label+'.json'),verification/(label+'.json'))
        for line in reversed(Path(receipt['log']).read_text().splitlines()):
            try:
                value=json.loads(line)
            except ValueError:
                continue
            if value.get('status')=='passed':
                (verification/(label+'-result.json')).write_text(json.dumps(value,indent=2)+'\n')
                break
    for name in ('RUN_CONTRACT.json','SMOKE_ACCEPTANCE.json','SMOKE_READBACK.json','CACHE_TARGETS_AUDIT.json','SOURCE_LOCK.json'):
        shutil.copy2(ROOT/name,verification/name)
    manifest={}
    for path in BUNDLE.rglob('*'):
        if path.is_file() and path.name!='MANIFEST.json' and '__pycache__' not in path.parts:
            digest=hashlib.sha256()
            with path.open('rb') as source:
                while block:=source.read(8*1024*1024):digest.update(block)
            manifest[str(path.relative_to(BUNDLE))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    (BUNDLE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Verify the complete delivery inventory after adding actual test receipts.
    for relative,expected in manifest.items():
        assert hashlib.sha256((BUNDLE/relative).read_bytes()).hexdigest()==expected['sha256']
    archive=ROOT/'B601_ACT_DEPLOYMENT.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as output:
        output.add(BUNDLE,arcname=BUNDLE.name,filter=lambda entry:None if '__pycache__' in entry.name or entry.name.endswith('.pyc') else entry)
    digest=hashlib.sha256()
    with archive.open('rb') as source:
        while block:=source.read(8*1024*1024):digest.update(block)
    (ROOT/'B601_ACT_DEPLOYMENT.sha256').write_text(digest.hexdigest()+'  '+archive.name+'\n')
    result={'status':'complete','bundle':str(BUNDLE),'archive':str(archive),'archive_bytes':archive.stat().st_size,
        'archive_sha256':digest.hexdigest(),'selected':best,'training_completion':completion,
        'cpu_and_gpu_full5118_readback_passed':True,'manifest_files':len(manifest),
        'windows_machine_executed_here':False,'robot_execution_tested':False,
        'main_pi05_longrun_still_requires_its_own12h_completion_and_delivery':True}
    (ROOT/'DELIVERY_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    status('act_delivery_complete_main_pi05_separate',delivery=result)
except Exception as exc:
    status('failed_needs_agent_analysis',error=repr(exc))
    raise
