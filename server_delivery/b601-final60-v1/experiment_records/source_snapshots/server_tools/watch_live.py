"""Bounded read-only observations; this process never owns or restarts training."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--polls', type=int, default=34)
parser.add_argument('--interval', type=float, default=55)
args = parser.parse_args()
assert args.polls > 0 and 1 <= args.interval <= 60
for tick in range(args.polls):
    observation = subprocess.run(['python3', str(ROOT / 'collect_all_status.py')],
                                 text=True, capture_output=True)
    if observation.returncode:
        print(json.dumps({'observation_error': observation.stderr,
                          'returncode': observation.returncode}), flush=True)
    else:
        status = json.loads(observation.stdout)
        services = {}
        for block in status['services'].split('\n\n'):
            fields = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
            if fields.get('Id'):
                services[fields['Id']] = fields
        row = {'at': status['observed_at'],
               'active_hours_estimate': round(status['pi05_live_active_hours_estimate'], 4),
               'training': status['pi05_training_process'],
               'stage': status['pi05_stage']['stage'], 'last_log': status['pi05_last_log'],
               'latest': status['pi05_latest'], 'final_acceptance': status['final_acceptance'],
               'services': services, 'gpus': status['gpus']}
        with (ROOT / 'WAIT_OBSERVATIONS.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        main = services['b601-pipeline-20261109.service']
        final = services['b601-final-acceptance-20261109.service']
        print(json.dumps({'at': row['at'], 'hours': row['active_hours_estimate'],
            'trainer_live': row['training']['live'], 'step': row['last_log']['step_label'],
            'loss': row['last_log']['loss'], 'checkpoint': row['latest']['step_label'],
            'flow_validation': row['latest']['validation_flow_loss'], 'stage': row['stage'],
            'final': row['final_acceptance']['stage'], 'main_service': main['ActiveState'],
            'final_service': final['ActiveState'], 'gpu2_3': row['gpus'][2:]}), flush=True)
        if (row['stage'] != 'formal-longrun-v1' or not row['training']['live'] or
                row['final_acceptance']['stage'] != 'waiting_actual_main_pipeline_completion' or
                main['ActiveState'] != 'active' or final['ActiveState'] != 'active'):
            print('Observed stage/liveness change; return control for inspection.', flush=True)
            break
    if tick + 1 < args.polls:
        time.sleep(args.interval)
