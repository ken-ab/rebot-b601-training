"""Re-evaluate the two fixed interim candidates under one numerical protocol."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
kit = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601')
python = str(kit / '.openpi/.venv/bin/python')
sources = [('checkpoint10000', Path('/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1/10000')),
           ('interim5000', Path(json.loads((root / 'early_evaluation/best5000/METRICS.json').read_text())['checkpoint']))]
for label, checkpoint in sources:
    command = [python, str(kit / 'evaluate_b601.py'), '--checkpoint', str(checkpoint),
               '--output', str(root / 'evaluation' / label)]
    (root / 'evaluation').mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(root / 'run_stage.py'), 'repro-candidate-'+label+'-v1', *command], check=True)
subprocess.run([sys.executable, str(root / 'run_stage.py'), 'deployment-rehearsal10000-v2',
    python, str(root / 'rehearse_deployment.py'), '--version', 'v2'], check=True)
