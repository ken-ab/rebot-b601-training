"""Sequential diagnostic inference only on separately assigned idle GPU1."""
import os
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--version', default='v1')
args = parser.parse_args()
if args.version not in ('v1', 'v2'):
    parser.error('Choose a preserved diagnostic version')
python = '/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/.venv/bin/python'
for layout in ('native', 'portable'):
    command = [python, str(root/'probe_inference_precision.py'), '--layout', layout,
        '--output', str(root/'diagnostics'/('precision_'+layout+'_'+args.version))]
    subprocess.run([sys.executable, str(root/'run_stage.py'), 'precision-'+layout+'-'+args.version, *command], check=True)
