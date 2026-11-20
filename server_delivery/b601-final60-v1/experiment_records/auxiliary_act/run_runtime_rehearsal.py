"""CPU-only independent runtime acceptance while GPU1 ACT training continues."""
import os
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
logroot = root.parent/'B601_LONGRUN_20261109'
python = '/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/.venv/bin/python'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--version', choices=('v1', 'v2'), default='v1')
args = parser.parse_args()
bundle = Path('/data2/yujiegao/b601-act-final60/validation/runtime_rehearsal_'+args.version)
for label, command in [
    ('act-runtime-selftest-rehearsal-'+args.version, [python, str(bundle/'self_test.py'), '--device', 'cpu']),
    ('act-runtime-full-rehearsal-'+args.version, [python, str(root/'verify_runtime_full.py'), '--bundle', str(bundle),
        '--device', 'cpu', '--output', str(root/'diagnostics'/('runtime_full_rehearsal_'+args.version))])]:
    subprocess.run([sys.executable, str(logroot/'run_stage.py'), label, *command], check=True)
