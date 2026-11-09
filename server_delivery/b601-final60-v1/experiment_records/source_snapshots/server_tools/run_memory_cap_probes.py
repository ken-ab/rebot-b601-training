"""Separate inference-only BFC pool caps, not physical8GB GPU emulation."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
python = '/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/.venv/bin/python'
for pool_mib in (7424, 7168):
    environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       XLA_PYTHON_CLIENT_MEM_FRACTION=str(pool_mib / 20470))
    command = [python, str(root / 'profile_inference_memory.py'), '--output',
               str(root / 'diagnostics' / f'inference_pool_{pool_mib}_v1')]
    subprocess.run([sys.executable, str(root / 'run_stage.py'), f'inference-pool-{pool_mib}-v1',
                    *command], env=environment, check=True)
