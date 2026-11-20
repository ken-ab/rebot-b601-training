"""Record sampled GPU memory and utilization; memory includes any JAX pools."""
import csv
from pathlib import Path
import subprocess
import time

path = Path(__file__).resolve().parent / 'GPU_SAMPLES.csv'
with path.open('a', buffering=1) as output:
    writer = csv.writer(output)
    if path.stat().st_size == 0:
        writer.writerow(['unix_time', 'gpu_index', 'memory_used_MiB', 'memory_free_MiB', 'gpu_utilization_percent',
                         'temperature_C', 'power_W'])
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                row = [part.strip() for part in line.split(',')]
                writer.writerow([time.time(), *row])
        time.sleep(1)
