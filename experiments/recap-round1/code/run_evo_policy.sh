#!/usr/bin/env bash
set -euo pipefail
work=${B601_EVO_ROOT:-/data/b601-recap-round1/evo}
bundle=/root/b601-deploy/B601_PI05_DEPLOYMENT
cd "$work"
umask 077
exec 9>policy-run.lock
flock -n 9
deadline=$(python3 -c 'import json;print(json.load(open("contract.json"))["budget_end_epoch"])')
while ! test -s value/complete; do
    if (( $(date +%s) >= deadline )); then exit 1; fi
    sleep 15
done
python="$bundle/openpi/.venv/bin/python"
export OPENPI_DATA_HOME="$bundle/cache/openpi" HF_HOME=/root/b601-deploy/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.94
export XLA_FLAGS=--xla_gpu_autotune_level=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if ! test -s label_summary.json; then "$python" code/value.py evo-label --output "$work"; fi
if ! test -s policy/result.json; then
    "$python" code/policy.py --root "$work" --inputs-only --until-epoch "$deadline"
    "$python" -u code/policy.py --root "$work" --steps 3000 --until-epoch "$deadline" --closeout-seconds 300
fi
if ! test -s policy/parent-action-readback.json; then "$python" code/check_policy.py --root "$work" --parent; fi
if ! test -s policy/round1-action-readback.json; then "$python" code/check_policy.py --root "$work"; fi
date --iso-8601=seconds > policy/complete
