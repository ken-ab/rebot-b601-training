set -euo pipefail
cd /data/b601-recap-round1/evo
umask 077
exec 9>value.lock
flock -n 9
export PYTHONPATH="$PWD/sources/Evo-RL/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
while [ ! -s inputs/stacking/model.safetensors ]; do sleep 3; done
if [ ! -s value/probe.ok ]; then
   venv/bin/python -u code/evo_value.py probe --batch 4
   date --iso-8601=seconds > value/probe.ok
fi
if [ ! -s value/metrics.json ]; then venv/bin/python -u code/evo_value.py train --batch 4 --eval-batch 12; fi
if [ ! -s value/predictions/complete.json ]; then venv/bin/python -u code/evo_value.py score --eval-batch 16; fi
date --iso-8601=seconds > value/complete
