set -eu
cd /data/b601-recap-round1/evo
export UV_CACHE_DIR=/data/b601-recap-round1/evo/sources/uv-cache
/root/.local/bin/uv pip install --python venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps transformers==4.57.1 huggingface-hub==0.35.3 tokenizers==0.22.1 accelerate==1.10.1 deepdiff==8.6.1 orderly-set==5.5.0 gymnasium==1.2.1 farama-notifications==0.0.4 safetensors==0.6.2
PYTHONPATH="$PWD/sources/Evo-RL/src" venv/bin/python - <<'PY'
import torch, transformers
from lerobot.values.pistar06.modeling_pistar06 import Pistar06Policy
print('evo module',Pistar06Policy.__module__, 'torch',torch.__version__,torch.cuda.is_available(),'transformers',transformers.__version__)
PY
