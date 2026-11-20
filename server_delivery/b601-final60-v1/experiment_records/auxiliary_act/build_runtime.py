"""Export an actual ACT checkpoint without training-framework dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/.venv/lib/python3.11/site-packages/lerobot/common/policies/act/modeling_act.py')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--scope', choices=('interim', 'final'), default='interim')
args = parser.parse_args()
training_root = args.checkpoint.parent
if args.scope == 'final':
    completed = json.loads((training_root/'COMPLETE.json').read_text())
    selected = json.loads((training_root/'BEST.json').read_text())
    assert completed['status']=='complete' and completed['step']>=20000
    assert args.checkpoint.resolve()==Path(selected['checkpoint']).resolve()
args.bundle.mkdir(parents=True, exist_ok=False)
source = SOURCE.read_text()
license_header = source[:source.index('"""Action Chunking Transformer Policy')]
imports = '''from __future__ import annotations
import math
from itertools import chain
from typing import Callable
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
'''
(args.bundle/'act_network.py').write_text(license_header+imports+'\n'+source[source.index('class ACT(nn.Module):'):])
shutil.copy2(ROOT/'act_runtime.py', args.bundle/'act_runtime.py')
shutil.copy2(ROOT/'runtime_self_test.py', args.bundle/'self_test.py')
shutil.copy2(args.checkpoint/'policy/config.json', args.bundle/'config.json')
state = safetensors.torch.load_file(args.checkpoint/'policy/model.safetensors', device='cpu')
actor = {key.removeprefix('model.'): value for key,value in state.items() if key.startswith('model.')}
normalizers = {key:value for key,value in state.items() if key.startswith(('normalize_inputs.', 'unnormalize_outputs.'))}
assert len(normalizers) == 8
torch.save({'model': actor, 'normalizers': normalizers}, args.bundle/'weights.pt')
raw = np.load('/data2/yujiegao/b601-pi05/validation/deployment_rehearsal10000_v2/example_observation.npz')
reference = np.load(args.checkpoint/'evaluation/predictions.npz')['prediction'][0]
np.savez_compressed(args.bundle/'example_observation.npz', overhead=raw['overhead'], wrist=raw['wrist'],
    state=raw['state'], expected_actions=reference)
shutil.copytree(args.checkpoint/'evaluation', args.bundle/'evaluation')
shutil.copy2('/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/.openpi/LICENSE', args.bundle/'LICENSE')
provenance = {'scope': args.scope, 'checkpoint': str(args.checkpoint), 'model_family': 'ACT',
    'source_repository': 'https://github.com/huggingface/lerobot', 'source_revision': '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5',
    'upstream_model_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'vendored_components': 'Unchanged ACTnn.Module and encoder/decoder/positionhelpers; excluded trainingPolicy/Hub/data dependencies',
    'state_tensor_elements': sum(value.numel() for value in actor.values()),
    'normalizers': 'Exact savedfloat32buffers; train-only stats/ImageNet constants',
    'zero_latent_inference': True, 'language_conditioned': False,
    'cross_backend_validation_tolerance': {'atol': 1e-3, 'rtol': 1e-4},
    'scope_limit': 'No claim of Windows runtime execution or physicalrobot success'}
(args.bundle/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
(args.bundle/'requirements_runtime.txt').write_text('numpy==1.26.4\neinops==0.8.1\nPillow==11.2.1\n')
shutil.copy2(training_root/'RUN_PARAMETERS.json', args.bundle/'TRAINING_PARAMETERS.json')
metrics = json.loads((args.checkpoint/'evaluation/METRICS.json').read_text())
summary = {'scope': args.scope, 'selected_checkpoint': str(args.checkpoint), 'metrics': metrics,
    'source_dataset': {'train_episodes': 50, 'train_frames': 28729, 'validation_episodes': 10, 'validation_frames': 5118},
    'deployment_target': 'NativeWindowsCUDA,RTX4070Laptop8188MiB; actualWindowsvalidationpending',
    'test_evidence_limit': 'ServerLinuxCPU/GPU checks are not Windows execution or physicalrobot grasp success'}
for name in ('BEST.json','LATEST.json','COMPLETE.json'):
    if (training_root/name).exists():
        summary[name.removesuffix('.json').lower()] = json.loads((training_root/name).read_text())
(args.bundle/'TRAINING_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
readme = f'''# B601 ACT 原生Windows推理包

本包范围：{args.scope}。参数来源 `{args.checkpoint.name}`，固定任务为蓝色餐巾纸抓放。完整验证包含10条episode、5118个有效动作行；六关节平均MAE={metrics['mean_joint_1to6_mae']:.6f}，夹爪MAE={metrics['gripper_mae']:.6f}，单位为原始absolute_adapter_input。验证集参与选模，不能当独立测试或真机成功率。

模型使用两路RGB图像、七轴实测反馈，预测25×7动作窗口；ACT不按语言prompt切换任务。网络与精确float32归一化缓冲已包含，加载不下载骨干权重，不需要JAX、LeRobot、Transformers。来源/许可证见PROVENANCE.json、LICENSE和act_network.py原始头部。

## Windows安装与真实样本自检

推荐Python3.11；在本包目录PowerShell执行以下命令。核心CUDA包单独从PyTorch官方CUDA12.6索引安装，避免误装CPU版。

```powershell
py -3.11 -m venv .venv-act
.\\.venv-act\\Scripts\\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\\.venv-act\\Scripts\\python.exe -m pip install -r requirements_runtime.txt
.\\.venv-act\\Scripts\\python.exe self_test.py --device cuda
```

脚本核验文件SHA256，用真实样本推理并对比服务器参考输出，报告显存与延迟；不连接或驱动机械臂。若默认在CPU测试，可传`--device cpu`。本包已在服务器验证数学/参数读取一致性；用户的Windows4070实机仍需执行上述验收。

## 接入机械臂程序

```python
from act_runtime import B601ACT
policy = B601ACT(device="cuda")
actions = policy.infer({{
    "observation.images.overhead": overhead_rgb,
    "observation.images.wrist": wrist_rgb,
    "observation.state": measured_seven_joints,
}})["actions"]
```

图像为RGB HWC uint8，或CHW/HWC的[0,1]浮点数组；运行时使用与训练缓存相同的224letterbox。七轴顺序是肩转、肩抬、肘、腕屈、腕偏、腕转、夹爪，准确键名见运行时metadata。state是未归一化的电机侧反馈，输出是示教的absolute_adapter_input，交给原B601适配器完成方向和夹爪比例转换，不重复转换。

25是预测窗口，实际执行多少步后获取新图像、重新预测由原控制循环决定；不能把模型窗口长度当作已验证的25Hz闭环性能。固定任务ACT不需要LoRA；另一个π₀.₅长跑及其WSL2低显存方案独立保留，最终比较应同时看动作误差、显存和延迟。

训练配置在TRAINING_PARAMETERS.json，逐轴/逐episode指标和可复算预测在evaluation/。文件SHA256清单为MANIFEST.json。
'''
(args.bundle/'README_CN.md').write_text(readme)
manifest = {}
for path in args.bundle.rglob('*'):
    if path.is_file():
        manifest[str(path.relative_to(args.bundle))] = {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(args.bundle/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'bundle': str(args.bundle), 'checkpoint': str(args.checkpoint),
                  'scope': args.scope, 'files': len(manifest)}), flush=True)
