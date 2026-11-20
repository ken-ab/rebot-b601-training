"""Build a portable complete-parameter B601 inference delivery after evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import numpy as np

from deployment_selection import select_deployment

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/yujiegao/桌面/zzk/rebot-b601-training')
KIT = REPO / 'openpi_b601'
MONITOR = KIT / 'runs/pi05_b601_lora/b601-final60-v1_monitor'
completion = json.loads((MONITOR / 'LONGRUN_COMPLETE.json').read_text())
assert completion['active_seconds'] >= 12.25 * 3600
best = json.loads((MONITOR / 'BEST.json').read_text())
action_selection = select_deployment(ROOT, MONITOR,
    KIT / 'prepared/assets/rebot-b601-train/norm_stats.json')
selected = action_selection['primary']
checkpoint = Path(selected['checkpoint'])
BUNDLE = ROOT / 'B601_PI05_DEPLOYMENT'
BUNDLE.mkdir(exist_ok=False)

def copy_link(source, destination):
    try:
        return os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)

for name in ('params', 'assets'):
    shutil.copytree(checkpoint / name, BUNDLE / 'checkpoint' / name, copy_function=copy_link)
models = {'primary': {**selected, 'relative_checkpoint': 'checkpoint',
          'relative_predictions': f"evaluation/candidates/{selected['label']}/predictions.npz"}}
for alternative in action_selection['alternatives']:
    label = alternative['label']
    for name in ('params', 'assets'):
        shutil.copytree(Path(alternative['checkpoint']) / name,
                       BUNDLE / 'alternatives' / label / 'checkpoint' / name, copy_function=copy_link)
    models[label] = {**alternative, 'relative_checkpoint': f'alternatives/{label}/checkpoint',
                    'relative_predictions': f'evaluation/candidates/{label}/predictions.npz'}
(BUNDLE / 'DEPLOYMENT_SELECTION.json').write_text(json.dumps(
    {**action_selection, 'delivered_models': models}, indent=2, ensure_ascii=False) + '\n')
code = BUNDLE / 'code'
code.mkdir()
for name in ('b601_data.py', 'train_pi05.py', 'b601_policy.py', 'serve_b601.py', 'evaluate_b601.py'):
    shutil.copy2(KIT / name, code / name)
shutil.copytree(ROOT / 'windows_client', BUNDLE / 'client',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(ROOT / 'WINDOWS_WSL2_DEPLOYMENT_CN.md', BUNDLE)
shutil.copy2(ROOT / 'PI05_INFERENCE_ARCHITECTURE_CN.md', BUNDLE)
shutil.copy2(ROOT / 'DEPLOYMENT_READINESS_CN.md', BUNDLE)
shutil.copy2(ROOT / 'PUBLICATION_PLAN_CN.md', BUNDLE)
tokenizer = Path('/data2/yujiegao/b601-pi05/cache/openpi/big_vision/paligemma_tokenizer.model')
assert tokenizer.is_file(), tokenizer
cache = BUNDLE / 'cache/openpi/big_vision'
cache.mkdir(parents=True)
shutil.copy2(tokenizer, cache / tokenizer.name)
metrics = {}
for name in ('base', 'best', 'last'):
    path = ROOT / 'evaluation' / name / 'METRICS.json'
    value = json.loads(path.read_text())
    assert value['full_protocol'] and value['evaluated_action_rows'] == 5118
    predictions = np.load(ROOT / 'evaluation' / name / 'predictions.npz')
    error = (predictions['prediction'].astype(np.float64) - predictions['target'])[predictions['valid_mask']]
    np.testing.assert_allclose(np.abs(error).mean(axis=0), value['per_axis_mae'], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.sqrt((error**2).mean(axis=0)), value['per_axis_rmse'], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.abs(error).mean(axis=0)[:6].mean(), value['mean_joint_1to6_mae'], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.abs(error).mean(axis=0)[6], value['gripper_mae'], rtol=2e-5, atol=2e-5)
    episode_mae, episode_rmse = [], []
    assert len(value['episodes']) == 10
    for episode in value['episodes']:
        rows = predictions['episode_source_frame'][:, 0] == episode['local_episode']
        assert (predictions['episode_source_frame'][rows, 1] == episode['source_episode']).all()
        errors = (predictions['prediction'][rows].astype(np.float64)-predictions['target'][rows])[predictions['valid_mask'][rows]]
        assert len(errors) == episode['evaluated_action_rows']
        mae, rmse = np.abs(errors).mean(axis=0), np.sqrt(np.square(errors).mean(axis=0))
        np.testing.assert_allclose(mae, episode['per_axis_mae'], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(rmse, episode['per_axis_rmse'], rtol=2e-5, atol=2e-5)
        episode_mae.append(mae)
        episode_rmse.append(rmse)
    np.testing.assert_allclose(np.mean(episode_mae, axis=0), value['macro_episode_per_axis_mae'], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.mean(episode_rmse, axis=0), value['macro_episode_per_axis_rmse'], rtol=2e-5, atol=2e-5)
    metrics[name] = value
evaluation = BUNDLE / 'evaluation'
evaluation.mkdir()
for name in ('base', 'best', 'last'):
    shutil.copytree(ROOT / 'evaluation' / name, evaluation / name, copy_function=copy_link)
for candidate in action_selection['candidates']:
    shutil.copytree(candidate['evaluation_directory'], evaluation / 'candidates' / candidate['label'],
                    copy_function=copy_link)
receipts = BUNDLE / 'receipts'
receipts.mkdir()
for path in MONITOR.glob('*.json*'):
    shutil.copy2(path, receipts / path.name)
for name in ('DATA_ACCEPTANCE.json', 'JAX_GPU_ACCEPTANCE.json', 'ISSUE_RESOLUTION_LOG.json', 'ENVIRONMENT.json',
             'TRAINING_SOURCE_LOCK.json', 'LR_SCHEDULE_CHECK.json', 'CONSOLE_TRAIN_METRICS.jsonl'):
    shutil.copy2(ROOT / name, receipts / name)
contract = json.loads((ROOT / 'RUN_CONTRACT.json').read_text())
contract.pop('default_config_sha256', None)
(receipts / 'RUN_CONTRACT.json').write_text(json.dumps(contract, indent=2, ensure_ascii=False) + '\n')
shutil.copytree(ROOT / 'source_versions', BUNDLE / 'executed_training_source_versions')
for name in ('LICENSE', 'LICENSE_GEMMA.txt'):
    shutil.copy2(KIT / '.openpi' / name, BUNDLE / name)
shutil.copy2(ROOT / 'NOTICE', BUNDLE / 'NOTICE')
for label, model in models.items():
    notice = BUNDLE / model['relative_checkpoint'] / 'B601_MODIFICATIONS.md'
    notice.write_text(
        '# Modified model parameters: B601 final-60\n\n'
        'The serialized weights in this checkpoint were produced by B601 fine-tuning '
        'of OpenPI pi05_base; they are not the unmodified base checkpoint. '
        f"Candidate: {model['label']}; optimizer updates: {model['optimizer_step']}. "
        'Training covers LoRA language/action branches and dense vision parameters. '
        'Retain the release NOTICE, LICENSE_GEMMA.txt and LICENSE. '
        'Gemma components and derivatives remain subject to the Gemma Terms of Use, '
        'including Section 3.2 use restrictions.\n')
for pattern in ('*-acceptance.json', '*-orbax-readback.json'):
    for path in (ROOT / 'receipts').glob(pattern):
        shutil.copy2(path, receipts / path.name)
for path in ROOT.glob('QUALITY_REVIEW_*.json'):
    shutil.copy2(path, receipts / path.name)
for name in ('ACTION_SELECTION_PROTOCOL.json', 'ACTION_SELECTION_RESULT.json'):
    shutil.copy2(ROOT / name, receipts / name)
shutil.copy2(ROOT / 'diagnostics/DETERMINISM_V2_ACCEPTANCE.json', receipts)
shutil.copy2(ROOT / 'INFERENCE_SOURCE_LOCK.json', receipts)
shutil.copy2(ROOT / 'diagnostics/CLIENT_PREPROCESS_EQUIVALENCE.json', receipts)
shutil.copy2(ROOT / 'diagnostics/CLIENT_TRANSPORT_ACCEPTANCE.json', receipts)
shutil.copy2(ROOT / 'diagnostics/CLIENT_TRANSPORT_ISSUE_RESOLUTION.json', receipts)
shutil.copy2(ROOT / 'diagnostics/ACTION_GUARD_ACCEPTANCE.json', receipts)
shutil.copy2(ROOT / 'deployment_selection.py', receipts)
for name in ('deployment-rehearsal10000-v2.json', 'action-selector-canonical-candidates-check.json'):
    shutil.copy2(ROOT / 'receipts' / name, receipts / name)
shutil.copy2(ROOT / 'receipts/websocket-rehearsal-v1.json', receipts)
shutil.copy2(ROOT / 'diagnostics/websocket_rehearsal_v1/REPORT.json', receipts / 'WEBSOCKET_REHEARSAL_REPORT.json')

# Include one real held-out observation and an already evaluated golden output.
os.environ['HF_LEROBOT_HOME'] = str(KIT / 'prepared/datasets')
sys.path.insert(0, str(KIT))
from b601_data import OVERHEAD, VAL_ID, WRIST
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(VAL_ID, video_backend='pyav')
sample = dataset[0]
golden = np.load(Path(selected['evaluation_directory']) / 'predictions.npz')['prediction'][0]
noise = np.random.default_rng(20261109).standard_normal((25, 32), dtype=np.float32)
np.savez_compressed(BUNDLE / 'example_observation.npz', overhead=np.asarray(sample[OVERHEAD]),
    wrist=np.asarray(sample[WRIST]), state=np.asarray(sample['observation.state']),
    noise=noise, expected_actions=golden)
shutil.copy2(ROOT / 'deployment_self_test.py', BUNDLE / 'self_test.py')
(BUNDLE / 'install_openpi.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 2; }
revision=215abfb217dbac7d5f1273282331b9b1866c0479
if [[ ! -d "$root/openpi/.git" ]]; then
  git init "$root/openpi"
  git -C "$root/openpi" -c http.version=HTTP/1.1 fetch --depth 1 https://github.com/Physical-Intelligence/openpi.git "$revision"
  git -C "$root/openpi" checkout --detach "$revision"
fi
[[ "$(git -C "$root/openpi" rev-parse HEAD)" == "$revision" ]]
git -C "$root/openpi" submodule update --init --recursive
cd "$root/openpi"
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --frozen
export OPENPI_DATA_HOME="$root/cache/openpi"
exec "$root/openpi/.venv/bin/python" "$root/self_test.py"
''')
(BUNDLE / 'serve.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
export OPENPI_DATA_HOME="$root/cache/openpi"
exec "$root/openpi/.venv/bin/python" "$root/code/serve_b601.py" --checkpoint "$root/checkpoint" "$@"
''')
for script_name, entry in [('serve_low_memory.sh', 'code/serve_b601.py'),
                           ('self_test_low_memory.sh', 'self_test.py')]:
    extra = ' --checkpoint "$root/checkpoint"' if script_name.startswith('serve') else ''
    (BUNDLE / script_name).write_text('''#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
export OPENPI_DATA_HOME="$root/cache/openpi"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_CLIENT_MEM_FRACTION
''' + 'exec "$root/openpi/.venv/bin/python" "$root/' + entry + '"' + extra + ' "$@"\n')
summary = {'training': completion, 'selected': selected, 'original_flow_best': best,
    'action_selection': action_selection, 'metrics': metrics,
    'openpi_revision': '215abfb217dbac7d5f1273282331b9b1866c0479',
    'deployment': 'Complete JAX params + training norm assets + B601 transforms + tokenizer; optimizer state stays on server',
    'robot_execution_tested': False, 'target_client_machine': 'WindowsRTX4070Laptop8188MiB; WSL2 available, localacceptancepending',
    'deployment_target': json.loads((ROOT/'DEPLOYMENT_TARGET.json').read_text())}
(BUNDLE / 'TRAINING_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
readme = f'''# B601 π₀.₅ 完整参数部署包

正式长跑活动墙钟 {completion['active_seconds']/3600:.3f} 小时，累计 {completion['optimizer_step']} 次更新。默认交付模型 {selected['label']} 完成 {selected['optimizer_step']} 次更新。原 flow-best 编号 {best['step_label']} 保留在训练记录中；真实七轴动作验证的选择规则、原始来源和替代模型见 DEPLOYMENT_SELECTION.json。完整验证使用 10 条 episode / 5,118 帧，逐轴指标见 TRAINING_SUMMARY.json 与 evaluation/。

这是完整 params、训练集归一化 assets、B601 数据映射、tokenizer 和自定义服务入口。不是只含 LoRA 叶子的 adapter 包；可恢复训练的优化器 checkpoint 仍保留在服务器。

在具备兼容 NVIDIA GPU 的 Linux / Ubuntu 22.04 上，安装 uv 后运行 `bash install_openpi.sh`。该步骤按固定 commit 安装依赖并用真实验证样本做离线推理检查，不连接或控制机械臂。环境首次安装需要网络；模型参数和 tokenizer 已包含。

随后可 `bash serve.sh --host 127.0.0.1 --port 8000`。远程客户端可使用 OpenPI 的 openpi-client websocket client。上游配置注册表没有 pi05_b601_lora，因此请使用这里的 serve_b601.py，而非只给上游 serve_policy.py 一个未注册的配置名。

默认 checkpoint 对应动作验证选择的 primary。若关节与夹爪误差存在取舍，其他非支配候选完整保存在 alternatives/<标签>/checkpoint；用 `self_test.py --model <标签>` 检查，并给 serve_b601.py 指定该 checkpoint。选择没有混合不同单位的综合分数，也不代表真机任务最优。

输入键：`observation.images.overhead`、`observation.images.wrist`（RGB，HWC uint8 或 CHW [0,1] float）、`observation.state`（七轴从臂电机侧实测反馈）、`prompt`（任务指令）。输出 `actions` 为 25×7 的 absolute_adapter_input，与示教相同坐标，由原 B601 适配器完成方向与夹爪比例转换；服务端不重复转换。名义采样率 25 Hz，窗口约一秒。

离线动作误差反映与示教的一致性，不能当作真机抓放成功率。服务器没有连接现场机械臂，尚未做闭环机器人测试。用户目标为WindowsRTX4070Laptop8GB，可使用WSL2；原生WindowsJAXGPU不支持，低显存WSL入口及Windows轻量客户端见WINDOWS_WSL2_DEPLOYMENT_CN.md。该目标机尚未实际验收；另有ACT轻量路线独立比较，未替换本包的π₀.₅参数。

所有交付文件的 SHA-256 见 MANIFEST.json；`self_test.py` 首先逐文件核验大小和 SHA-256，再检查真实样本推理为有限的 25×7 动作，并强制核对服务器参考动作（默认 atol=1e-4、rtol=1e-5）。跨硬件差异超出阈值时应先分析原因，不自动放宽阈值。

推理使用固定 XLA 配置 `--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true`。b601_policy 会在导入 JAX 前配置并检查这两个 flags；自写客户端/服务程序请先导入 b601_policy，或在首次导入 JAX 前设置同样环境变量。默认自动选核曾产生跨进程约0.126动作单位差异，固定配置后已在本服务器两个独立进程验证一致；所有最终候选动作指标及 golden 使用同一配置。训练的选核配置没有改变。
'''
(BUNDLE / 'README_CN.md').write_text(readme)
manifest = {}
for path in BUNDLE.rglob('*'):
    if path.is_file():
        value = hashlib.sha256()
        with path.open('rb') as source:
            while block := source.read(8 * 1024 * 1024):
                value.update(block)
        manifest[str(path.relative_to(BUNDLE))] = {'bytes': path.stat().st_size, 'sha256': value.hexdigest()}
(BUNDLE / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
archive = ROOT / 'B601_PI05_DEPLOYMENT.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=1) as output:
    output.add(BUNDLE, arcname=BUNDLE.name)
archive_digest = hashlib.sha256()
with archive.open('rb') as source:
    while block := source.read(8 * 1024 * 1024):
        archive_digest.update(block)
(ROOT / 'B601_PI05_DEPLOYMENT.sha256').write_text(archive_digest.hexdigest() + '  ' + archive.name + '\n')
client_archive = ROOT / 'B601_PI05_WINDOWS_CLIENT.zip'
client_manifest = {relative: value for relative, value in manifest.items()
    if relative.startswith('client/') or relative in ('example_observation.npz',
        'PI05_INFERENCE_ARCHITECTURE_CN.md', 'WINDOWS_WSL2_DEPLOYMENT_CN.md',
        'DEPLOYMENT_READINESS_CN.md')}
with zipfile.ZipFile(client_archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
    for relative in client_manifest:
        output.write(BUNDLE / relative, arcname='B601_PI05_WINDOWS_CLIENT/' + relative)
    output.writestr('B601_PI05_WINDOWS_CLIENT/MANIFEST.json',
                    json.dumps(client_manifest, indent=2) + '\n')
client_digest = hashlib.sha256(client_archive.read_bytes()).hexdigest()
(ROOT / 'B601_PI05_WINDOWS_CLIENT.sha256').write_text(client_digest + '  ' + client_archive.name + '\n')
receipt = {'bundle': str(BUNDLE), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
           'archive_sha256': archive_digest.hexdigest(), 'manifest_files': len(manifest),
           'all_evaluation_metrics_recomputed': True, 'selected': selected,
           'delivered_model_labels': list(models),
           'windows_client_archive': str(client_archive), 'windows_client_bytes': client_archive.stat().st_size,
           'windows_client_sha256': client_digest, 'windows_client_manifest_files': len(client_manifest)}
(ROOT / 'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
