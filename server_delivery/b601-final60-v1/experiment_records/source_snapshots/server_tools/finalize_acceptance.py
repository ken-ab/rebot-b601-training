"""Wait for the real main pipeline, then verify its final deliverables."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / 'rebot-b601-training/openpi_b601'
PYTHON = str(KIT / '.openpi/.venv/bin/python')
BUNDLE = ROOT / 'B601_PI05_DEPLOYMENT'
GPU = 'GPU-a2df82e1-ad48-ea47-fa69-5ee02ec32b0e'


def read(path):
    return json.loads(Path(path).read_text())


def status(stage, **extra):
    row = {'at': datetime.datetime.now().astimezone().isoformat(), 'stage': stage, **extra}
    temporary = ROOT / 'FINAL_ACCEPTANCE_STATUS.tmp'
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(ROOT / 'FINAL_ACCEPTANCE_STATUS.json')
    print(json.dumps(row), flush=True)


def run(label, command, gpu=False, platform_allocator=False):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU if gpu else '',
                       JAX_PLATFORMS='cuda' if gpu else 'cpu',
                       OPENPI_DATA_HOME=str(BUNDLE / 'cache/openpi'),
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    for name in ('XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                 'XLA_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_PREALLOCATE'):
        environment.pop(name, None)
    if gpu:
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        if platform_allocator:
            environment['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        else:
            environment['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    status(label, command=command)
    subprocess.run([sys.executable, str(ROOT / 'run_stage.py'), label, *command],
                   env=environment, check=True)
    receipt = read(ROOT / 'receipts' / (label + '.json'))
    assert receipt['status'] == 'passed'
    return receipt


def write_report(audit):
    act_root = ROOT.parent / 'B601_ACT_20261109'
    act = read(act_root / 'INDEPENDENT_DELIVERY_AUDIT.json')
    assert act['status'] == 'passed'
    pi_receipt = read(ROOT / 'DELIVERY_RECEIPT.json')
    act_receipt = read(act_root / 'DELIVERY_RECEIPT.json')
    parameters = read(BUNDLE / 'receipts/RUN_PARAMETERS.json')
    completion, selected = audit['completion'], audit['selected']
    rows = []
    for label, metrics in audit['evaluations_recomputed_float64'].items():
        rows.append(f"| π₀.₅ {label} | {metrics['mean_joint_1to6_mae']:.6f} | {metrics['gripper_mae']:.6f} |")
    act_metrics = act['selected_metrics']
    rows.append(f"| ACT selected15k | {act_metrics['mean_joint_1to6_mae']:.6f} | {act_metrics['gripper_mae']:.6f} |")
    memory = audit['final_low_memory']
    act_gpu = act['selftests']['cuda']['self_test']
    text = f'''# B601 微调、评估及部署最终交接

实际验收时间：{audit['at']}。π₀.₅ 正式长跑完成 {completion['active_seconds']/3600:.4f} 小时、{completion['optimizer_step']} 次优化器更新，最后 checkpoint 编号 {completion['step_label']}。计时从第一次成功训练调度结束后开始，排除环境准备和初始编译，包含正常保存与验证。最终完成标记、checkpoint绑定计数、CPU参数/优化器读回、完整动作评估、部署包自检、实际WebSocket协议与压缩包逐文件校验均已通过。

## 数据与训练方法

final-60 原固定50训练/10验证episode划分，训练28,729帧、验证5,118帧，两路RGB、七轴状态、25步动作窗口、名义25Hz。归一化统计只来自训练集，验证使用相同统计。完整动作评估包含211个窗口，去掉padding后每条原验证动作计一次；标签、mask及源episode/frame与先前固定协议逐元素一致。

服务器为双Xeon Gold6530、Linux可见系统内存总量约503GiB、四张RTX A4500 20GB；π₀.₅使用物理GPU2/3、FSDP2，ACT独立使用物理GPU1。训练与评估入口、版本、哈希、smoke和恢复探针保存在原记录及部署包receipts中。

π₀.₅使用OpenPI固定commit215abfb217dbac7d5f1273282331b9b1866c0479，官方pi05_base，JAX0.5.3，LoRA语言与动作分支，原生过滤器同时训练视觉部分，因此交付完整参数。global batch={parameters['global_batch_size']}，workers={parameters['arguments']['workers']}，seed42，500步warmup、峰值lr2.5e-5、10k后衰减至2.5e-6。训练的200k是步数上限，实际按12.25小时门槛后的完整保存点结束。AdamW与模型精确配置见receipts/RUN_PARAMETERS.json。原始flow-best保持独立；最终部署按已锁定的最多四个候选在六关节MAE/夹爪MAE上的Pareto规则选择，默认交付{selected['label']}（{selected['optimizer_step']}次更新）。

两组20步π₀.₅短跑均通过有限loss/grad和参数读回：batch2首/末loss0.0350/0.0312；batch4为0.0358/0.0477。短跑是工程验收，不能当作精度提升。正式训练的控制台指标按100步区间均值记录且四位小数；本次W&B重绑定问题的原因、补救与原执行源码保存在记录中，不能把该日志称为每一步全精度loss。固定验证使用80个观察的flow代理指标，真实动作比较使用全部5,118行。

## 完整验证结果

| 模型/评估标签 | 六关节平均MAE | 夹爪MAE |
| --- | ---: | ---: |
{chr(10).join(rows)}

单位为原始absolute_adapter_input，两个指标分开报告。逐轴RMSE、逐episode和宏平均、原预测及mask均保存，可独立复算。验证集参与选模，不能当作独立测试；离线示教动作一致性不是机械臂抓放成功率。base是单独重新运行的无LoRA纯基座对照，初始随机LoRA flow值不作为纯base替代。

## 针对用户Windows笔记本的部署

目标为联想Y7000P IRX9（83DG），RTX4070Laptop8188MiB，RAM32GB或更多，可使用WSL2。LoRA不限定操作系统；JAX原生Windows GPU的兼容限制与LoRA本身无关。

用户已澄清，笔记本只需部署已微调模型进行推理，不需要在本机训练；主模型保留π₀.₅。两种部署均使用本次微调后的同一模型：服务器GPU推理＋Windows采集图像/反馈并经原B601适配器通过USB执行动作；或WSL2在4070Laptop上本地运行π₀.₅，再由Windows控制USB。USB连接位置不要求模型必须同机运行。建议先完成服务器推理链路与本机客户端联调，再依据WSL2实机验收决定是否脱离服务器；目标机到服务器的网络和现场闭环尚未测试。

ACT仅保留为轻量独立对照，不默认替换π₀.₅。它不按语言prompt切换任务，不能仅凭体积小宣称抓放更好。ACT正式完成20k更新，原normalized L1规则选中15k；CPU与GPU整套5,118行读回通过。服务器单样本暖机中位{act_gpu['warm_median_seconds']*1000:.2f}ms，PyTorch峰值allocated{act_gpu['cuda_peak_allocated_mib']:.2f}MiB，不含CUDA上下文/驱动开销，不是4070本机性能承诺。

π₀.₅保留WSL2 Ubuntu低显存路线。当前最终交付参数在本服务器platform分配器10次严格golden检查通过，采样进程峰值{memory['sampled_peak_process_residency_mib']}MiB、主机内存峰值{memory['host_peak_rss_mib']:.2f}MiB。采样可能漏短峰值，A4500/Linux不能代替4070/WSL实机验证。32GB主机可先以16GB WSL预算验收。官方JAX平台表将原生Windows NVIDIA GPU标为不支持、WSL2 GPU为实验性，见https://docs.jax.dev/en/latest/installation.html#supported-platforms。

输入为两路RGB、七轴原始电机侧反馈；输出25×7 absolute_adapter_input，由原B601适配器转换方向与夹爪比例，避免重复转换。预测25步不等于必须开环执行25步，现场执行/重观测节奏需在原控制循环中验证。服务器未连接或驱动现场机械臂。

## 参数及文件

- π₀.₅最终训练checkpoint：`{completion['checkpoint']}`。
- π₀.₅默认部署参数源：`{selected['checkpoint']}`。
- π₀.₅包：`{pi_receipt['archive']}`，SHA256 `{pi_receipt['archive_sha256']}`。
- ACT包：`{act_receipt['archive']}`，SHA256 `{act_receipt['archive_sha256']}`。
- 完整独立核验：`INDEPENDENT_FINAL_AUDIT.json`、ACT目录`INDEPENDENT_DELIVERY_AUDIT.json`。
- 安装入口：两个包各自`README_CN.md`；π₀.₅另有`WINDOWS_WSL2_DEPLOYMENT_CN.md`、`self_test_low_memory.sh`及Windows轻量客户端。

服务器保留优化器checkpoint。π₀.₅已在独立探针验证优化器/模型/更新计数恢复，最终checkpoint的全部优化器数组和计数也已读回；上游DataLoader位置不恢复，不承诺逐步完全一致续训。ACT保存了优化器/RNG材料，但尚无已验收的精确resume入口。Windows4070本机自检与现场机械臂闭环验收仍由目标机执行，这里不作已经完成的声明。
'''
    (ROOT / 'FINAL_REPORT_CN.md').write_text(text)


try:
    status('waiting_actual_main_pipeline_completion')
    while True:
        pipeline = read(ROOT / 'PIPELINE_STATUS.json')
        if pipeline['stage'] == 'failed_needs_agent_analysis':
            raise RuntimeError(pipeline)
        if pipeline['stage'] == 'complete':
            break
        state = subprocess.check_output(['systemctl', '--user', 'show',
            'b601-pipeline-20261109', '--property=ActiveState', '--value'], text=True).strip()
        if state not in ('active', 'activating'):
            # The completed unit may exit between the two observations.
            # Re-read its terminal receipt before treating an inactive unit as failure.
            pipeline = read(ROOT / 'PIPELINE_STATUS.json')
            if pipeline['stage'] == 'complete':
                break
            raise RuntimeError('Main pipeline not live and not complete: ' + state)
        time.sleep(15)
    summary = read(BUNDLE / 'TRAINING_SUMMARY.json')
    assert summary['training']['active_seconds'] >= 44100
    selection = read(BUNDLE / 'DEPLOYMENT_SELECTION.json')
    checkpoints = {summary['training']['optimizer_step']: Path(summary['training']['checkpoint'])}
    for model in selection['delivered_models'].values():
        checkpoints[model['optimizer_step']] = Path(model['checkpoint'])
    for step, checkpoint in checkpoints.items():
        run('final-params-' + str(step) + '-readback-v1',
            [PYTHON, str(ROOT / 'verify_checkpoint.py'), str(checkpoint), '--output',
             str(ROOT / 'diagnostics' / ('final_params_' + str(step) + '.json'))])
    run('final-optimizer-readback-v1',
        [PYTHON, str(ROOT / 'verify_optimizer_checkpoint.py'), summary['training']['checkpoint'],
         '--output', str(ROOT / 'diagnostics/final_optimizer.json')])
    for label in selection['delivered_models']:
        if label != 'primary':
            run('deployment-inference-' + label + '-readback-v1',
                [PYTHON, str(BUNDLE / 'self_test.py'), '--model', label], gpu=True)
    run('websocket-final-v1', [PYTHON, str(ROOT / 'verify_websocket_deployment.py'),
        '--bundle', str(BUNDLE), '--output', str(ROOT / 'diagnostics/websocket_final_v1'),
        '--scope', 'final'], gpu=True)
    run('final-low-memory-profile-v1', [PYTHON, str(ROOT / 'profile_inference_memory.py'),
        '--bundle', str(BUNDLE), '--output', str(ROOT / 'diagnostics/inference_platform_final_v1'),
        '--scope', 'final'], gpu=True, platform_allocator=True)
    run('independent-final-delivery-audit-v1', [PYTHON, str(ROOT / 'audit_final_delivery.py')])
    audit = read(ROOT / 'INDEPENDENT_FINAL_AUDIT.json')
    assert audit['status'] == 'passed'
    write_report(audit)
    status('complete_pending_root_completion_audit', audit=str(ROOT / 'INDEPENDENT_FINAL_AUDIT.json'),
           report=str(ROOT / 'FINAL_REPORT_CN.md'), completion=audit['completion'], selected=audit['selected'])
except Exception as exc:
    status('failed_needs_agent_analysis', error=repr(exc))
    raise
