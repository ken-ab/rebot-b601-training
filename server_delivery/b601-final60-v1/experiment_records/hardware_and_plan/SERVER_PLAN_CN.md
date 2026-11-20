# B601 π₀.₅ JAX LoRA：本服务器适配评估与执行方案

评估日期：2026-11-09，北京时间。当前阶段是执行前评估；未安装训练环境、未完整下载数据或模型、未启动训练。不能据此声称已有有限 loss 或可恢复 checkpoint。

结论：硬件具备采用 JAX FSDP + LoRA 微调的条件，建议先用两张 RTX A4500 验收。原命令目前不能直接完成任务：缺少 uv，子脚本的 GitHub 身份需要隔离，动作序列列名不匹配，而且 B601 入口没有暴露 FSDP 参数。修正这些问题后，按“输入检查 → 双卡 20 步 → 正式 batch 的 20 步 → 正式训练”逐项验收。显存峰值、NCCL 通信和依赖完整安装仍需运行验证，不能保证静态评估等同于训练成功。

## 已确认的仓库、数据与硬件

- 已用隔离的 `ken-ab` 身份克隆到 `/home/yujiegao/桌面/zzk/rebot-b601-training`，main 为 `bd7f604a2fadf3ddaf6617f6a25e474d3ceda2bd`，与远端 main 一致。训练源码保持未修改，没有提交或推送。
- `final-60` Release 和附件均可访问。实际发出 1,024 字节 Range 请求，返回 HTTP 206，gzip 头为 `1f8b`。这证明附件读取权限；尚未下载完整归档，所以不代表本机 SHA-256 校验已通过。
- 附件大小 601,619,657 字节，约 573.7 MiB；仓库和 Release 一致声明 SHA-256 为 `cf8e129b5e1bbffe32bcdf85b90d35696ef08024a5a858c744738262bf0ab36f`。
- `episodes.csv` 与 `splits.json` 重新汇总一致：训练 50 条 / 28,729 帧，验证 10 条 / 5,118 帧。保留原划分，不按帧重新随机拆分。
- 2 × Xeon Gold 6530，共 64 核 / 128 线程；系统可见约 503 GiB 内存，当前可用约 487 GiB。
- 4 × RTX A4500，每张总显存 20,470 MiB，约 20 GiB；驱动 580.76.05，计算能力 8.6。检查时无计算训练任务，GPU0 有桌面/远程桌面进程。
- GPU0 可用约 19.18 GiB，GPU1/2/3 各约 19.56 GiB。GPU2/3 同属 NUMA 1，拓扑为 NODE；没有检测到活动 NVLink 连接。这些卡的空闲状态没有被锁定或预订，启动前必须重查。
- 根分区可用约 2.1 TiB，`/data2` 约 3.3 TiB，`/data3` 约 3.2 TiB。缓存计划放用户有写权限的 `/data2/yujiegao/b601-pi05/`，checkpoint 保留仓库原输出路径。
- `git`、`gh`、`flock`、`tar`、`sha256sum`、`ffmpeg` 可用；当前 PATH 找不到 `uv`。系统 Python 是 3.10.12，但用户目录已有 `/home/yujiegao/.local/bin/python3.11`，版本 3.11.15。
- 已访问官方模型对象清单：`pi05_base/params/` 的 20 个对象合计 12,441,721,931 字节，约 11.59 GiB；未下载权重。对象清单可读不等于所有大文件传输已验证。

精确快照、返回码、源文件 SHA-256 与模型对象列表见同目录 `PREFLIGHT_EVIDENCE.json`。

## 原提示词各步骤的适用性

| 步骤 | 本机判断 | 执行调整 |
| --- | --- | --- |
| 使用已有 GitHub 身份克隆私库、访问 Release | 已确认可行 | 使用 `github-zzk` / 独立 gh 配置，保持默认 gyj612612 账号 |
| 原样运行 smoke | 目前无法进入准备阶段 | 已实测入口 `check` 以返回码 2 停在缺少 uv |
| 下载、校验、v3 → v2.1 转换 | 代码路径、权限与主机资源具备条件 | 依赖安装后完成全归档 SHA-256，再检查真实样本 |
| 单卡 batch 1 / 20 步 | 可以作为受限单卡试验，成功与否未知 | 单卡 20 GiB 低于官方 LoRA 显存参考，优先双卡 FSDP |
| 不限 GPU 直接运行 smoke | 不适用 | 原版全局 batch 1 不能被四张可见卡整除，会触发上游检查 |
| 原样正式训练 batch 8 / 10,000 步 | 不建议直接启动 | 先明确分配双卡并启用 FSDP，再从全局 batch 4 验收 |
| 仅降低 batch size | 可降低激活显存，但不能保证解决问题 | 必要时同时对参数/优化器状态分片；不能把主机内存当显存 |
| 短跑完成后自动得到验证性能 | 原代码不提供此行为 | 原代码只读验证样本，不计算完整验证集动作指标 |

实际已复现的入口报错为：

```text
Install uv in the server environment first (https://docs.astral.sh/uv/getting-started/installation/).
```

以下问题来自锁定版本的静态源码对照，尚未在完整训练环境复现异常栈：

1. **动作列名。** 转换器保留源数据列 `action`，`B601Inputs` 也读取 `action`；但 `DataConfig` 默认 `action_sequence_keys=("actions",)`。上游加载器在 B601 transform 之前，先按这些列名生成动作窗口；锁定版 LeRobot 随后直接索引这些列。因此要显式设为 `("action",)`，否则将请求不存在的 `actions` 列。
2. **FSDP 没有接入命令行。** 上游 `TrainConfig.fsdp_devices` 默认 1。B601 argparse 不接受 `--fsdp-devices`，也没有传值给 TrainConfig。仅设置多卡可见会使用数据并行，不能获得所需的参数分片。
3. **子脚本身份。** `run_pi05.sh` 内部直接调用 `gh release download`；只用项目 wrapper 克隆，并不会让后续普通 `gh` 自动使用同一账号。启动器对其子进程设置独立 GH_CONFIG_DIR，避免私有 Release 下载使用错误账号。

同目录 `b601_server_adaptation.patch` 仅修正动作窗口列名、暴露 FSDP 参数、增加相关参数检查和设备日志。补丁已通过 `git apply --check` 和 Python 语法检查，尚未应用到仓库。没有变更模型、50/10 划分、绝对动作定义、相机映射、优化器或学习率。

## 推荐运行参数

| 项目 | 输入检查 | 基础短跑 | 正式 batch 短跑 | 正式训练 |
| --- | --- | --- | --- | --- |
| 物理 GPU | 2、3 | 2、3 | 2、3 | 2、3 |
| FSDP devices | 2 | 2 | 2 | 2 |
| 全局 batch | 不训练 | 2 | 4 | 4 |
| 每卡样本数 | — | 1 | 2 | 2 |
| 训练更新次数 | 0 | 20 | 20 | 10,000 |
| DataLoader workers | 0 | 0 | 0 | 2 |
| 实验名 | — | b601-final60-smoke-fsdp2-b2 | b601-final60-smoke-fsdp2-b4 | b601-final60-v1 |

模型仍为官方 JAX `pi05_base`；OpenPI 锁定 `215abfb217dbac7d5f1273282331b9b1866c0479`，环境包含 JAX CUDA12 0.5.3、Flax 0.10.2、Orbax 0.11.13、Torch 2.7.1 和锁定版旧 LeRobot，不借用先前作业环境。

沿用 `gemma_2b_lora` + `gemma_300m_lora`，bfloat16 计算、EMA 关闭；模型内部已有 remat。注意锁定版冻结过滤器针对 `llm` 中非 LoRA 参数，并没有冻结视觉编码器；其显存和 checkpoint 成本不能按“只有少量 adapter 参数”估计。本方案保留该原始行为，不默默增加冻结视觉编码器的实验改动。

训练仍使用 25 Hz、25 步动作窗口、七轴原始动作及模型内部 32 维 padding、双路相机缩放到 224×224、第三路空图且 mask 为 false。归一化仅用训练集，验证共用训练统计；不额外应用关节正负号、夹爪缩放或 delta action。

正式训练学习率峰值 2.5e-5，warmup 500 步，cosine 衰减至 2.5e-6；日志每 100 步，checkpoint 每 1,000 步及最终更新后保存。W&B 关闭。全局 batch 4、10,000 步约使用 40,000 个训练窗口起点，约 1.39 遍训练帧；原 batch 8 对应约 2.78 遍。降低 batch 后不能声称数据曝光量完全相同，也不在未验证前自行延长训练。

## 执行顺序

以下是**下一阶段执行命令**，本轮未执行这些安装、下载或训练操作。

先安装用户级 uv，不改 shell 启动文件；这是一个缺失的前置工具，不需要更换 NVIDIA 驱动或系统 CUDA：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/b601-uv-install.sh
env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR=/home/yujiegao/.local/bin sh /tmp/b601-uv-install.sh
/home/yujiegao/.local/bin/uv --version
```

在仓库核对并应用已准备的补丁：

```bash
cd /home/yujiegao/桌面/zzk/rebot-b601-training
git status --short
git apply --check ../B601_SERVER_ASSESSMENT_20261109/b601_server_adaptation.patch
git apply ../B601_SERVER_ASSESSMENT_20261109/b601_server_adaptation.patch
git diff --check
git diff -- openpi_b601/train_pi05.py
```

保留补丁 diff 作为本机改动记录；若远端 main 后续更新，重新审查补丁，不能强行覆盖本地改动。

先重查 GPU 分配及空闲状态，然后准备环境和数据并读取训练、验证样本：

```bash
nvidia-smi
bash /home/yujiegao/桌面/zzk/B601_SERVER_ASSESSMENT_20261109/run_on_server.sh check
```

启动器只给子进程设置项目 GitHub 身份、GPU2/3 的 UUID、现有 Python 3.11 和独立缓存。它不切换默认 gh 身份，不自动应用补丁，也不自动衔接后续训练。需要先为 `/data2/yujiegao/b601-pi05/` 留出约 150–200 GiB 的保守空间预算；这是容量预算，不是实测消耗。

此阶段验收完整压缩包 SHA-256、prepared manifest 中的 50/10 条数与 28,729/5,118 帧、两路图像与任务文本、动作窗口、训练统计以及 JAX 可见设备。原 checker 只抽读每个集合首尾；正式长跑前宜补读所有 episode 的首/中/尾帧，排除局部解码异常。

在新建 OpenPI 环境中做小规模双卡通信探测，再启动训练；探测时关闭预分配以免仅检查就预占大块显存：

```bash
CUDA_VISIBLE_DEVICES=GPU-a2df82e1-ad48-ea47-fa69-5ee02ec32b0e,GPU-8d91782a-1f41-2aa0-ef48-585ded17365d \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
openpi_b601/.openpi/.venv/bin/python - <<'PY'
import functools
import jax
import jax.numpy as jnp
devices = jax.devices()
assert len(devices) == 2 and all(d.platform == 'gpu' for d in devices), devices
@functools.partial(jax.pmap, axis_name='g')
def collective(x):
    return jax.lax.psum(x, 'g')
result = jax.device_get(collective(jnp.ones(2)))
assert (result == 2).all(), result
print('JAX devices:', devices, 'all-reduce:', result)
PY
```

每个短跑分开启动、分开验收，防止失败后继续正式训练：

```bash
set -o pipefail
bash /home/yujiegao/桌面/zzk/B601_SERVER_ASSESSMENT_20261109/run_on_server.sh smoke \
  --exp-name b601-final60-smoke-fsdp2-b2 2>&1 | tee /tmp/b601-smoke-fsdp2-b2.log
```

验收退出码为 0、Step 0–19 共 20 条 loss 全部有限、grad_norm 有限、异步 checkpoint 完成，然后用正式 batch 再验收一次：

```bash
set -o pipefail
bash /home/yujiegao/桌面/zzk/B601_SERVER_ASSESSMENT_20261109/run_on_server.sh smoke \
  --batch-size 4 --exp-name b601-final60-smoke-fsdp2-b4 \
  2>&1 | tee /tmp/b601-smoke-fsdp2-b4.log
```

同时每秒采样两卡 `memory.used`、利用率、温度，标为“采样最大占用”；若启用了 JAX 预分配，该值包含内存池，不能称为张量实际峰值。checkpoint 必须能被 Orbax 实际读回，不能仅凭目录存在判定成功。短跑末尾目录预期是 `.../b601-final60-smoke-fsdp2-b2/19/` 与 `.../b601-final60-smoke-fsdp2-b4/19/`：上游以 0 开始命名，`19` 对应已执行 20 次更新。

两次短跑通过后，执行正式训练：

```bash
set -o pipefail
bash /home/yujiegao/桌面/zzk/B601_SERVER_ASSESSMENT_20261109/run_on_server.sh train \
  --exp-name b601-final60-v1 2>&1 | tee /tmp/b601-final60-v1.log
```

正式 checkpoint 根目录：

```text
/home/yujiegao/桌面/zzk/rebot-b601-training/openpi_b601/runs/pi05_b601_lora/b601-final60-v1/
```

完成 10,000 次更新时，预期最终目录为 `9999/`，包含 `params`、`train_state` 与归一化 `assets`。这是预期路径，目前没有生成。已有同名目录时不会默认覆盖，应查明历史运行或用原参数加 `--resume`。上游会清理部分旧 checkpoint，若要逐 checkpoint 比较验证指标，应在被删除前评测或事先调整保留策略。

长跑应由用户级 systemd 服务或持久终端管理，并将日志保存在任务目录；具体耗时先从编译后的稳定 step/s 实测估算。首次 JAX 编译时间不代表每一步速度。当前没有实测速度，不预报固定小时数。

## 显存不足及评测安排

- 双卡全局 batch 4 失败而 batch 2 通过时：可以 batch 2 正式运行，但必须记录与原计划不同的数据曝光量。双卡不能改为 batch 1，因为锁定版上游要求 batch 能被可见卡数整除。
- 双卡 batch 2 仍 OOM 时：核实两卡实际空闲和 FSDP 已启用；若本任务另行获得全部四卡，可改为四卡 FSDP、全局 batch 4（每卡 1）。需要显式设置 `B601_GPUS` 为四个 UUID、`B601_FSDP_DEVICES=4` 并覆盖短跑 `--batch-size 4`，重新验收。GPU0 桌面进程保留。四卡跨 NUMA，通信开销可能上升。
- 不在本方案里悄悄切成 PyTorch、SmolVLA、全参数训练、冻结视觉编码器，或将 RAM 当作自动显存扩展。若双卡和获分配的四卡方案均无法通过，应附 XLA/NCCL 原始报错再决定是否更换方法或硬件。
- 训练 loss 是归一化、带随机噪声的 flow-matching 目标，并包含模型 padding 维度；它不是七轴角度 MAE，也不是抓放成功率。原入口没有完整验证循环。
- 后续验证使用固定的 10 条 episode、训练集统计、同一动作适配器，在原始七维动作单位分别报告每轴 MAE/RMSE，单独报告夹爪误差，并固定随机种子和抽样策略。不要把高度重叠的 25 步窗口当独立试验。
- 离线动作误差仅用于检查/选择模型；实际效果还需同一工位的闭环抓放试验，记录抓取、搬运、放入黑框各环节成功率和失败类型。当前 60 条数据主要覆盖固定场景，不能由有限训练 loss 推断跨场景泛化。

## 交付与证据边界

本轮交付：本方案、资源/权限证据 JSON、未应用的适配补丁、修改后源码预览，以及待执行的服务器启动器。已完成真实仓库克隆、main 核对、Release 小范围读取、硬件快照与源码审查。安装依赖、完整归档校验、转换后样本读取、JAX 双卡通信、短跑 loss、正式步数与 checkpoint 可恢复性均是后续执行项目。

参考的一手来源：

- [锁定版 OpenPI 要求与安装说明](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479#requirements)：单卡 LoRA 参考需求 >22.5 GB，支持通过 FSDP 降低每卡内存需求。
- [锁定版 TrainConfig](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)、[训练循环](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py)、[数据加载器](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py)。
- [锁定版 LeRobot 读取逻辑](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py)。
- [JAX NVIDIA 安装要求](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)：驱动与 GPU 架构具备 CUDA12 JAX 的基础兼容条件，实际以锁定环境验收为准；没有 nvcc 不能单独用来判定不可运行。
- [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)与[不改 shell 配置的选项](https://docs.astral.sh/uv/reference/installer/#disabling-shell-modifications)。
