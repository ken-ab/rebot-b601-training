# B601-RS 上的 OpenPI π₀.₅ 微调

任务是「拿起蓝色餐巾纸，把它放进黑色框里」。从 [final-60 私有 Release](https://github.com/ken-ab/rebot-b601-training/releases/tag/final-60) 下载 LeRobot v3.0 数据，使用仓库的 [校验清单](../DATASET.sha256) 和 [划分](../splits.json)。50 条 / 28,729 帧用于训练，10 条 / 5,118 帧留作验证。

## 在服务器运行

在 Linux NVIDIA GPU 服务器上使用 `git`、`gh`、`uv`、`flock`、`tar` 和 `sha256sum`。GitHub CLI 需要能读取本私有仓库和 Release。首次运行需要访问 GitHub、OpenPI 锁定的 Python 依赖源和模型资产，或已有相应缓存。[安装 uv](https://docs.astral.sh/uv/getting-started/installation/) · [OpenPI 环境要求](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479#requirements)

```bash
gh auth status
nvidia-smi
gh repo clone ken-ab/rebot-b601-training
cd rebot-b601-training
bash openpi_b601/run_pi05.sh smoke
```

已克隆仓库时，在现有目录用 `git pull --ff-only` 更新。短跑默认用单卡、batch 1、20 步；服务器可见多张卡时，先用 `CUDA_VISIBLE_DEVICES` 选定分配给这次任务的一张。入口会准备固定版本的 OpenPI 环境，从 GitHub Release 下载并校验数据，将 LeRobot v3.0 转成 OpenPI 所需的 v2.1。随后会抽读训练和验证样本，检查图像、任务文本和动作形状，再训练并保存 checkpoint。可用 `bash openpi_b601/run_pi05.sh check` 仅检查输入。

确认短跑 loss 为有限数值且 checkpoint 写入成功后，启动正式微调。

```bash
bash openpi_b601/run_pi05.sh train --exp-name b601-final60-v1
```

默认加载官方 JAX `pi05_base`，用 LoRA 训练 10,000 步，batch 8。显存紧张时加 `--batch-size 4`，并按可用显存调整。已有本地 JAX 基座时加 `--base-params /absolute/path/pi05_base/params`。已有固定版本的 OpenPI checkout 时，可将 `B601_OPENPI_ROOT` 指向其绝对路径。正式训练 checkpoint 写入 `openpi_b601/runs/pi05_b601_lora/b601-final60-v1/`。恢复同一实验时使用原参数并加 `--resume`。

## 数据和动作

- 俯视相机映射到 `base_0_rgb`，腕部相机映射到 `left_wrist_0_rgb`，第三路为空且被掩码。两路源视频均为 640×480、25 Hz，转换时保留原 MP4 字节。任务文本从 LeRobot 元数据读取。
- 七维状态是从臂电机侧反馈，七维动作是 B601 适配器输入坐标中的绝对关节目标。动作窗口 25 步，约 1 秒。状态和动作统计只用训练集计算，验证集共用这份统计。真机推理时沿用相同的 B601 适配器处理关节方向和夹爪比例。
- 下载缓存位于 `openpi_b601/.data/`，转换结果和原始条号记录位于 `openpi_b601/prepared/`。第二次启动会复用转换结果。若校验报缓存损坏，按提示移走该缓存文件后重试。若修改了划分，将旧 `prepared/` 移到旁边再运行。

先在留出的完整 episode 上检查预测动作，再在同一工位记录真机抓放成功率和失败类型。后续如要采集人工纠错，从这次微调的 checkpoint 开始跑真机。
