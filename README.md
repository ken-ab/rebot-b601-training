# B601-RS 抓放训练交接

任务是「拿起蓝色餐巾纸，把它放进黑色框里」。示教用 reBot Arm 102 控制 B601-RS。当前数据版为 `blue-tissue-center-20261109-final-60`，包含 **60 条 episode、33,847 帧**。推荐按整批划分为 50 条训练和 10 条验证。筛选理由和原始条号见 [episodes.csv](episodes.csv)，两端进度记在 [STATUS.md](STATUS.md)。

数据是 LeRobot v3.0，标称 25 fps。俯视和腕部相机都是 640×480。每帧有 7 维状态和动作，顺序是底座、肩、肘、腕俯仰、腕偏航、腕滚转、夹爪。状态是从臂电机侧实测角度，动作是采集适配器输入坐标中的绝对目标。真机推理通过同一个适配器下发动作，由适配器完成关节方向和夹爪比例转换。

## 在服务器微调 OpenPI π₀.₅ Base + JAX LoRA

用能读取本私有仓库和 Release 的 GitHub 账号，在 Linux NVIDIA GPU 服务器上运行。

~~~bash
gh auth status
gh repo clone ken-ab/rebot-b601-training
cd rebot-b601-training
bash openpi_b601/run_pi05.sh smoke
~~~

已克隆仓库时，在现有目录执行 `git pull --ff-only`，再跑短测。脚本会从 [final-60 Release](https://github.com/ken-ab/rebot-b601-training/releases/tag/final-60) 下载原始数据，核对 SHA-256，按 50/10 划分转换为 OpenPI 所需格式。短测用官方 `pi05_base` 权重，以 batch 1 训练 20 步。

确认短测 loss 和 checkpoint 正常后启动正式训练。

~~~bash
bash openpi_b601/run_pi05.sh train --exp-name b601-final60-v1
~~~

环境、相机与动作映射、显存参数和输出目录见 [OpenPI 训练说明](openpi_b601/README.md)。

## 单独验收 LeRobot v3 数据

请使用有权读取这个私有仓库的 GitHub 账号。在服务器上首次接手时克隆仓库；已经克隆的，进入现有目录执行 `git pull --ff-only`，再运行下面的目录设置。

~~~bash
git clone https://github.com/ken-ab/rebot-b601-training.git
cd rebot-b601-training
~~~

~~~bash
repo="$PWD"
archive="blue-tissue-center-20261109-final-60.tar.gz"
inbox="$HOME/incoming"
root="$repo/datasets/blue-tissue-center-20261109-final-60"
mkdir -p "$inbox" "$repo/datasets"
~~~

如果服务器已登录 GitHub CLI（`gh auth status`），运行：

~~~bash
gh release download final-60 --repo ken-ab/rebot-b601-training --pattern "$archive" --dir "$inbox"
~~~

也可以在 ToDesk 连入的服务器浏览器里打开 [final-60 Release](https://github.com/ken-ab/rebot-b601-training/releases/tag/final-60)，在 Assets 中下载 `blue-tissue-center-20261109-final-60.tar.gz`，保存到 `~/incoming`。

用仓库的 [SHA-256 清单](DATASET.sha256) 核对压缩包。校验通过才会解压，随后检查数据集：

~~~bash
(cd "$inbox" && sha256sum -c "$repo/DATASET.sha256") &&
  tar -xzf "$inbox/$archive" -C "$repo/datasets" &&
  python scripts/check_dataset.py "$root"
~~~

查看服务器环境：

~~~bash
nvidia-smi
python -V
python -m pip show lerobot torch
~~~

本机使用 LeRobot 0.6.2 和 PyAV 验证。服务器如需重建环境，先确认其 Python、CUDA 和 Torch，再参考官方安装指引安装所选模型需要的依赖。[LeRobot 安装指引](https://huggingface.co/docs/lerobot/installation)

## LeRobot 备选训练

验证集来自两个完整录制批次，新补录的 7 条均留作验证。条号写在 [splits.json](splits.json) 中，使用从 0 开始的训练版索引。用 LeRobot 的拆分工具生成独立的训练和验证目录，分别重建各自的统计。

~~~bash
lerobot-edit-dataset \
  --repo_id=local/blue-tissue-center-20261109-final-60 \
  --root="$root" \
  --new_root="$repo/datasets/splits-final-60" \
  --operation.type=split \
  --operation.splits="$(cat splits.json)" \
  --push_to_hub=false
~~~

如果选用官方 LeRobot π0.5，可先跑下面的 20 步，检查双路图像输入、损失、显存和 checkpoint 写入。服务器需有 CUDA、支持 bfloat16 的 GPU 和 PaliGemma tokenizer 的访问权限。

~~~bash
lerobot-train \
  --dataset.repo_id=local/blue-tissue-center-20261109-final-60_train \
  --dataset.root="$repo/datasets/splits-final-60/train" \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --rename_map='{"observation.images.overhead":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --policy.empty_cameras=1 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --batch_size=1 --num_workers=0 --steps=20 --log_freq=1 --save_freq=20 \
  --wandb.enable=false \
  --output_dir="$repo/runs/pi05-smoke-$(date +%Y%m%d-%H%M%S)"
~~~

俯视相机对应基座视角，腕部对应单个腕相机槽位。短跑通过后记录 checkpoint、软件版本、训练参数、峰值显存和输出目录，再开始正式微调。先在留出的完整 episode 上比对预测动作与示教，并用现场抓放成功率做后续评估。[LeRobot π0.5](https://huggingface.co/docs/lerobot/pi05) · [基础权重配置](https://huggingface.co/lerobot/pi05_base/blob/main/config.json) · [数据拆分文档](https://huggingface.co/docs/lerobot/using_dataset_tools)


## 本服务器 final-60 微调结果与部署交接

本次 π₀.₅ 长跑、完整离线评估、实际硬件、方法与部署检查见
[服务器交付报告](server_delivery/b601-final60-v1/docs/server_final60/FINAL_REPORT_CN.md)。
权重下载与发布状态见 [MODEL_DOWNLOAD.json](server_delivery/b601-final60-v1/MODEL_DOWNLOAD.json)。
软件和离线验收已在报告中逐项区分；Windows/WSL 与现场机械臂闭环验证仍需在目标机执行。
