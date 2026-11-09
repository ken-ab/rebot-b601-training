# B601-RS 抓放训练交接

任务是「拿起蓝色餐巾纸，把它放进黑色框里」。示教用 reBot Arm 102 控制 B601-RS。当前数据版为 `blue-tissue-center-20261109-final-60`，包含 **60 条 episode、33,847 帧**。推荐按整批划分为 50 条训练和 10 条验证。筛选理由和原始条号见 [episodes.csv](episodes.csv)，两端进度记在 [STATUS.md](STATUS.md)。

数据是 LeRobot v3.0，标称 25 fps。俯视和腕部相机都是 640×480。每帧有 7 维状态和动作，顺序是底座、肩、肘、腕俯仰、腕偏航、腕滚转、夹爪。状态是从臂电机侧实测角度，动作是采集适配器输入坐标中的绝对目标。真机推理通过同一个适配器下发动作，由适配器完成关节方向和夹爪比例转换。

## 在服务器验包

用服务器已有的 GitHub 登录克隆本仓库。已有工作目录可先执行 `git pull --ff-only`。通过 QDS 环境允许的文件传输方式，把 [校验清单](DATASET.sha256) 中列出的数据包放进服务器的传输入口。下面以 `~/incoming` 为例。

~~~bash
git clone https://github.com/ken-ab/rebot-b601-training.git
cd rebot-b601-training
repo="$PWD"
archive="blue-tissue-center-20261109-final-60.tar.gz"
inbox="$HOME/incoming"
root="$repo/datasets/blue-tissue-center-20261109-final-60"
mkdir -p "$repo/datasets"
(cd "$inbox" && sha256sum -c "$repo/DATASET.sha256") &&
  tar -xzf "$inbox/$archive" -C "$repo/datasets"
nvidia-smi
python -V
python -m pip show lerobot torch
python scripts/check_dataset.py "$root"
~~~

本机使用 LeRobot 0.6.2 和 PyAV 验证。服务器如需重建环境，先确认其 Python、CUDA 和 Torch，再参考官方安装指引安装所选模型需要的依赖。[LeRobot 安装指引](https://huggingface.co/docs/lerobot/installation)

## 划分和微调

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

前文提到的 π0.6 先在服务器上确认具体 checkpoint、来源和配套训练代码。Physical Intelligence 发表了 π*0.6 的研究，其官方 openpi 仓库的基础权重清单列出了 π0、π0-FAST 和 π0.5。若服务器已有可用的 π0.6 权重，先按它的加载与相机映射说明验跑。[π*0.6 研究](https://www.pi.website/blog/pistar06) · [openpi 权重清单](https://github.com/Physical-Intelligence/openpi#model-checkpoints)

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
