# B601 抓放价值模型

这个权重为同一时刻的俯视 RGB、腕部 RGB 和七轴实测电机角度评分。角度顺序沿用训练数据的 J1 到 J7，单位是度。任务是把蓝色餐巾纸放进黑框。输出在 −1 到 0 之间，越接近 0，模型预测的剩余代价越低。每帧的正负标签还需要结合后续状态、耗时、人工接管区间和训练集阈值计算。

`model.safetensors` 包含完整骨干和价值头。`b601.json` 保存训练集归一化统计和权重指纹。`metrics.json` 给出验证与测试成绩，`readback.json` 保存了六帧独立回读结果。

## 在训练环境或具有相同依赖的 Linux 机器上使用

原实验使用 Python 3.11、PyTorch 2.7.1、torchvision 0.22.1、transformers 4.57.1、safetensors 0.6.2、huggingface-hub 0.35.3、draccus 0.10.0、NumPy 1.26.4 和 OpenCV 4.11。从本目录运行以下命令。首次加载将从完整权重在本地创建骨干缓存，预留约 3 GB 空间。

```bash
mkdir -p evorl
tar -xzf evorl-source.tar.gz -C evorl --strip-components=1
export PYTHONPATH="$PWD/evorl/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python evo_bundle.py verify --bundle . --device cpu
```

给自己的观测打分时，传入两张同一时刻的照片和对应的七个实测角度。下面的零值只是命令格式示例。

```bash
python evo_bundle.py predict --bundle . --device cuda \
    --overhead overhead.jpg --wrist wrist.jpg --state 0 0 0 0 0 0 0
```

对整条轨迹连续打分时，可以只创建一次 `B601Value`，重复调用 `predict(overhead_rgb, wrist_rgb, motor_degrees)`。
