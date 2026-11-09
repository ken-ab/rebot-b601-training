# B601 本地部署方案与已完成证据

截至 2026-11-09 20:49（北京时间），ACT 的独立 20,000 步训练、参数导出、CPU/GPU 全量读回和压缩包独立核验均已完成。原 π₀.₅ JAX LoRA 长跑仍在运行，其 12.25 小时目标及最终评估尚未完成，不能把本 ACT 交付当作主长跑完成。

用户机器：联想拯救者 Y7000P IRX9（83DG），Windows，RTX 4070 Laptop 8188MiB 专用显存，系统内存 32GB 或以上，可安装 WSL2 Ubuntu。LoRA 是微调方法，不要求 WSL2；当前 π₀.₅ 路线需要 WSL2，是因为所选 JAX NVIDIA GPU 平台兼容性，而非 LoRA 本身的限制。

## 当前建议

对于固定的蓝色餐巾纸抓放任务，优先用本 ACT 原生 Windows PyTorch 包进行本机离线验收，再评估机械臂闭环表现。ACT 不按语言指令切换任务。保留 π₀.₅ 完整训练和低显存 WSL2 路线，待其最终动作验证完成后，结合精度与部署成本选择；不能仅凭 ACT 更小就宣称抓取更好。

| 路线 | 本次证据 | 目标机状态 |
| --- | --- | --- |
| ACT + PyTorch 原生 Windows CUDA | 已完成20k训练；主模型15k；独立CPU/GPU全部5118动作行读回通过 | 尚需用户4070 Windows实机自检 |
| π₀.₅ + JAX + WSL2 | Linux低显存platform分配器10次真实样本自检通过；采样进程峰值6902MiB、主机内存峰值约10.45GiB | 尚需WSL实机自检；最终长跑交付尚未完成 |
| SmolVLA等更小语言动作模型 | 本次未训练或验收 | 语言/多任务需求明确时再评估，当前不交付未验证参数 |

JAX 官方将原生 Windows NVIDIA GPU 标为不支持、WSL2 GPU 标为实验性，见[官方兼容表](https://docs.jax.dev/en/latest/installation.html#supported-platforms)。PyTorch 提供[Windows CUDA安装路径](https://pytorch.org/get-started/locally/)。ACT 的任务定位见[LeRobot官方说明](https://huggingface.co/docs/lerobot/act)。

## ACT 实际训练与选模

沿用 final-60 固定50/10划分：训练28,729帧，验证5,118帧；两路RGB相机，七轴状态，25步动作窗口。图像224像素letterbox，状态/动作统计仅来自训练集。验证模型输入不含action或padding标签，使用部署时的零潜变量。

训练在服务器物理GPU1（RTX A4500 20GB）上进行，FP32、batch8、seed42、AdamW、学习率及骨干学习率均1e-5、weight_decay1e-4、KL权重10，共20,000次更新。记录活动墙钟2,383.525秒，训练峰值PyTorch allocated1,593.96MiB、reserved1,970MiB。20步短跑全部loss有限并保存/读回参数后才进入正式训练；短跑loss首88.53569、末9.06018，包含KL，不能当作抓取准确率。

原定规则是全部有效验证动作行的normalized L1最小值，最后选中15,000步 checkpoint；原BEST未因后验指标分析而更换。20,000步的夹爪误差更低、关节误差更高，属于取舍。所有训练checkpoint保留在服务器。

| 模型 | 六关节平均MAE | 夹爪MAE | 说明 |
| --- | ---: | ---: | --- |
| ACT选中15k | 2.013707 | 2.538114 | normalized valid L1=0.126679，实际交付模型 |
| ACT最后20k | 2.224401 | 2.256288 | 最后checkpoint，未替换原BEST |
| π₀.₅已完成的10k动作评估 | 2.063429 | 1.846545 | 主长跑中途对照，最终选模尚未完成 |

单位均为原始absolute_adapter_input；关节均值与夹爪分开报告，不合成不同单位的分数。三者使用相同211个验证窗口、5,118有效动作行；target、padding mask和源episode/frame逐元素一致。验证集参与选模，这些不是独立测试或真实抓放成功率。ACT当前比π₀.₅ 10k关节误差略低，夹爪误差仍较高；不能称整体胜出。

## 导出与验收

部署包：`B601_ACT_DEPLOYMENT.tar.gz`，192,293,479 bytes（约183.39MiB）。SHA256：`5df0e2ccb64783f83f45b4726021a6816fbfed5139808bde26e684d11c940aab`。

主参数来源：`/data2/yujiegao/b601-act-final60/runs/act-b601-final60-v1/step_015000`。最后可恢复材料保存在同级`step_020000`；ACT未实现并验收精确的数据迭代器恢复入口，不承诺逐步完全复现resume。π₀.₅另有已验收的优化器恢复流程。

包内只需PyTorch/Torchvision、NumPy、Pillow、Einops，不依赖JAX、LeRobot或Transformers。模型、精确float32归一化缓冲、配置、真实样例、参考输出、许可证和验证预测均已包括。加载不再下载ImageNet骨干权重。包内README包含Python3.11以及Torch2.7.1/Torchvision0.22.1官方CUDA12.6安装命令。

独立服务器验收包括27个manifest文件及压缩包全部28个普通文件的大小/SHA256核验、指标float64重算、CPU与GPU完整211窗口/5,118行推理。CPU相对原训练GPU最大差5.340576e-5，GPU整批读回最大差0，均在事先固定的atol1e-3/rtol1e-4内。

服务器A4500单样本预热后中位耗时36.0ms、P95约63.0ms，CPU约97.6ms。GPU单样本测试PyTorch峰值allocated258.27MiB、reserved300MiB；这不包含驱动/CUDA上下文等进程开销。以上是本服务器证据，不是4070笔记本性能承诺，也不代表已证明25Hz闭环推理。

解包后先按`B601_ACT_DEPLOYMENT/README_CN.md`在Windows执行`self_test.py --device cuda`，它核验文件并用真实样例对比参考动作，不驱动机械臂。输入两路RGB及七轴原始电机侧反馈，输出25×7绝对适配器动作；由原B601适配器完成方向及夹爪比例转换，不能重复转换。模型窗口25步不等于必须开环执行25步；实际重新观察和执行节奏由原控制循环决定。

完整本次服务器验收在`INDEPENDENT_DELIVERY_AUDIT.json`，实际训练完成在`DELIVERY_RECEIPT.json`，逐轴/逐episode指标在包内`evaluation/METRICS.json`。Windows/WSL2目标机和现场机械臂尚未在此服务器验收。
