# B601 微调、评估及部署最终交接

实际验收时间：2026-11-10T01:46:16.541024+08:00。π₀.₅ 正式长跑完成 12.2732 小时、43001 次优化器更新，最后 checkpoint 编号 43000。计时从第一次成功训练调度结束后开始，排除环境准备和初始编译，包含正常保存与验证。最终完成标记、checkpoint绑定计数、CPU参数/优化器读回、完整动作评估、部署包自检、实际WebSocket协议与压缩包逐文件校验均已通过。

## 数据与训练方法

final-60 原固定50训练/10验证episode划分，训练28,729帧、验证5,118帧，两路RGB、七轴状态、25步动作窗口、名义25Hz。归一化统计只来自训练集，验证使用相同统计。完整动作评估包含211个窗口，去掉padding后每条原验证动作计一次；标签、mask及源episode/frame与先前固定协议逐元素一致。

服务器为双Xeon Gold6530、Linux可见系统内存总量约503GiB、四张RTX A4500 20GB；π₀.₅使用物理GPU2/3、FSDP2，ACT独立使用物理GPU1。训练与评估入口、版本、哈希、smoke和恢复探针保存在原记录及部署包receipts中。

π₀.₅使用OpenPI固定commit215abfb217dbac7d5f1273282331b9b1866c0479，官方pi05_base，JAX0.5.3，LoRA语言与动作分支，原生过滤器同时训练视觉部分，因此交付完整参数。global batch=4，workers=2，seed42，500步warmup、峰值lr2.5e-5、10k后衰减至2.5e-6。训练的200k是步数上限，实际按12.25小时门槛后的完整保存点结束。AdamW与模型精确配置见receipts/RUN_PARAMETERS.json。原始flow-best保持独立；最终部署按已锁定的最多四个候选在六关节MAE/夹爪MAE上的Pareto规则选择，默认交付last（43001次更新）。

两组20步π₀.₅短跑均通过有限loss/grad和参数读回：batch2首/末loss0.0350/0.0312；batch4为0.0358/0.0477。短跑是工程验收，不能当作精度提升。正式训练的控制台指标按100步区间均值记录且四位小数；本次W&B重绑定问题的原因、补救与原执行源码保存在记录中，不能把该日志称为每一步全精度loss。固定验证使用80个观察的flow代理指标，真实动作比较使用全部5,118行。

## 完整验证结果

| 模型 | 六关节平均 MAE | 夹爪 MAE |
| --- | ---: | ---: |
| π₀.₅ 纯基座（未微调） | 20.970796 | 11.627045 |
| π₀.₅ 5k（flow代理最优） | 2.356301 | 2.105016 |
| π₀.₅ 10k | 2.063429 | 1.846545 |
| π₀.₅ 43k（最终交付） | 1.914644 | 1.820183 |
| ACT 15k（独立对照） | 2.013707 | 2.538114 |

单位为原始absolute_adapter_input，两个指标分开报告。逐轴RMSE、逐episode和宏平均、原预测及mask均保存，可独立复算。验证集参与选模，不能当作独立测试；离线示教动作一致性不是机械臂抓放成功率。base是单独重新运行的无LoRA纯基座对照，初始随机LoRA flow值不作为纯base替代。

## 对本次结果的结论

微调后的43k模型相对纯基座，六关节平均MAE下降90.87%，夹爪MAE下降84.35%。在本次固定验证协议下，微调明显提高了与B601示教动作的一致性。

从10k继续到43k，关节MAE再下降7.21%，夹爪再下降1.43%；额外训练的收益已有差异，不能据此推断步数越多会无限改善。本次末端模型在这两个指标上都优于已比较的早期候选，因此按预定规则选中末端模型；这个结论不是因为它步数最大。12小时以上是本次约定的运行预算，部署版本仍由验证指标决定。

80观察的flow代理在5k更低，但完整动作评估在43k更好，说明该代理不能替代真实七轴动作误差。表中的5k、10k和43k是三个不同的微调候选；审计文件保留的best/candidate_flow_best、last/candidate_last是同一候选的来源标签，不代表额外独立模型。

验证仅覆盖这10条episode，且参与选模；本次没有多随机种子、未触碰测试集或现场闭环成功率证据。因此可以确认这批数据上的离线适配有效，尚不能确认抓放成功率或新场景泛化，也不能把剩余误差单独归因于数据量、训练时长或研究方向。

## 针对用户Windows笔记本的部署

目标为联想Y7000P IRX9（83DG），RTX4070Laptop8188MiB，RAM32GB或更多，可使用WSL2。LoRA不限定操作系统；JAX原生Windows GPU的兼容限制与LoRA本身无关。

用户已澄清，笔记本只需部署已微调模型进行推理，不需要在本机训练；主模型保留π₀.₅。两种部署均使用本次微调后的同一模型：服务器GPU推理＋Windows采集图像/反馈并经原B601适配器通过USB执行动作；或WSL2在4070Laptop上本地运行π₀.₅，再由Windows控制USB。USB连接位置不要求模型必须同机运行。建议先完成服务器推理链路与本机客户端联调，再依据WSL2实机验收决定是否脱离服务器；目标机到服务器的网络和现场闭环尚未测试。

ACT仅保留为轻量独立对照，不默认替换π₀.₅。它不按语言prompt切换任务，不能仅凭体积小宣称抓放更好。ACT正式完成20k更新，原normalized L1规则选中15k；CPU与GPU整套5,118行读回通过。服务器单样本暖机中位36.04ms，PyTorch峰值allocated258.27MiB，不含CUDA上下文/驱动开销，不是4070本机性能承诺。

π₀.₅保留WSL2 Ubuntu低显存路线。当前最终交付参数在本服务器platform分配器10次严格golden检查通过，采样进程峰值6900MiB、主机内存峰值10207.57MiB。采样可能漏短峰值，A4500/Linux不能代替4070/WSL实机验证。32GB主机可先以16GB WSL预算验收。官方JAX平台表将原生Windows NVIDIA GPU标为不支持、WSL2 GPU为实验性，见https://docs.jax.dev/en/latest/installation.html#supported-platforms。

输入为两路RGB、七轴原始电机侧反馈；输出25×7 absolute_adapter_input，由原B601适配器转换方向与夹爪比例，避免重复转换。预测25步不等于必须开环执行25步，现场执行/重观测节奏需在原控制循环中验证。服务器未连接或驱动现场机械臂。

## 参数及文件

- π₀.₅最终训练checkpoint：`/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1/43000`。
- π₀.₅默认部署参数源：`/data2/yujiegao/b601-pi05/runs/pi05_b601_lora/b601-final60-v1/43000`。
- π₀.₅包：`/home/yujiegao/桌面/zzk/B601_LONGRUN_20261109/B601_PI05_DEPLOYMENT.tar.gz`，SHA256 `258c85c7405d6d8e67241a8acfd93f4020b397fa1ce302e7602e038f8b653657`。
- ACT包：`/home/yujiegao/桌面/zzk/B601_ACT_20261109/B601_ACT_DEPLOYMENT.tar.gz`，SHA256 `5df0e2ccb64783f83f45b4726021a6816fbfed5139808bde26e684d11c940aab`。
- 完整独立核验：`INDEPENDENT_FINAL_AUDIT.json`、ACT目录`INDEPENDENT_DELIVERY_AUDIT.json`。
- 安装入口：两个包各自`README_CN.md`；π₀.₅另有`WINDOWS_WSL2_DEPLOYMENT_CN.md`、`self_test_low_memory.sh`及Windows轻量客户端。

服务器保留优化器checkpoint。π₀.₅已在独立探针验证优化器/模型/更新计数恢复，最终checkpoint的全部优化器数组和计数也已读回；上游DataLoader位置不恢复，不承诺逐步完全一致续训。ACT保存了优化器/RNG材料，但尚无已验收的精确resume入口。Windows4070本机自检与现场机械臂闭环验收仍由目标机执行，这里不作已经完成的声明。

## Windows轻量客户端与部署检查

Windows专用客户端ZIP：`/home/yujiegao/桌面/zzk/B601_LONGRUN_20261109/B601_PI05_WINDOWS_CLIENT.zip`，大小627176字节，SHA256 `a6571511a2e34e86066525ea787661be23ed65ef61808a71191613f50e90b14b`。包内不含模型权重，适用于服务器推理路线；最终ZIP中20个内容文件已逐一与完整模型包清单核对。它提供真实记录样本RPC与动作检查接口；尚需接入Windows本机原B601适配器、两路相机及实际控制循环，当前示例不会下发USB动作。

软件动作检查器的36个合成案例已通过，默认配置保持不能启用动作批准。真实限位、校准、反馈和停止行为需要现场确认；检查器不能物理停止已经发出的动作。详细边界与本机操作见 `DEPLOYMENT_READINESS_CN.md`、`PI05_INFERENCE_ARCHITECTURE_CN.md` 和 `WINDOWS_WSL2_DEPLOYMENT_CN.md`。

模型发布目标为 https://huggingface.co/dahjgfa/b601-pi05-final60 。实际上传状态和固定下载版本以独立发布回执为准；服务器验收完成与已上传模型是两个独立状态。代码、方法和结果在原私有GitHub仓库的独立交付分支发布，保留原训练目录与main。
