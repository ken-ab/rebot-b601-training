# 已微调 π₀.₅ 的本地与服务器推理方案

用户于2026-11-09澄清：Windows笔记本只需运行微调后的模型进行推理，不需要在笔记本训练；主模型为本次B601数据微调的π₀.₅。ACT是独立轻量对照，不默认替换π₀.₅。以下两种部署使用同一套微调参数、训练集归一化统计、图像/动作映射和任务指令，无需为改变部署位置而重新训练。

## USB位置与模型位置

机械臂和相机连接Windows，意味着Windows负责读取相机、读取七轴反馈并通过原B601适配器下发USB指令。模型既可在这台电脑运行，也可在服务器运行。模型输出25×7的absolute_adapter_input动作窗口，不直接产生USB协议字节；原适配器继续完成关节方向、夹爪比例和硬件执行。

服务器方案的循环：相机/机械臂→Windows采集图像和关节反馈→网络请求服务器π₀.₅→返回动作窗口→Windows控制程序→原B601适配器→USB机械臂。Windows是本机采集与执行节点，不能只盲目转发网络字节。

## 方案一：服务器运行微调π₀.₅，Windows控制机械臂

这是先验证微调模型真实抓放表现的建议起点。笔记本不加载大模型，不占模型推理显存；无需在Windows安装JAX或把USB重定向给Linux。前提是笔记本与服务器存在可用网络通路，网络带宽和时延需要实际测量。OpenPI官方提供[远程推理协议](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/docs/remote_inference.md)。

正式交付包完成后，在Linux服务器包目录启动服务：

```bash
bash serve.sh --host 127.0.0.1 --port 8000
```

如已有SSH访问通路，在Windows PowerShell建立本地转发（把占位符换成实际登录信息）：

```powershell
ssh -N -L 8000:127.0.0.1:8000 用户名@服务器地址
```

Windows客户端仍连`127.0.0.1:8000`，由隧道转给服务器。也可通过已有局域网/VPN通路连接；目前未验证用户笔记本到服务器的实际连通性。

包内`client/client_smoke.py`已具备轻量WebSocket请求和模型接口检查，可先发送随包真实样例：

```powershell
.\.venv-client\Scripts\python.exe .\client\client_smoke.py --host 127.0.0.1 --port 8000
```

客户端环境安装见`WINDOWS_WSL2_DEPLOYMENT_CN.md`。正式交付会另有轻量`B601_PI05_WINDOWS_CLIENT.zip`：采用服务器推理时只需把客户端和样例下载到Windows，不必下载服务器模型参数；WSL2本地推理则需要完整模型包。该脚本只测试网络与有限25×7输出，**不读取现场摄像头、不打开USB/串口、不下发机械臂动作**。真正真机bridge需接入用户本机原有相机采集和B601适配器代码；这部分尚未在服务器上完成现场联调。

实际接入时，Windows每轮构造`observation.images.overhead`、`observation.images.wrist`、`observation.state`和`prompt`，收到动作后按原适配器坐标与本机控制节奏执行，再获取新反馈。观察时间、网络超时、过期结果及断线处理应接入本机原控制程序；不要让网络阻塞直接决定USB电机指令时序。预热/JIT先完成样例推理，再进入真机执行。

已测证据是本服务器环回网络：原始图像请求暖机往返约459–479ms；客户端等价预处理后，另一轮三请求测试暖机往返约313–321ms（不含一次预处理约51ms），与原始图像独立直接推理逐元素一致。单独执行随包客户端CLI也通过，含预处理约395ms。两轮不是严格同场性能实验，不能把差值全部归因于某一个改动。以上均不是用户网络时延或25Hz闭环能力。首次请求约13.6秒含JIT，模型一次预测25步，名义25Hz执行时约1秒动作窗口；模型刷新率与本机电机指令发送频率是两个概念。

网络带宽也需考虑。旧样例发送两路640×480 float32图像，仅图像约7.37MB；当前客户端默认用`b601_observation.prepare_observation`做与服务端完全相同的224×224 RGB uint8预处理，两图301,056bytes、当前完整请求301,356bytes，相比旧float32样例图像约缩小24.5倍。全部211个真实验证观察的输入经MsgPack传输后逐元素一致，三请求网络输出也已在本服务器验证一致。`--raw-images`保留原始输入对照。Windows/真实网络仍未验收，不能把上述环回结果当作现场结果。

客户端使用`b601_websocket_client.B601WebsocketClientPolicy`，只为这条模型连接显式直连，不修改系统HTTP代理或上游vendor文件；避免websockets15继承代理后错误转发localhost请求。握手和响应均有时限，样例默认响应60秒以容纳首次JIT，可用`--timeout`设置。推理超时或通信错误后关闭并废弃当前连接，不自动重试；真实bridge应重新采集观察再发新请求，不能用迟到响应继续控制机械臂。该行为已在独立本机CPU延迟服务器测试中通过，不代表真机桥已完成。

## 方案二：Windows的WSL2运行微调π₀.₅，本机执行USB

模型放在WSL2 Ubuntu中使用4070Laptop GPU，Windows原生控制程序通过本机localhost调用它，USB仍由Windows管理。这样不依赖远端服务器和外部网络，但要在用户电脑上验收显存、主机内存与实际速度。[JAX官方平台表](https://docs.jax.dev/en/latest/installation.html#supported-platforms)将原生Windows NVIDIA GPU标为不支持、WSL2 GPU为实验性。

不能把训练资源需求当成纯推理需求。本次platform低显存分配器已在Linux A4500上用完整π₀.₅参数通过10次严格真实样例推理，采样进程峰值6902MiB、稳定6714MiB，主机内存峰值约10.45GiB。用户显存8188MiB，说明本地路径值得实际验收；但还需容纳Windows显示与其他GPU占用，A4500测量不等于4070/WSL已通过，也不是精确最低资源需求。

用户RAM32GB或更多，WSL可先按16GB预算验收。最终包的`self_test_low_memory.sh`和`serve_low_memory.sh`已经准备；安装与验收步骤见`WINDOWS_WSL2_DEPLOYMENT_CN.md`。若本机完整自检通过且延迟可接受，就能在笔记本上独立运行微调π₀.₅，无需本机再训练。

## ACT在本项目中的位置

ACT是用同一数据训练的另一种模型，不是π₀.₅的压缩或转换版本。它更小、原生Windows PyTorch部署较方便，所以被用作资源对照；这不证明它抓取更好。已测的π₀.₅中途10k模型夹爪MAE约1.8465，ACT选中15k约2.5381，关节指标另有取舍；离线误差也不能代替真机成功率。

当前交付主线：完成π₀.₅长跑与最终离线选模，保留服务器推理＋Windows执行和WSL2本地推理两条路径；ACT作为可选对照保留，不因笔记本只需推理就默认改模型。
