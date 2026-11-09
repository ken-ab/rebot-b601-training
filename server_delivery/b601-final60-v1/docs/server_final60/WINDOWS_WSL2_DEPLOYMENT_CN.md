# Windows + NVIDIA 部署交接

用户机器为联想Y7000P IRX9（83DG），RTX4070Laptop，8188MiB专用显存，系统RAM32GB或更多，可安装WSL2。用户已澄清本机只需运行微调后的π₀.₅推理，不需要本机训练。主模型保留π₀.₅；本文介绍WSL2本地推理，另一路是服务器运行相同模型、Windows采集和USB执行。完整架构与服务器连接步骤见`PI05_INFERENCE_ARCHITECTURE_CN.md`。已完成训练的ACT仅作可选对照，不默认替换π₀.₅。

JAX 官方将原生 Windows NVIDIA GPU 标为不支持，将 WSL2 GPU 标为实验性。本次已在 Linux RTX A4500 上通过完整动作评估、独立部署目录自检和 WebSocket 环回检查；尚未在用户的 Windows/WSL2 机器上执行验收。

## WSL2 推理服务

1. 按 NVIDIA 和 Microsoft 官方说明准备 Windows NVIDIA 驱动及 WSL2 Ubuntu 22.04。Windows PowerShell 可检查 `wsl --list --verbose`；WSL1 不满足 GPU 路径。WSL 内使用 Windows 映射的驱动，不另装 Linux NVIDIA 显示驱动。
2. 在 WSL 内运行 `nvidia-smi` 和 `free -h`，核对可见 GPU、空闲显存与可用系统内存。显存需求不能从服务器预分配的18GiB显存池直接推断；上游的推理>8GB只是粗略估计，必须以本机完整自检为准。
3. 把参数压缩包解压到 WSL 的 Linux 文件系统，例如 `~/models/b601/`。进入 `B601_PI05_DEPLOYMENT` 目录，确认 `git`、`uv` 可用后执行：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false bash install_openpi.sh
```

该命令使用固定 OpenPI commit 和锁定依赖，并运行包含文件 SHA256、真实样本、参考动作比较的自检。保持 JAX0.5.3及锁定CUDA依赖，不直接用最新版JAX/CUDA替换。安装环境首次需要网络，参数和 tokenizer 已在包内。

自检通过后，在同一目录启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 bash serve_low_memory.sh --host 127.0.0.1 --port 8000
```

保持终端运行。首个请求含 JIT 编译，后续请求复用已编译模型。服务端自动设置交付的确定性推理 flags；先导入 JAX 的自写程序须按主 README 提前设置同样 flags。

当前A4500实测：未预分配但采用默认池时进程驻留8382MiB；改用platform分配器，十次真实样本推理均通过严格golden，采样进程峰值6902MiB、稳定驻留6714MiB。采样可能漏掉极短瞬态，且A4500不是用户的4070/WSL，因此不能当作目标机已验收。此配置加载的主机RAM峰值约10.45GiB，也须核对WSL内存限额与Windows总RAM。可用 `bash self_test_low_memory.sh` 重做完整交付自检。

Microsoft文档给出的WSL2默认内存上限为Windows总RAM的50%。在已确认的32GB或更多主机上，16GB的WSL预算可作为本次约10.45GiB加载峰值的起点，仍需以`free -h`和实际自检为准。若现有WSL限额太小，可在Windows的WSL设置中调整，或合并到用户目录`.wslconfig`（保留已有其他配置）：

```ini
[wsl2]
memory=16GB
```

设置需要WSL重启生效；先保存并结束正在运行的WSL工作，再按Microsoft说明重启。无需为ACT原生Windows路线设置WSL内存。

当前锁环境已在服务器A4500验证，用户4070Laptop的WSL环境仍应执行完整自检。若出现库、驱动或算子错误，保留原包和报错，单独分析推理环境；保持训练参数与已验证环境的来源可追溯。

## 原生 Windows 客户端

把交付的 `client` 目录和 `example_observation.npz` 放在同一个父目录。使用 Python3.11，在该父目录 PowerShell 执行：

```powershell
py -3.11 -m venv .venv-client
.\.venv-client\Scripts\python.exe -m pip install -r .\client\requirements_minimal.txt
.\.venv-client\Scripts\python.exe .\client\client_smoke.py --host 127.0.0.1 --port 8000
```

客户端附带同一固定 commit 的 OpenPI 纯 Python WebSocket 源码及最小依赖；不需要在 Windows 客户端安装 JAX、Torch 或完整训练环境。脚本用随包真实样本请求模型，检查七轴名称、25Hz元数据及有限25×7输出，只打印结果，不驱动机械臂。

客户端默认把两路图像等价处理成224×224 RGB uint8以降低网络流量，已在服务器对211个验证观察及实际WebSocket请求验证输入/输出一致；Windows实机仍需运行上述命令。它对模型连接显式直连，保留系统其他代理配置；响应默认60秒超时，超时后废弃连接且不自动重试。`--raw-images`可发送原始图像作对照。现场相机和原B601 USB适配器需接入同一个请求接口，样例脚本不包含这部分。

Microsoft 文档说明 Windows 可通过 localhost 访问 WSL 内服务。若连接失败，应先在 WSL 内检查服务 `/healthz`，再检查 WSL 的 localhost 转发配置；保留两个终端的关键报错后分析，不改模型参数来解决网络问题。

接入实际 B601 程序时，Windows 端发送 `observation.images.overhead`、`observation.images.wrist` 两路 RGB 图像、`observation.state` 七轴电机实测反馈及 `prompt`。输出是与示教一致的 absolute_adapter_input，由原 B601 适配器完成方向和夹爪比例转换，避免重复转换。摄像头和机械臂串口可继续留在 Windows，无需为了模型推理迁入 WSL。

若机器不能使用 WSL2，可保留 Windows 相机/机械臂客户端，把相同推理服务放在本 Linux GPU 服务器，通过可达网络或 SSH 端口转发连接；这条路径不是原生 Windows 本地 GPU 推理。

## 验收边界与来源

最终应先通过 WSL GPU 可见性、交付包自检、Windows 客户端真实样本请求，再验证现场机械臂闭环行为。离线动作误差、文件自检和网络一致性都不能替代真机抓放成功率。

- [JAX 平台兼容表](https://docs.jax.dev/en/latest/installation.html#supported-platforms)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Microsoft WSL 网络](https://learn.microsoft.com/en-us/windows/wsl/networking)
- [Microsoft WSL 内存配置](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- [本次固定 OpenPI 的环境要求](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#requirements)
