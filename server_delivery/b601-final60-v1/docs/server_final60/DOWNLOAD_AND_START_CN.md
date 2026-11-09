# 下载模型与开始部署

本次交付主模型为微调后的π₀.₅，最后checkpoint编号43000、实际43001次更新。固定模型版本：`8c11eb8751d5cc6f688fa51db523da5bd8f2518a`。模型仓库为私有仓库，需使用dahjgfa或有读取权限的账号。

## 路线一：服务器推理，Windows负责相机和USB

Windows只需下载[客户端ZIP](https://huggingface.co/dahjgfa/b601-pi05-final60/resolve/8c11eb8751d5cc6f688fa51db523da5bd8f2518a/B601_PI05_WINDOWS_CLIENT.zip?download=true)，大小627176字节，SHA256 `a6571511a2e34e86066525ea787661be23ed65ef61808a71191613f50e90b14b`。这条路线不需要把完整模型放进笔记本显存。

客户端环境与连接命令见ZIP中的`WINDOWS_WSL2_DEPLOYMENT_CN.md`、`PI05_INFERENCE_ARCHITECTURE_CN.md`和`client/README_CN.md`。服务器运行已微调模型，Windows发送观察并接收动作；远程连接先按指南建立可达的SSH通道，真实网络时延仍需测量。

提供的`client_smoke.py`只发送记录样本进行RPC自检。实际机械臂还需要接入Windows本机原B601适配器、相机和控制循环；默认动作配置不能启用USB动作。

## 路线二：本机WSL2运行π₀.₅

下面命令在**你自己的Windows笔记本的WSL2 Ubuntu**运行。先按现有部署指南确认WSL NVIDIA驱动可用，并准备Python3/venv、Git和uv。完整模型下载约6.34GB；环境安装还需额外磁盘空间与网络。CPU下载环境与模型推理环境分开。

```bash
mkdir -p ~/b601-deploy
cd ~/b601-deploy
python3 -m venv .venv-download
.venv-download/bin/python -m pip install huggingface_hub==0.32.3
export HF_HOME="$HOME/.cache/huggingface-b601-download"
.venv-download/bin/huggingface-cli login
```

登录时在隐藏提示处输入有模型读取权限的token，不把token写进命令或脚本。无需添加Git凭据。随后下载固定版本：

```bash
.venv-download/bin/huggingface-cli download dahjgfa/b601-pi05-final60 \
  --revision 8c11eb8751d5cc6f688fa51db523da5bd8f2518a \
  --local-dir B601_PI05_DEPLOYMENT
cd B601_PI05_DEPLOYMENT
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_CLIENT_MEM_FRACTION
CUDA_VISIBLE_DEVICES=0 bash install_openpi.sh
CUDA_VISIBLE_DEVICES=0 bash self_test_low_memory.sh
```

安装脚本末尾也会运行模型自检，因此从安装开始就设置低显存分配器。通过后启动本机模型服务：

```bash
CUDA_VISIBLE_DEVICES=0 bash serve_low_memory.sh --host 127.0.0.1 --port 8000
```

Windows原生客户端再按指南连接本机WSL服务；USB适配器仍由Windows控制。服务器最终参数在A4500/Linux上采样显存峰值6900MiB、主机内存峰值约10208MiB；这不是4070Laptop/WSL实测结果。32GB主机可从16GB WSL内存预算开始验收。若严格golden自检失败，保留报错分析，不自动放宽误差容限。

## 真机前的边界

先运行记录样本RPC和软件检查，再完成原适配器/相机集成、实际限位与校准、观察时效、停止行为和闭环验证。详见`DEPLOYMENT_READINESS_CN.md`。检查器不能物理停止已发出的动作。离线MAE下降不等于已经验证真机抓放成功率。
