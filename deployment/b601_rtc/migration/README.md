# 本地下载、换卡与恢复方案

目标：换卡后恢复当前 π0.5 + RTC 推理服务。模型与下载缓存留在本机，租卡时间用于运行模型。

## 已核对的下载内容

| 内容 | 固定来源 / 版本 | 用途 |
| --- | --- | --- |
| 已微调完整 checkpoint、归一化统计、tokenizer、推理底座代码 | Hugging Face `dahjgfa/b601-pi05-final60`，revision `8c11eb8751d5cc6f688fa51db523da5bd8f2518a`，139 个文件、约 6.34 GB | 当前 final43000 / optimizer43001 模型 |
| 当前 RTC 与 Windows 执行代码 | 本仓库 `deployment/b601_rtc` | 时间对齐、RTC、成功停止、回零夹爪持续控制、CMD 编码修复 |
| OpenPI 源码 | `Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479` | 当前模型对应实现 |
| LeRobot 源码 | `huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` | OpenPI 数据/配置模块使用的旧版依赖；不同于笔记本的 LeRobot 0.6.2 |
| Linux CPython | 云端实际 Python 3.11.18，解释器压缩包约 33 MB | 保持当前环境及绝对路径 |
| Linux Python/GPU 依赖 | `python-packages.json` 逐包固定版本 | JAX/JAXlib/CUDA 插件 0.5.3、Flax 0.10.2、Torch 2.7.1、Transformers 4.53.2、NumPy 1.26.4、Orbax 0.11.13 等 |

`deployment-info.json` 记录原机 Ubuntu 24.04.3、x86_64、NVIDIA 驱动 595.84 等实际环境；`installed-sources.json` 记录 editable 路径与 Git 依赖版本。优先租相同 Ubuntu 24.04 x86_64、带兼容 NVIDIA 驱动的镜像。Linux 环境包不能在 Windows 中直接运行；Windows 只保存和转运这些文件。

本次恢复推理无需重新下载 60 条训练数据，也无需另外下载一份 π0.5 Base 或重新微调。GitHub 原训练说明保留在 `openpi_b601/README.md`；这里记录的是实际已完成微调后的推理路径。

## 两种环境来源

优先取回已经装好的 Linux 环境，解压后使用原路径，省去安装步骤。本次生成的文件如下，精确字节数和 SHA-256 见 `ARCHIVES.json`：

| 文件 | 压缩后字节数 |
| --- | ---: |
| `b601-code-assets.tar.gz` | 293,827,142 |
| `b601-linux-runtime.tar.zst` | 3,291,300,404 |
| `b601-pi05-checkpoint.tar.zst` | 6,335,504,821 |

checkpoint 也已经能从 Hugging Face 独立下载。最终本地包使用 `hf-model`，不再重复装入第三个 checkpoint 压缩包。下载缓存、令牌、SSH 密钥、临时进程文件不进入迁移包。显卡驱动和 Linux 内核由新云镜像提供。

同时缓存 Linux CPython 3.11 的 wheel/source 安装文件，作为环境包未取回或不兼容时的重建备用。此路径需要在新 Linux 上安装与验证；源代码分发包可能还需要编译工具。`linux-wheelhouse/status.json` 会列出只有源码包的依赖和下载失败项，不将其宣称为已安装成功。

## 在本机下载

使用本机已有 Python 与 Hugging Face 读取凭证运行。脚本不保存令牌、不安装 Linux 依赖到 Windows；已有文件复用，HTTP 下载支持续传并检查哈希。

```powershell
python download_assets.py --destination D:\B601-cloud-backups\20261111-rtc
```

模型已下载时可只补缓存：

```powershell
python download_assets.py --skip-model --destination D:\B601-cloud-backups\20261111-rtc
```

私有模型凭证沿用 Hugging Face 标准登录缓存。缺少读取权限时先在本机完成 Hugging Face 登录；不要把令牌写入 GitHub、归档或命令行。

生成一个可以整体上传的 ZIP。外层使用存储模式，避免再次压缩已压缩的大文件：

```powershell
python build_portable_bundle.py --directory D:\B601-cloud-backups\20261111-rtc
```

优先使用完整且校验通过的环境包。若它未取回，则使用校验通过的 Linux wheel/source 缓存创建备用包；`portable-manifest.json` 的 `environment_mode` 区分两者，半个环境文件不会进入 ZIP。源码重建模式额外缓存 Hatchling、Poetry Core 等构建工具，新镜像需要 GCC 与 Linux 头文件（缺少时安装 `build-essential linux-libc-dev`）。构建依赖与当前推理依赖分开标记。

## 换卡后上传并恢复

Windows 自带 SFTP，经 SSH 加密，并支持 `reput` 续传，比中断后重传整个大文件省时间：

```text
sftp -P 新端口 用户@新IP
reput D:/B601-cloud-backups/20261111-rtc/B601-RTC-portable.zip /root/B601-RTC-portable.zip
bye
```

在新 Ubuntu 上：

```bash
python3 -m zipfile -e /root/B601-RTC-portable.zip /root/B601-RTC-portable
cd /root/B601-RTC-portable
bash restore.sh --verify-only
bash restore.sh
```

恢复程序校验文件，恢复到 `/root/b601-deploy`，检查 JAX GPU，启动并预热 RTC 服务：模型 8002、JPEG 8001。模型预热包括无显式 noise 的实际请求分支。

随后关闭旧 SSH 隧道，笔记本使用新 IP/端口启动桌面 CMD，或手工转发：

```text
ssh -N -L 127.0.0.1:8766:127.0.0.1:8001 -p 新端口 用户@新IP
```

## 传输方式怎么选

- 同平台若允许复用保留磁盘或自定义镜像，可省去整包上传；是否能换卡挂载按平台实例选项确认。
- 换平台或原实例释放后，用本地完整包恢复；大文件通过 SFTP 续传。
- 新服务器能快速访问 Hugging Face/PyPI 时，可从原站下载模型并复用本地 wheel 缓存，避免所有流量都绕过笔记本。
- 新旧服务器同时可用时，直接服务器到服务器传输可省去下载到本机再上传的一程。

SSH/SFTP 不会提升线路本身带宽。本次原云机 SFTP 约 2 MB/s，而本机直连 Hugging Face 已在约两分多钟取得模型，因此把公开依赖和模型下载移到本地更合适。权重文件已经压缩，主要省时来自复用现成环境、断点续传和选择更快的数据来源。
