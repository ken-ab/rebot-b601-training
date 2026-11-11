# B601 当前 RTC 真机推理版本（2026-11-11）

此目录保存当前笔记本与 A800 实际使用的推理改进，以及 CMD 编码修复。主流程保持 25 Hz；当前自定义运动参数为每步最大电机侧变化 0.3°、起始姿态容差 5°，其余电机配置使用已安装 Seeed Follower 的默认值。

- 动作按观测 tick 对齐，跳过回包时已经过去的动作；过期动作包维持已下发目标。
- Linux 服务实现推理期 RTC（指数前缀引导、PI 参考 beta=5），在模型完整输入变换中处理前缀，不对七轴直接求平均。
- 纸巾进入黑框、夹爪撤离且纸巾稳定后结束模型动作，调用官方两阶段回零。180 秒保留为最长时限。
- 回零时持续刷新夹爪的官方主机侧 PD，包含 SDK 的末尾 2 秒等待。
- Windows CMD 入口改为 ASCII，中文路径在 PowerShell 处理。

## Linux 服务

将 `linux` 中三个文件放入 `/root/b601-deploy/`，其中 `start_inference.sh` 是启动入口。已安装的模型与 OpenPI 环境位于 `/root/b601-deploy/B601_PI05_DEPLOYMENT/`。模型端口 8002，JPEG 转发端口 8001。启动程序先完成 GPU 自检和实际无 noise 请求分支预热，再开放服务。

```bash
bash /root/b601-deploy/start_inference.sh
```

模型为私有 Hugging Face 仓库 `dahjgfa/b601-pi05-final60`，固定版本 `8c11eb8751d5cc6f688fa51db523da5bd8f2518a`。完整模型及 Linux 运行依赖的下载/迁移方案随后存入本目录的 `migration`。

## Windows 快照

`windows` 是当前 `lerobot-windows` 项目的覆盖快照，在原项目目录使用；依赖该电脑已经安装的官方 Seeed/MotorBridge、原有客户端与 PCAN/FFmpeg 路径、USB 相机绑定。相机 640×480，腕部旋转 180°。固定任务板图像也随快照保存，供此任务完成判定使用。此目录不包含机械标定文件和 Windows 驱动。

```powershell
.\b601-start-live.cmd -CheckOnly
.\b601-start-live.cmd
```

前一条只做软件和服务检查，后一条恢复起点并启动真机。Q 正常回零退出，Esc 立即失能。桌面入口可由本机既有 CMD 使用。

## 验证

`validation` 保留了 37 项软件检查、四份历史录像完成判定，以及云端真实 checkpoint 的 RTC RPC 检查记录。录制图像回放 12 次往返中位数 382.96 ms、最大 458.34 ms。5 秒软件控制回放为 125 步（25 Hz）。这些记录区分软件回放与实际机器人执行。

算法依据：[Physical Intelligence RTC](https://www.pi.website/research/real_time_chunking)，[论文](https://arxiv.org/abs/2506.07339)，[参考代码](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)。参考代码许可证见 `third_party/rtc`。
