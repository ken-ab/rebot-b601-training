# B601 抓放 · RECAP 第一轮

2026 年 11 月 20 日在一张 RTX 3090 上完成训练和离线验证。服务加载本轮第 47,000 步 π₀.₅ 策略，Evo-RL 价值模型选用第 4,000 步。新策略的真机自主成功率待评测。[实验报告](REPORT.md) · [带版本号的私有模型备份](https://huggingface.co/dahjgfa/b601-recap-round1/tree/bb8c9f4de33c3a7771d4ef31775c11fcf95601bd)

| 目录 | 内容 |
| --- | --- |
| [code](code) | 3090 实际执行的训练、标签、导出和离线验证代码，含固定版本的 OpenPI 训练入口 |
| [sources](sources) | 本轮 Evo-RL 源码版本、公开初始权重与环境准备脚本 |
| [preparation](preparation) | 数据打包、早期基线与逐帧核查 |
| [records](records) | 固定实验划分、训练与验证记录、备份回执及推理读回 |
| [serving](serving) | 在 3090 运行的 OpenPI 服务、图像转发及启动脚本 |
| [windows](windows) | 现场使用的纠错采集客户端与本轮评测归档 |

## 复用模型

Hugging Face 的 `policy/checkpoint` 是完整的 OpenPI 权重，`policy/example_observation.npz` 是同一版本的参考观测。将它们放到已有 B601 服务器的策略目录，目录内分别命名为 `checkpoint` 与 `example_observation.npz`。已有启动器从 `/root/b601-deploy/active-policy` 找到这两个文件。服务端按 `conditioning.json` 自动使用 positive 条件，默认关闭 RTC。推理环境沿用[项目已有部署](../../deployment/b601_rtc)。

`value-model/` 含完整价值权重与可独立回读的脚本。`tagged-data/` 含逐帧标签和轨迹划分。每个权重和数据文件的 SHA-256 记录在 [SHA256SUMS.json](https://huggingface.co/dahjgfa/b601-recap-round1/blob/bb8c9f4de33c3a7771d4ef31775c11fcf95601bd/SHA256SUMS.json)。

## 复查本轮训练

`code/value-run.sh` 对准备好的训练目录运行价值迁移与评分，`code/run_evo_policy.sh` 生成优势标签并训练策略。实际使用的数据目录是 `/data/b601-recap-round1/evo`。时间预算、步数和划分保存在 [contract.json](records/tagged-data/contract.json) 及 HF 划分文件中。复现实验时要为新运行设置新的截止时间。OpenPI 衍生文件与 Evo-RL 许可见 [licenses](licenses)。

## 真机评测

在已有 Windows 项目目录中使用本次客户端文件，保留现场标定、任务参考图和 PCAN 配置。网络连接由本机的 `b601_execution.json` 指定，示例见 [配置](windows/b601_execution.example.json)。本轮在原纠错采集入口使用 `b601-correction.cmd --evaluation`。客户端使用 SSH 密钥或通过 `B601_EVAL_SSH_HELPER` 指定本机已有的 SSH 凭据封装，封装提供 `run` 和 `put` 命令。

本轮已对照采集图核对机位。当前 USB 布局下，俯视是 0 号，腕部是 2 号，腕部按原配置旋转 180°。采集时只在笔记本暂存当前记录，结束后上传并校验服务器上的 TAR，终局判断另存 JSON。远端路径为 `/data/b601-evaluations/recap-round1-step-47000`。在本批窗口输入 N 开始一条，H 手拖或交回模型，S 确认已放好。结束后按现场结果填 1 或 0。
