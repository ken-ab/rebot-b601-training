# B601 交付与发布方案

用户已要求长跑和评估完成后，一并整理微调模型、服务器推理＋Windows执行、本机WSL2推理和风险检查。主模型为本次final-60微调π₀.₅，ACT仅为可选对照。

建议GitHub保存代码、训练/评估配置、方法与硬件报告、部署入口、动作检查工具和下载说明；Hugging Face私有模型库保存完整模型参数、归一化统计、tokenizer、文件清单与模型说明。若只选一个平台交付整包，优先Hugging Face。上传位置负责版本与下载，推理仍由本Linux服务器或用户本机WSL2进程执行。

- GitHub目标：已有私有仓库`ken-ab/rebot-b601-training`。发布时使用本工作区的`github-zzk`身份，不修改默认gyj612612账号或全局Git设置；保留正在运行的脏工作树，另做干净发布准备。
- Hugging Face目标：用户最初提供dahjgfa，之后改选当前浏览器登录的GYJ账号，并报告已创建仓库和令牌；实际仓库URL及服务器授权尚待核对。以用户提供的真实仓库为准，不把GitHub用户名当作HF用户名。本助手尚未执行上传。
- 模型版本需绑定最终checkpoint、精确代码/依赖版本、完整动作评估、独立交付审计和SHA256。训练结束前的模型卡/发布包不能标成最终完成。

[GitHub文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)说明普通Git文件超过100MiB会被阻止；[Release文档](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)说明单个附件须小于2GiB。多GB完整模型不适合直接混进普通Git历史。[Hugging Face仓库](https://huggingface.co/docs/hub/repositories)面向AI/ML大文件和版本管理；实际上传前按包大小核对[账号存储配额](https://huggingface.co/docs/hub/storage-limits)。

## 发布内容

1. 完整π₀.₅推理参数、训练集统计、tokenizer和固定OpenPI依赖入口，含默认模型与明确标记的必要替代候选。
2. Linux服务器服务入口、Windows轻量客户端ZIP、WSL2低显存入口及两条路线的说明。
3. 实际训练参数、20步短跑、正式时长/更新数、基座/候选/最后模型评估、原始预测及选模规则。
4. 文件/归一化哈希、参数/优化器读回、真实样例golden、网络一致性和客户端预处理证明。
5. 默认禁止执行的动作检查配置模板、离线软件验收和现场待验证项。
6. 模型卡、上游代码许可证及Gemma条款。不能把全部衍生权重笼统标为Apache代码许可。

模型库用已有manifest作为文件白名单准备，避免把运行后新增缓存、凭证、虚拟环境、Git配置或无关数据目录上传。发布前检查文字文件中的已知凭证模式；发现问题只记录文件/类型，不在日志或聊天回显秘密内容。原始完整数据集仍按原私有数据仓库管理，不随模型任意另传。

## 授权现状

2026-11-10再次检查：默认缓存、旧版缓存、环境变量和项目缓存均没有可用上传令牌；服务器HTTPS接口可达，未认证返回401。浏览器登录与终端授权分开处理。每个项目账号使用独立缓存`/home/yujiegao/.cache/huggingface-zzk/<已确认HF用户名>`，保留默认配置。交互授权脚本`PUBLICATION_PREP/hf_login_zzk.py`可从终端隐藏输入的token自动核对账号，必要时在终端显示用户名请用户确认，再保存独立缓存，不把token写入聊天/argv/Git，也不上传文件。

实际发布必须等真实长跑、完整评估及最终审计通过。软件验收不能替代用户本机USB、相机、关节限位、校准、物理急停和闭环抓放测试；未验证项目在交付中保留为待现场验证。
