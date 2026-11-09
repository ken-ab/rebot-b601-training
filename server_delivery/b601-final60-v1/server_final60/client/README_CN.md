# Windows π₀.₅ 轻量客户端

本目录只运行网络客户端，不包含π₀.₅参数、JAX或Torch。模型可以运行在Linux服务器，也可以运行在同一笔记本的WSL2中。Windows继续负责相机、关节反馈和原B601 USB适配器。`client_smoke.py`只发送记录样本并打印推理结果，不读取现场相机、不下发机械臂动作。

在解压后包含`client/`和`example_observation.npz`的父目录中，用Python3.11的PowerShell执行：

```powershell
py -3.11 -m venv .venv-client
.\.venv-client\Scripts\python.exe -m pip install -r .\client\requirements_minimal.txt
.\.venv-client\Scripts\python.exe .\client\client_smoke.py --host 127.0.0.1 --port 8000
```

如果服务运行在远端服务器，先在另一PowerShell建立已有SSH通路：

```powershell
ssh -N -L 8000:127.0.0.1:8000 用户名@服务器地址
```

如果服务运行在本机WSL2，则使用Windows到WSL的localhost通路。服务器或WSL端须先按完整模型包说明加载本次微调后的π₀.₅。单独的Windows客户端ZIP不含模型权重，不能独立产生动作。

在`client`目录运行、或将该目录加入Python模块路径后，真实控制程序可复用以下请求接口；摄像头采集和原B601适配器的动作执行仍需接入用户本机已有代码：

```python
from b601_observation import prepare_observation
from b601_websocket_client import B601WebsocketClientPolicy

client = B601WebsocketClientPolicy('127.0.0.1', 8000, timeout=60)
request = prepare_observation({
    'observation.images.overhead': overhead_rgb,
    'observation.images.wrist': wrist_rgb,
    'observation.state': seven_motor_feedback_values,
    'prompt': '拿起蓝色餐巾纸，把它放进黑色框里',
})
actions = client.infer(request)['actions']
client.close()
```

图像需为RGB，HWC uint8或CHW/HWC的[0,1]浮点值。状态是原始七轴电机侧反馈，不在客户端做归一化或夹爪比例转换。返回25×7 absolute_adapter_input，由原适配器执行转换，勿重复转换。

图像默认沿服务端同样的np.rint和PIL letterbox流程处理为224RGBuint8，两图301,056bytes。服务器已对211个真实验证观察核验输入逐元素一致，并完成实际RPC/独立直接推理对比；这不代表用户Windows、网络或机械臂已验收。`--raw-images`保留原图对照。

客户端为这条连接显式直连，不改变系统其他代理配置。响应默认60秒超时，允许首次JIT；可用`--timeout`调整。超时或通信错误后连接即关闭/废弃，不自动重发。真实控制程序应在预热后设置适合现场的时限，重新采集观察再请求，并由原控制程序处理过期动作和断线；本样例不实现完整真机bridge。

完整两条部署路线见同包`PI05_INFERENCE_ARCHITECTURE_CN.md`。上游OpenPI源码与版本哈希见`UPSTREAM.json`及`UPSTREAM_LICENSE`；b601_observation.py和b601_websocket_client.py是本项目适配层。

动作检查接口与现场待验证项见同包`DEPLOYMENT_READINESS_CN.md`。`guard_self_test.py`可运行36个合成软件案例；默认`robot_deployment_template.json`不可启用，`deployment_check.py`会列出待配置项。检查器不会控制USB，也不能让已运动硬件停止；真实桥接仍需接入原B601适配器。
