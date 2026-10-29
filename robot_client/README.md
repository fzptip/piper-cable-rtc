# Piper 客户端

当前入口：`pi0.5_rtc_control_v3.sh`。完整部署、运行和历史版本说明见项目根目录的 `README.md`。

| 文件 | 功能 |
|---|---|
| `pi05_ws_client.py` | ROS 观测采集、WebSocket 异步推理、机械臂控制状态机 |
| `rtc_action_queue.py` | 承诺前缀、过期结果处理、换块过渡和插值端点预留 |
| `action_interpolator.py` | 以单调时钟推进的关节线性插值 |
| `diagnostic_recorder.py` | 异步记录模型输出、发布目标、关节反馈和阶段标记 |
| `pi0.5_control.sh` | 加载已有 Conda/ROS 环境，按本目录定位程序 |
| `pi0.5_rtc_control_v3.sh` | 当前 v3 参数预设，不依赖旧版本脚本 |

`test_rtc_v2.py` 检查当前代码继承的换块过渡功能，是当前回归测试；仓库没有存放 v2 源码快照。

离线测试（模拟 ROS、相机、网络和发布者，不启动客户端主程序）：

```bash
cd robot_client
python -W error::ResourceWarning -m unittest test_rtc_client test_diagnostics test_rtc_v2 test_action_interpolation -v
```
