# 当前任务的 RTC 部署

完整部署和运行流程见仓库根目录 [README](../../../README.md)，动作与日志语义见 [protocol.md](../../../docs/protocol.md)。当前目录保存修改后的 openpi 工程及依赖锁文件。

在 server/openpi 中启动：

```bash
bash deployment/start_rtc.sh --host 127.0.0.1 --port 8000 --smoothing-alpha 1.0
```

程序默认端口仍为 8001，因此与机器人 18000→8000 隧道配合时需要显式传入 8000。脚本自动读取仓库根目录的 configs/server.env，也可通过 RTC_ENV_FILE 指定文件。

- RTC_PYTHON：已有服务器 Python 解释器。
- RTC_CHECKPOINT / --checkpoint：微调权重位置。
- RTC_ASSETS_BASE_DIR / --assets-base-dir：归一化统计基目录；命令行覆盖环境变量。
- --steps 10：默认去噪步数。
- --max-guidance 10：默认 guidance 上限。
- --smoothing-alpha 1.0：关闭 EMA。

程序先预热普通及 RTC 路径，再监听。/healthz 返回 HTTP 200 用于检查连通性；experiments/rtc-ready.json 是启动记录，不能单独证明服务仍然存活。

benchmark_rtc.py 和 replay_rtc.py 使用外部观测进行离线检查，不计算真机成功率。依赖旧 Git 历史的 validate_baseline.py 和 --check-baseline 选项未纳入本仓库。

CPU 测试在当前目录、已有服务器环境中执行：

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/scripts"
JAX_PLATFORMS=cpu /root/miniconda3/envs/pi0.5_wq/bin/python -m unittest discover -s tests -p 'test_rtc_*.py' -v
```

上游通用文档保留作参考，本任务以根目录 README 为准。
