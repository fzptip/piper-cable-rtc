# 实验记录

- `reports/validation.md`：历史结果、适用范围和本轮整理验证。
- `reports/v3_historical_replay_20261019.json`：2026-10-19 保存的历史插值回放结果，属于历史产物，不是本轮重新测出的数据。
- `reports/repository_checks.json`：本轮整理后的测试与校验摘要。
- `analysis/replay_interpolation.py`：使用外部保存的原始数据复现历史插值检查。

原始示教、权重、视频和诊断日志不包含在本仓库中。新增实验请记录代码提交号/标签、权重路径及版本、运行参数、场景条件、任务结果和原始数据位置。生成的大文件放入被 Git 忽略的 `outputs/` 或外部目录。

历史回放要求原先的 2026-10-16 数据集和对应 CSV，共 292 个模型块、235 个边界窗口；不是任意新日志分析器。示例在仓库根目录执行：

```bash
python experiments/analysis/replay_interpolation.py \
  --diagnostics /path/to/robot_diagnostics \
  --analysis-dir /path/to/robot_diagnostics_analysis_20261016 \
  --output outputs/interpolation_replay
```

此脚本使用模拟控制器，既不连接 GPU 服务器，也不发机器人指令。历史日期和数据规模检查保留，用于避免误把其他数据作为原实验复现。
