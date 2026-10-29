# 第三方代码来源

- `server/openpi/` 来自 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi)，并包含原有机器人适配及本项目的推理期 RTC 改动。Apache-2.0 文本保存在 `server/openpi/LICENSE`，Gemma 相关条款保存在 `server/openpi/LICENSE_GEMMA.txt`。
- RTC 的 JAX VJP guidance 参考 [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 的作者实现，对应 MIT 文本保存在 `server/openpi/deployment/RTC_THIRD_PARTY_LICENSE.txt`。
- `robot_client/` 源自本任务已有的 Piper ROS 客户端，叠加异步队列、诊断记录、换块过渡和插值功能。源码来源提交见 `docs/source_manifest.json`。

本次整理保留第三方许可证，不替团队自有代码选择新的开源许可；对外发布时应由项目所有者确定自有部分的许可。
