# 动作、通信和诊断语义

机器人通过 WebSocket/msgpack 发送 `state`、三路 RGB HWC uint8 图像和 `prompt`。内部相机名是 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。外部动作 14 维，顺序为左 6 关节、左夹爪、右 6 关节、右夹爪；模型内部补齐为 32 维。

RTC 请求包含 `mode`、`session_id`、`request_id`、`observation_tick`、`delay`、`prefix`、`prefix_attention_horizon`。前缀是原任务单位的绝对目标，服务按当前状态重做相对动作与归一化；不要传入旧的归一化动作。`prefix[0]` 对应 `observation_tick`。

delay、前缀长度和队列 tick 的单位均为 **30 Hz 模型动作步**，不是毫秒或 90 Hz 发布次数。首次无前缀；之后从旧队列构造。最多一个请求在途，拒绝跨会话和过期结果。

12 个关节在模型节点之间线性插值，夹爪保持起点值，到下一个模型节点更新。模型时钟由单调时间推进；预留的当前插值终点不会被异步计划替换，后续再接局部过渡。无前瞻端点时保持末点，下一模型边界暂停；调度间隔超过一个模型周期或时间异常时暂停，不跳点追赶。

日志语义：

- `model_response.actions` 已经过 RTC 和输出变换，alpha<1 时还包含 EMA。
- `command.target` 是尝试发布的最终目标，`publish_ok` 只表示 publish 调用返回。
- `command.interpolation` 保存起终点、phase 和两端来源；关节按两端插值，夹爪来自起点。
- `source` 表示模型起点来源，不应直接把该原始预测当作 v3 最终发布值。局部过渡也可能改变目标，要结合 `planned_targets`。
- `feedback` 是两臂独立回调，不假定同步；effort 不直接解释为接触力。
- ROS header 与 monotonic 时间分别保存，不可直接相减。`queue_tick` 始终数模型节点。
- 数字键只写阶段注释。summary 完整只证明记录器写完且自身未丢记录，不证明上游无丢帧。

单位和标定沿用已有数据/驱动，本次整理未重新标定。
