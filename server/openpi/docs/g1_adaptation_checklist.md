# G1 Robot Adaptation Checklist

本文档用于梳理把 `openpi` 适配到智元 `G1` 机器人的代码改动范围。重点不是直接给出完整实现，而是明确：

- 哪些代码必须新增
- 哪些现有代码必须修改
- 哪些属于可选增强
- 每个文件在适配中的职责是什么
- 推荐的数据字段约定是什么

本文档基于以下目标约束：

- 不接入移动底盘
- 不接入腰部维度
- 接入左臂 7 个关节角
- 接入右臂 7 个关节角
- 接入左手二指夹爪
- 接入右手二指夹爪
- 接入头部图像
- 接入左手腕图像
- 接入右手腕图像


## 1. 适配目标和总体判断

从当前 `openpi` 的模型接口看，`G1` 这次适配不需要改模型核心，原因是它和当前系统的默认输入假设是兼容的：

- 模型固定支持 3 路图像输入
- 模型固定支持一个低维 `state`
- 模型动作维度可以比真实机器人动作维度更大，真实维度可以先组织好，再在 transform 阶段补零到模型维度

对 `G1` 而言，最自然的一套状态/动作定义是 16 维：

```text
[left_arm_q(7), left_gripper(1), right_arm_q(7), right_gripper(1)]
```

总维度：

- 左臂 7
- 左夹爪 1
- 右臂 7
- 右夹爪 1
- 合计 16

这意味着：

- `state` 可以定义成 16 维
- `actions` 可以定义成 16 维
- 在送入 `pi0/pi0.5` 模型之前，再 pad 到模型的 `action_dim`，通常为 32


## 2. 推荐的统一输入输出约定

为了让训练、推理、数据转换三部分保持一致，建议先统一一套 `G1` 的中间语义定义。

### 2.1 推荐的原始推理输入字典

建议在机器人运行时代码中构造如下输入：

```python
{
    "observation/head_image": ...,
    "observation/left_wrist_image": ...,
    "observation/right_wrist_image": ...,
    "observation/left_joint_position": ...,
    "observation/left_gripper_position": ...,
    "observation/right_joint_position": ...,
    "observation/right_gripper_position": ...,
    "prompt": ...,
}
```

其中：

- `observation/head_image`：头部 RGB 图像
- `observation/left_wrist_image`：左腕 RGB 图像
- `observation/right_wrist_image`：右腕 RGB 图像
- `observation/left_joint_position`：长度为 7 的左臂关节角
- `observation/right_joint_position`：长度为 7 的右臂关节角
- `observation/left_gripper_position`：长度为 1 的左夹爪状态
- `observation/right_gripper_position`：长度为 1 的右夹爪状态

如果你后续希望和 LeRobot 数据格式更贴近，也可以统一成：

```python
{
    "images": {
        "cam_head": ...,
        "cam_left_wrist": ...,
        "cam_right_wrist": ...,
    },
    "state": ...,
    "actions": ...,
    "prompt": ...,
}
```

但建议把这种结构作为 transform 之后的中间格式，而不是机器人 SDK 的第一手输出格式。

### 2.2 推荐的模型内部图像映射

当前模型固定使用 3 个图像 key：

- `base_0_rgb`
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`

因此建议在 `G1Inputs` 里做如下映射：

- `head_image -> base_0_rgb`
- `left_wrist_image -> left_wrist_0_rgb`
- `right_wrist_image -> right_wrist_0_rgb`

### 2.3 推荐的 state/actions 顺序

统一使用：

```text
[left_arm_q(7), left_gripper(1), right_arm_q(7), right_gripper(1)]
```

这样做的好处：

- 双臂结构清晰
- 与 ALOHA 风格相近，便于理解
- 后续 delta-action mask 很容易定义

### 2.4 推荐的 gripper 表达

`G1` 适配时要尽早确定二指夹爪的表达方式。建议在训练数据和在线推理中都统一为：

- `0.0` 表示完全张开
- `1.0` 表示完全闭合

如果 `G1` SDK 输出的是电机编码器值、毫米、百分比或其他范围，建议在 `g1_policy.py` 中统一做归一化转换，不要把不同单位直接混入训练和推理链路。


## 3. 必须新增的代码

下面这些文件建议作为 `G1` 适配的第一批新增文件。

### 3.1 `src/openpi/policies/g1_policy.py`

这是本次适配最关键的新文件，建议定义：

- `make_g1_example()`
- `G1Inputs`
- `G1Outputs`
- 如有必要，额外定义 gripper 编码/解码函数

它的职责是：

- 把 `G1` 原始输入字段整理成模型统一格式
- 把三路图像转成模型需要的 `uint8 HWC`
- 把左右臂和左右夹爪拼成 16 维 `state`
- 训练时把数据集中的 `actions` 整理成 16 维动作
- 推理时把模型输出裁回前 16 维

建议参考实现：

- `src/openpi/policies/aloha_policy.py`
- `src/openpi/policies/droid_policy.py`

建议输出格式：

```python
{
    "image": {
        "base_0_rgb": ...,
        "left_wrist_0_rgb": ...,
        "right_wrist_0_rgb": ...,
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "state": state_16d,
    "prompt": prompt,
}
```

若训练时带动作，则再附加：

```python
"actions": actions_16d_or_action_chunk
```

`G1Outputs` 的主要职责是：

- 从模型动作输出中只取前 16 维

### 3.2 `examples/g1/main.py`

这是 `G1` 真机推理入口，建议作为新的示例目录单独维护。

它的职责是：

- 连接 `G1` 的机器人环境/SDK
- 读取头部、左腕、右腕相机
- 读取左右臂关节角和左右夹爪状态
- 组织请求并通过 websocket 发给 policy server
- 接收动作 chunk 并逐步下发给机器人

建议参考：

- `examples/droid/main.py`

建议在这里明确处理：

- 图像转 `uint8`
- open-loop action chunk 执行逻辑
- 控制频率同步
- 紧急停止或 `Ctrl+C` 的中断处理


### 3.3 `examples/g1/convert_g1_data_to_lerobot.py`

这是离线训练数据准备入口。

它的职责是：

- 读取你采集的 `G1` 原始数据
- 将原始数据转换为 LeRobot 数据格式
- 确保训练时的数据字段和推理时的字段语义一致

推荐导出的 LeRobot 字段：

```text
observation.state
action
observation.images.cam_head
observation.images.cam_left_wrist
observation.images.cam_right_wrist
task
```

其中：

- `observation.state`：16 维
- `action`：16 维
- 三路图像为 RGB
- `task`：语言指令

建议参考：

- `examples/aloha_real/convert_aloha_data_to_lerobot.py`
- `examples/droid/convert_droid_data_to_lerobot.py`

### 3.4 `examples/g1/README.md`

建议新增这一份说明文档，记录：

- G1 环境依赖
- 服务端启动方式
- 本机/远程推理方式
- 数据转换方式
- 微调启动命令

这不是功能性必需文件，但从工程维护角度非常建议一开始就加上。


## 4. 必须修改的现有代码

### 4.1 `src/openpi/training/config.py`

这是最重要的现有文件，`G1` 适配一定要改。

需要修改的内容包括：

#### 4.1.2 新增 `LeRobotG1DataConfig`

建议新增一个专门的数据配置类，例如：

```python
class LeRobotG1DataConfig(DataConfigFactory):
    ...
```

这个类的职责是：

- 定义 LeRobot 数据集如何 repack
- 指定 `G1Inputs` / `G1Outputs`
- 选择是否做 delta action 转换
- 应用标准模型 transform

推荐的 repack 逻辑：

```python
{
    "images": {
        "cam_head": "observation.images.cam_head",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    },
    "state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}
```

如果语言指令保存在 `task`，则改成：

```python
"prompt": "task"
```

#### 4.1.3 确定是否需要 delta action

这里是一个关键设计点：

- 如果你的 `G1` 数据集里的动作是目标关节角，那么训练前建议做 delta 化
- 如果你的 `G1` 数据集里的动作本身就是相对当前状态的增量，就不要重复做 delta transform

如果采用“关节做 delta、gripper 保持绝对值”的设计，推荐 mask：

```python
make_bool_mask(7, -1, 7, -1)
```

解释如下：

- 前 7 维左臂做 delta
- 第 8 维左夹爪不做 delta
- 接着 7 维右臂做 delta
- 最后 1 维右夹爪不做 delta

#### 4.1.4 新增 `pi05_g1_finetune`

建议新增一条微调配置，例如：

```python
TrainConfig(
    name="pi05_g1_finetune",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=<your_horizon>,
    ),
    data=LeRobotG1DataConfig(
        repo_id="<your_g1_dataset>",
        ...
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
)
```

建议默认从 `pi05_base` 开始，而不是从 `pi05_droid` 开始，原因是：

- `pi05_droid` 是单臂 DROID 专项模型
- `G1` 是双臂机器人
- 相机布局和动作语义与 DROID 差异较大

因此，`pi05_base` 作为通用 base checkpoint 更稳妥。

#### 4.1.5 归一化统计量建议

对 `G1` 来说，建议优先使用基于你自己的 `G1` 数据集重新计算得到的 normalization stats，而不是直接复用 DROID 或 ALOHA 的 stats。

原因：

- `G1` 没有官方现成 stats
- `G1` 的双臂 7-DoF + 双夹爪布局与已有公开 stats 并不严格一致
- 强行复用不匹配的 stats，容易带来动作尺度偏移

因此这里推荐：

- `weight_loader` 用 `pi05_base`
- `norm_stats` 用 `G1` 自己的数据算

#### 4.1.6 可选新增 `pi05_g1`

如果你希望后续部署命令更清晰，也可以新增一条纯推理配置，例如：

```python
TrainConfig(
    name="pi05_g1",
    ...
)
```

不过这不是必须的。很多情况下，直接用你自己的 `pi05_g1_finetune` checkpoint 配合对应 config 就可以推理。


## 5. 按功能划分的文件清单

下面给出一份更直接的“要改什么”的清单。

### 5.1 推理侧必须新增

- `src/openpi/policies/g1_policy.py`
- `examples/g1/main.py`

### 5.2 训练侧必须新增

- `examples/g1/convert_g1_data_to_lerobot.py`

### 5.3 训练侧必须修改

- `src/openpi/training/config.py`

### 5.4 文档建议新增

- `examples/g1/README.md`

### 5.5 测试建议新增

- `src/openpi/policies/g1_policy_test.py`


## 6. 可选修改项

下面这些不是第一阶段必须做，但做了会更顺手。

### 6.1 `scripts/serve_policy.py`

如果你想支持下面这种命令：

```bash
python scripts/serve_policy.py --env G1
```

那么需要修改：

- `EnvMode`
- `DEFAULT_CHECKPOINT`

增加一个 `G1` 分支。

如果你更习惯显式写：

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_g1_finetune \
  --policy.dir=/path/to/your/checkpoint
```

那这个文件可以先不改。

### 6.2 `examples/simple_client/main.py`

如果你想让随机假数据测试支持 `G1`，可以新增：

- `EnvMode.G1`
- `_random_observation_g1()`

但这只是本地 smoke test，和真机适配没有直接依赖关系。

### 6.3 `examples/simple_client/README.md`

如果你想把 `G1` 测试命令也统一写进仓库文档，可以顺手补充。


## 7. 暂时不需要修改的核心代码

下面这些文件，按你当前的 `G1` 需求，大概率不用改：

### 7.1 `src/openpi/models/model.py`

原因：

- 当前模型固定需要 3 路图像
- `G1` 正好也是 3 路图像输入

### 7.2 `src/openpi/models/pi0_config.py`

原因：

- `pi0/pi0.5` 默认 `action_dim` 可以设为 32
- `G1` 的真实动作维 16 可以在 transform 阶段 pad 到 32

### 7.3 `src/openpi/transforms.py`

原因：

- 已经有现成的 `RepackTransform`
- 已经有 `ResizeImages`
- 已经有 `PadStatesAndActions`
- 已经有 `DeltaActions` / `AbsoluteActions`

换句话说，这一层基础设施基本够用，不需要为了 `G1` 单独重写。


## 8. 建议的 `G1Inputs` 设计要点

下面给出 `src/openpi/policies/g1_policy.py` 中 `G1Inputs` 的建议职责拆分。

### 8.1 图像处理

建议处理以下兼容情况：

- 输入是 `uint8 HWC`
- 输入是 `float32 HWC`
- 输入是 `uint8 CHW`
- 输入是 `float32 CHW`

输出统一成：

- `uint8`
- `HWC`

### 8.2 state 拼接

推荐拼接方式：

```python
state = np.concatenate([
    left_joint_position,
    left_gripper_position,
    right_joint_position,
    right_gripper_position,
])
```

最终得到 16 维。

### 8.3 image key 映射

建议固定为：

```python
{
    "base_0_rgb": head_image,
    "left_wrist_0_rgb": left_wrist_image,
    "right_wrist_0_rgb": right_wrist_image,
}
```

### 8.4 image mask

如果三路图像都一定存在，则全部设为 `True`。

如果后续某一路可能缺失，可以仿照 `AlohaInputs` 的做法：

- 缺图像时补黑图
- 同时把对应的 mask 设为 `False`

### 8.5 actions 输出裁剪

`G1Outputs` 只需要返回前 16 维：

```python
return {"actions": np.asarray(data["actions"][:, :16])}
```

如果推理执行端需要特殊控制格式，比如：

- 夹爪二值化
- 夹爪范围映射
- 关节限幅

也可以在 `G1Outputs` 或 `examples/g1/main.py` 中处理。通常建议：

- 训练语义转换放在 `g1_policy.py`
- 执行安全相关的限幅放在 `examples/g1/main.py`


## 9. 建议的 `LeRobotG1DataConfig` 设计

建议这个配置类包含以下逻辑：

### 9.1 repack transforms

负责把 LeRobot 数据集字段映射到 `G1Inputs` 预期字段。

如果你的 LeRobot 数据集字段是：

```text
observation.images.cam_head
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
task
```

则可以先 repack 成：

```python
{
    "images": {
        "cam_head": "observation.images.cam_head",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    },
    "state": "observation.state",
    "actions": "action",
    "prompt": "task",
}
```

再由 `G1Inputs` 转成模型统一格式。

### 9.2 data transforms

建议使用：

- `G1Inputs(...)`
- `G1Outputs()`

如果动作是绝对关节角，再叠加：

- `DeltaActions(make_bool_mask(7, -1, 7, -1))`
- `AbsoluteActions(make_bool_mask(7, -1, 7, -1))`

### 9.3 model transforms

直接复用仓库已有的：

- `ModelTransformFactory()`

通常不需要为 `G1` 自定义。


## 10. 建议的 `G1` 数据集字段定义

在 `examples/g1/convert_g1_data_to_lerobot.py` 中，建议输出如下特征：

```python
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (16,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (16,),
        "names": ["action"],
    },
    "observation.images.cam_head": {
        "dtype": "image",
        "shape": (H, W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.cam_left_wrist": {
        "dtype": "image",
        "shape": (H, W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.cam_right_wrist": {
        "dtype": "image",
        "shape": (H, W, 3),
        "names": ["height", "width", "channel"],
    },
}
```

如果语言指令要参与训练，记得在每帧中加入：

```python
"task": language_instruction
```

或者根据 LeRobot 版本要求写入对应字段。


## 11. 建议的实施顺序

建议按下面的顺序推进，风险最低。

### 阶段 1：先打通最小离线链路

1. 新增 `src/openpi/policies/g1_policy.py`
2. 在 `src/openpi/training/config.py` 中新增 `LeRobotG1DataConfig`
3. 在 `src/openpi/training/config.py` 中新增 `pi05_g1_finetune`
4. 新增 `examples/g1/convert_g1_data_to_lerobot.py`
5. 先用少量 `G1` 数据转成 LeRobot
6. 跑一次最小微调，验证 shape、字段和 norm stats 流程

### 阶段 2：打通远程推理

1. 新增 `examples/g1/main.py`
2. 对接真实 `G1` SDK
3. 打通服务端 websocket 请求
4. 验证模型返回动作 shape 是否为 `(action_horizon, 16)`
5. 加入关节限幅和夹爪限幅

### 阶段 3：补齐易用性

1. 可选修改 `scripts/serve_policy.py`，支持 `--env G1`
2. 可选修改 `examples/simple_client/main.py`，支持 `G1` 假数据压测
3. 新增 `examples/g1/README.md`
4. 新增测试文件


## 12. 最终代码变更清单

下面给出一份适合直接开工的最终清单。

### 必须新增

- `src/openpi/policies/g1_policy.py`
- `examples/g1/main.py`
- `examples/g1/convert_g1_data_to_lerobot.py`

### 必须修改

- `src/openpi/training/config.py`

### 强烈建议新增

- `examples/g1/README.md`
- `src/openpi/policies/g1_policy_test.py`

### 可选修改

- `scripts/serve_policy.py`
- `examples/simple_client/main.py`
- `examples/simple_client/README.md`


## 13. 一句话结论

这次 `G1` 适配的核心，不是改模型，而是补一层新的机器人语义适配层。

真正要做的事情可以概括为三件：

- 新增 `G1` 的输入输出 transform
- 新增 `G1` 的数据转换与训练配置
- 新增 `G1` 的真实机器人推理入口

如果这三部分设计一致，`G1` 接入 `openpi` 的工程成本是可控的，而且不需要侵入式修改模型主体。
