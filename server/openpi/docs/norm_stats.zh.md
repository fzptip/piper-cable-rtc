# 归一化统计量

按照常见做法，我们的模型在 policy 训练和推理期间，会对本体感觉状态输入和动作目标进行归一化。用于归一化的统计量是基于训练数据计算得到的，并与模型 checkpoint 一起保存。

## 重新加载归一化统计量

当你在新数据集上微调我们的某个模型时，需要决定是：(A) 复用已有的归一化统计量，还是 (B) 基于新的训练数据重新计算统计量。哪种选择更适合你，取决于你的机器人和任务与预训练数据集中机器人和任务分布的相似程度。下面我们列出了各个模型可用的全部预训练归一化统计量。

**如果你的目标机器人与这些预训练统计量中的某一个相匹配，可以考虑重新加载同一套归一化统计量。** 这样做之后，你数据集中的动作会对模型来说更“熟悉”，从而可能带来更好的表现。你可以在训练配置中添加一个 `AssetsConfig`，指向对应的 checkpoint 目录和归一化统计量 ID，以重新加载归一化统计量。下面是一个例子，它使用的是 `pi0_base` checkpoint 中 `Trossen`（也就是 ALOHA）机器人的统计量：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

完整的训练配置示例可参见[训练配置文件](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)中的 `pi0_aloha_pen_uncap` 配置，它展示了如何重新加载归一化统计量。

**注意：** 若要成功重新加载归一化统计量，务必确保你的机器人和数据集遵循预训练时使用的 action space 定义。下面我们会对这些 action space 定义做详细说明。

**注意 2：** 重新加载归一化统计量是否有益，取决于你的机器人和任务与预训练数据集中机器人和任务分布的相似程度。我们建议两种方式都尝试一下：一种是重新加载已有统计量，另一种是在新数据集上重新计算一套新的统计量（如何计算可参见[主 README](../README.md)）。最后选择在你的任务上效果更好的方案。


## 提供的预训练归一化统计量

下面列出了我们提供的全部预训练归一化统计量。它们同时适用于 `pi0_base` 和 `pi0_fast_base` 两个模型。对于 `pi0_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_base/assets`；对于 `pi0_fast_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。
| Robot | 描述 | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF 双臂机器人，使用 parallel grippers | trossen |
| Mobile ALOHA | 安装在 Slate 底盘上的移动版 ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于 DROID 配置、带 parallel gripper 的 7-DoF 机械臂 | droid |
| Franka Emika (non-DROID) | 配备 Robotiq 2F-85 gripper 的 Franka FR3 机械臂 | franka |
| UR5e | 配备 Robotiq 2F-85 gripper 的 6-DoF UR5e 机械臂 | ur5e |
| UR5e bi-manual | 双臂 UR5e 配置，使用 Robotiq 2F-85 grippers | ur5e_dual |
| ARX | 双臂 ARX-5 机器人机械臂配置，使用 parallel gripper | arx |
| ARX mobile | 安装在 Slate 底盘上的移动版双臂 ARX-5 配置 | arx_mobile |
| Fibocom mobile | 配备 2x ARX-5 机械臂的 Fibocom 移动机器人 | fibocom_mobile |


## Pi0 模型的 action space 定义

默认情况下，`pi0_base` 和 `pi0_fast_base` 都使用如下 action space 定义（其中 left 和 right 的方向，是以“从机器人背后朝工作区看”的视角来定义的）：
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

本体感觉状态（proprioceptive state）使用与 action space 相同的定义；但对于移动机器人，底盘的 x-y 位置（最后两个维度）不会包含在 proprioceptive state 中。

对于 7-DoF 机器人（例如 Franka），我们使用 action space 的前 7 个维度表示关节动作，第 8 个维度表示 gripper 动作。

Pi 机器人的一般说明：
- 关节角使用弧度表示，其中零位对应各机器人接口库报告的零位；但 ALOHA 是个例外，标准 ALOHA 代码采用了略有不同的约定（详见 [ALOHA 示例代码](../examples/aloha_real/README.md)）。
- gripper 位置的范围是 `[0.0, 1.0]`，其中 `0.0` 表示完全张开，`1.0` 表示完全闭合。
- 控制频率方面，UR5e 和 Franka 为 20 Hz，ARX 和 Trossen（ALOHA）机械臂为 50 Hz。

对于 DROID，我们采用原始的 DROID 动作配置：前 7 个维度是关节速度动作，第 8 个维度是 gripper 动作，控制频率为 15 Hz。
