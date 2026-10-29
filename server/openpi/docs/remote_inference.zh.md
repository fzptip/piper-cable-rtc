# 远程运行 openpi 模型

我们提供了一些用于远程运行 openpi 模型的工具。这种方式适合把推理放到机器人本体之外、算力更强的 GPU 上运行，同时也有助于将机器人环境与 policy 环境隔离开来，从而避免机器人软件依赖过于复杂、彼此冲突的问题。

## 启动远程 policy server

要启动一个远程 policy server，你可以直接运行下面的命令：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

`env` 参数用于指定要加载哪个 $\pi_0$ checkpoint。这个脚本底层实际上会执行类似下面的命令；如果你想启动自己训练得到的 checkpoint 对应的 policy server，也可以直接这样做（下面以 DROID 环境为例）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

这样会启动一个 policy server，对外提供由 `config` 和 `dir` 参数指定的 policy。该服务会监听指定端口（默认是 8000）。

## 在机器人代码中查询远程 policy server

我们提供了一个依赖很少的客户端工具，你可以很方便地把它嵌入到任何机器人代码库中。

首先，在你的机器人环境中安装 `openpi-client` 包：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

之后，你就可以在机器人代码中使用这个客户端来查询远程 policy server。下面是一个使用示例：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

这里的 `host` 和 `port` 参数分别指定远程 policy server 的 IP 地址和端口。你也可以把它们作为命令行参数传给机器人代码，或者直接硬编码在机器人代码库里。`observation` 是一个包含观测和 prompt 的字典，其格式需要符合你当前所服务的 policy 的输入规范。针对不同环境，如何构造这个字典，我们在 [simple client 示例](../examples/simple_client/main.py) 中提供了更具体的例子。
