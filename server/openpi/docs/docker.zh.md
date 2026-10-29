### Docker 配置

本仓库中的所有示例都同时提供了常规运行方式和使用 Docker 的运行说明。虽然 Docker 不是必需的，但仍然推荐使用，因为它可以简化软件安装、提供更稳定的环境；对于依赖 ROS 的示例，还能避免在本机安装 ROS，减少对系统环境的干扰。

- Docker 的基础安装说明见[这里](https://docs.docker.com/engine/install/)。
- Docker 必须以 [rootless mode](https://docs.docker.com/engine/security/rootless/) 安装。
- 如果你要使用 GPU，还需要安装 [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 通过 `snap` 安装的 Docker 与 NVIDIA container toolkit 不兼容，会导致它无法访问 `libnvidia-ml.so`（见 [issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)）。可以使用 `sudo snap remove docker` 卸载 `snap` 版本。
- Docker Desktop 同样与 NVIDIA runtime 不兼容（见 [issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)）。可以使用 `sudo apt remove docker-desktop` 卸载 Docker Desktop。

如果你是从零开始配置，并且宿主机是 Ubuntu 22.04，那么可以使用这两个便捷脚本一次性完成上述配置：`scripts/docker/install_docker_ubuntu22.sh` 和 `scripts/docker/install_nvidia_container_toolkit.sh`。

使用下面的命令构建 Docker 镜像并启动容器：
```bash
docker compose -f scripts/docker/compose.yml up --build
```

如果你想为某个特定示例构建并运行 Docker 镜像，可以使用下面的命令：
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
其中 `<example_name>` 是你想运行的示例名称。

第一次运行任何示例时，Docker 都会先构建镜像，这个过程可能需要一些时间。等镜像缓存好之后，后续运行会快很多。
