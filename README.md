# Piper 双臂网线插拔：π0.5 + 推理期 RTC

本项目保存松灵 Piper 双臂网线插拔任务的当前客户端、π0.5 JAX 推理服务及离线测试。采用**异步推理 + 推理期 RTC + chunk 间局部过渡 + chunk 内高频线性插值**，继续使用已有微调权重，无需为这些推理改动重新训练。

整理日期：**2026-10-29**。仓库版本：**0.1.0**，机器人算法基线为 **RTC v3**。这是新仓库的初始整理版，`0.1.0` 与此前算法名称 `v3` 是两套编号。

**本轮只在本地整理和验证，未更新机器人或 GPU 服务器，未执行真机运动，未创建或推送 GitHub 远程仓库。** 2026-10-19 的旧部署记录不代表本次新目录已部署。

## 1. 目录与当前版本

```text
piper-cable-rtc/
├── README.md                     # 部署、运行、当前及历史版本
├── VERSION                       # 新仓库版本号
├── THIRD_PARTY_NOTICES.md         # 第三方来源及许可证位置
├── robot_client/                 # 当前 Piper ROS 客户端与模拟测试
│   ├── pi05_ws_client.py
│   ├── rtc_action_queue.py
│   ├── action_interpolator.py
│   ├── diagnostic_recorder.py
│   ├── pi0.5_control.sh           # 环境加载入口
│   └── pi0.5_rtc_control_v3.sh    # 当前运行入口
├── server/openpi/                # 当前 openpi 工程及依赖锁文件
│   ├── deployment/start_rtc.sh
│   ├── scripts/serve_rtc.py
│   ├── src/openpi/
│   ├── packages/openpi-client/
│   └── tests/
├── configs/                      # 配置示例，本机 *.env 不进入 Git
├── tests/                        # 整理后的路径、启动参数及配置检查
├── docs/                         # 来源清单与通信、日志语义
└── experiments/                  # 精选报告、离线历史回放脚本
```

仅保存当前实现，不包含旧版本源码目录、旧 Git 历史、权重、原始示教、诊断日志、视频和备份压缩包。服务器保留 openpi 通用模块和示例以维持完整依赖；本任务部署以本 README 为准。`test_rtc_v2.py` 检查当前代码继承的换块过渡功能，不是旧源码副本。

两端代码位于同一仓库，但**分别使用各自 Python 环境运行**。

| 当前默认参数 | 设置 |
|---|---|
| 模型 / GPU | 已微调 π0.5，JAX，已有 B200 环境 |
| 输出 | 每块 50 步，每步 14 维：左 6 关节、左夹爪、右 6 关节、右夹爪 |
| 模型动作频率 | 30 Hz，50 步约 1.667 秒 |
| 目标发布频率 | 90 Hz，不改变模型轨迹总时长 |
| chunk 内处理 | 12 个关节线性插值；夹爪在模型节点更新 |
| 推理调度 | 异步，最多一个请求在途；发布线程继续消费队列 |
| 重规划间隔 | 15 个模型动作，约 0.5 秒 |
| RTC 延迟预算 / 软约束范围 | 5 步 / 20 步，均为模型动作步 |
| chunk 间处理 | 6 步局部 Hermite 过渡，约 200 ms |
| EMA | alpha=1.0，关闭 |

90 Hz 是待实机验证的配置，不代表已测得 Piper 的稳定控制上限。线性插值仍有节点速度变化，不能消除全部模型抖动；第一模型目标沿用原行为，未新增从 A 到首目标的过渡。

## 2. 已有路径与新部署位置

| 内容 | 已有设备路径，供辨认和回退 |
|---|---|
| 原版 GPU 工程 | `/root/kube-user/ns100002-chenrui/cr/pi0.5_jax_wq` |
| 原版服务优先引用的源码 | `/root/kube-user/ns100002-chenrui/cr/tqq/openpi_cobot_magic_work/src` |
| 已有 RTC GPU 工程 | `/root/kube-user/ns100002-chenrui/cr/rtc_pi05_release` |
| 已有机器人客户端 | `/home/agilex/wwq/client_vla` |

建议将本仓库另外克隆到下面的位置，保留上述目录：

```text
GPU：  /root/kube-user/ns100002-chenrui/cr/piper-cable-rtc
机器人：/home/agilex/wwq/piper-cable-rtc
```

新入口分别为 `server/openpi/deployment/start_rtc.sh` 和 `robot_client/pi0.5_rtc_control_v3.sh`。

已有权重继续原地引用，不上传 GitHub：

```text
/root/kube-user/ns100002-chenrui/cr/tqq/openpi_cobot_magic_work/checkpoints/cobot_magic_finetune/pi05_cobot_magic_plug_cable_finetune/plug_cable_50k_bs32_bf16_full/49999
```

对应归一化统计的 assets 基目录：

```text
/root/kube-user/ns100002-chenrui/cr/tqq/openpi_cobot_magic_work/assets
```

程序会在该基目录下寻找 `pi05_cobot_magic_plug_cable_finetune/cobot_magic_plug_cable/norm_stats.json`。统计必须对应当前权重；仅有权重路径还不足以启动本服务。

## 3. 首次发布到 GitHub

本地整理目录是独立的新仓库，旧仓库的 `.git` 不随源码复制。先在 GitHub 网页新建空仓库 `piper-cable-rtc`，用实际用户名替换 `<GitHub用户名>`。在本地新项目根目录执行：

```bash
git status
git log -1 --oneline
git remote add origin https://github.com/<GitHub用户名>/piper-cable-rtc.git
git push -u origin main
git push origin v0.1.0
```

这是操作者后续执行的发布命令，本轮没有推送。新仓库从当前源码快照开始，过往版本只保留第 7 节的说明。

## 4. 两端首次部署

以下为 Linux 命令。克隆地址中的 `<GitHub用户名>` 需要替换，`v0.1.0` 标签需要已经推送。

### GPU 服务器

```bash
cd /root/kube-user/ns100002-chenrui/cr
git clone https://github.com/<GitHub用户名>/piper-cable-rtc.git
cd piper-cable-rtc
git switch -c deploy/server-v0.1.0 v0.1.0
cp configs/server.env.example configs/server.env
```

编辑 `configs/server.env`，确认 `RTC_PYTHON`、`RTC_CHECKPOINT`、`RTC_ASSETS_BASE_DIR` 指向实际文件。示例对应之前使用的服务器环境。启动脚本自动读取此文件；可通过 `RTC_ENV_FILE=/绝对路径/server.env` 指定其他文件。

**现有服务器优先复用已有 `pi0.5_wq` 环境，不在部署时升级依赖。** 本工程声明 Python >=3.11，沿用已有 B200 的 CUDA/JAX 配置。

全新机器可在 `server/openpi` 中按锁文件安装：

```bash
cd server/openpi
uv python install 3.11
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

然后将 `RTC_PYTHON` 设置为该 `.venv/bin/python` 的绝对路径。这些命令依据随附配置整理，本轮未重新安装或验证新 B200 环境；驱动/CUDA 仍需匹配 JAX，微调权重和统计需另行准备。

### 机器人

```bash
cd /home/agilex/wwq
git clone https://github.com/<GitHub用户名>/piper-cable-rtc.git
cd piper-cable-rtc
git switch -c deploy/robot-v0.1.0 v0.1.0
cp configs/robot.env.example configs/robot.env
```

编辑 `configs/robot.env`，确认已有 Conda、ROS Noetic 和可选驱动工作区的初始化路径。默认环境为 `wwq_aloha`。原机器人在 Python 3.8.20 上运行过模拟测试；客户端不使用服务器的 Python 环境。

机器人需要 `rospy`、`sensor_msgs`、`std_msgs`、`cv_bridge`、NumPy、OpenCV、msgpack 和 websocket-client。ROS 依赖沿用已有环境；缺少网络依赖时，在已激活的机器人环境中执行：

```bash
python -m pip install -r robot_client/requirements.txt
```

启动脚本按自身位置定位 Python 文件，不再依赖旧的 `client_vla` 目录名。

## 5. 每次实验的完整启动顺序

### ① GPU 容器终端启动 RTC 服务

```bash
bash /root/kube-user/ns100002-chenrui/cr/piper-cable-rtc/server/openpi/deployment/start_rtc.sh \
  --host 127.0.0.1 --port 8000 --smoothing-alpha 1.0
```

等待模型加载、普通及 RTC 路径预热完成，出现 `server listening on 127.0.0.1:8000`。保持终端运行；同一端口只运行一个服务。

`--checkpoint` 和 `--assets-base-dir` 可覆盖环境配置。脚本优先加载本次克隆的源码，避免引用旧工程。服务程序默认端口仍为 8001，所以这里显式指定 8000。

### ② 登录机器人并建立隧道

Windows CMD 或 PowerShell 登录：

```bash
ssh agilex@172.16.10.80
```

登录后，在**机器人终端**建立 GPU 连接隧道：

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  -p 30176 tqq@172.16.41.254
```

此命令对应既有网络与容器映射，地址/映射改变时应调整。按提示输入服务器 SSH 密码，项目不保存密码。`-N` 隧道认证后没有新输出属于正常现象，保持窗口打开。

### ③ 另一个机器人终端检查连接

```bash
curl -i --max-time 5 http://127.0.0.1:18000/healthz
```

预期 HTTP `200 OK`，正文可以为空。服务器日志中的 `connection rejected (200 OK)` 可能来自此 HTTP 健康检查，本身不代表推理失败。

链路为 `机器人 18000 → SSH 隧道 → GPU 容器 8000`。出现 `Connection refused` 时检查服务、隧道和客户端 URL。

### ④ 启动原有驱动和三路相机

沿用实验室原有流程。本仓库不包含已核验的驱动/相机启动配置，不提供猜测的 roslaunch 命令。默认话题：

```text
/puppet/joint_left
/puppet/joint_right
/camera_f/color/image_raw
/camera_l/color/image_raw
/camera_r/color/image_raw
```

### ⑤ 操作者启动机器人 v3 客户端

```bash
bash /home/agilex/wwq/piper-cable-rtc/robot_client/pi0.5_rtc_control_v3.sh \
  --url ws://127.0.0.1:18000/ \
  --move-duration 2 \
  --diagnostics-dir /home/agilex/wwq/piper-cable-rtc/robot_client/diagnostics \
  --diagnostics-label rtc_v3_linear_90hz_01
```

`--move-duration 2` 是前往 A/B 的规划时长，应沿用现场已确认的设置。默认 A/B 是原任务固定点位；更换工装/布局时需要核对，可用 `--pose-a` / `--pose-b` 提供各 14 维目标。

| 按键 | 行为 |
|---|---|
| `i` | 取消推理动作，前往 A |
| `s` | 确认到达 A 后开始推理 |
| `e` / `Ctrl+C` | 前往 B 后退出，**不是急停** |
| `1` / `2` / `3` | 仅记录自由空间、对准、接触标记 |
| `4` / `5` | 仅记录成功、失败标记，不停止运动 |

网络错误、队列耗尽或插值调度异常时暂停并保持最后目标，不自动重放旧动作。处理后按 `i → 到达 A → s` 开始新一轮。

正常停止顺序：**客户端按原流程退出 → 停止 SSH 隧道 → 停止推理服务**。诊断目录每次独立生成 `manifest.json`、`events.jsonl`、`summary.json`，正常退出后下载完整目录。

## 6. 验证、修改与回退

机器人模拟测试，从项目根目录执行：

```bash
cd robot_client
python -W error::ResourceWarning -m unittest test_rtc_client test_diagnostics test_rtc_v2 test_action_interpolation -v
```

返回项目根目录，运行本次整理的检查（需要可用 Bash）：

```bash
python -m unittest discover -s tests -v
```

以上测试不创建真实 ROS 发布者。服务器 CPU 测试需要完整服务器环境，在 `server/openpi` 执行：

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/scripts"
JAX_PLATFORMS=cpu /root/miniconda3/envs/pi0.5_wq/bin/python \
  -m unittest discover -s tests -p 'test_rtc_*.py' -v
```

本轮未访问 GPU 服务器或进行真机联调。通过项及未验证项见 [验证报告](experiments/reports/validation.md)。历史开环回放不能证明真机成功率改善。

后续修改在本地开发分支进行：

```bash
git switch main
git switch -c feature/你的改动名称
# 修改、验证后
git add 具体文件
git commit -m "说明本次改动"
git push -u origin feature/你的改动名称
```

合并并验证后打新标签。设备端先退出客户端、确认 `git status` 没有未提交源码改动，再 `git fetch origin --tags`，从目标标签新建部署分支。本机 `configs/*.env` 不被 Git 跟踪。

新仓库没有旧版本提交，不能在这里切到以前 v1/v2。需要旧源码时使用原仓库/备份。2026-10-19 原机器人仓库的已知分支：

```bash
cd /home/agilex/wwq/client_vla
git status
git switch backup/robot-v2-before-v3-20261019  # 当时实际安装的 v2
# 或回到原仓库的 v3
git switch rtc/client-v3-interpolation
```

以上只适用于旧目录，切换前确认现场仍保留对应分支。新入口可加 `--action-interpolation none --action-rate 30 --publish-rate 30` 做关闭插值的参数对照，但不会还原历史源码。

## 7. 过往版本说明

| 阶段 | 主要行为 | 已有证据与限制 |
|---|---|---|
| 原同步版本 | 每块输出 50 步，执行前 30 步后请求下一块 | 推理等待可能停顿，跨块可能跳变 |
| RTC v1 | 推理/执行异步，前缀 guidance；每 30 步请求；可选 EMA | 操作者反馈成功率有所改善，但仍抖动，EMA 可能降低成功率 |
| 诊断增强 | 记录模型输出、发布尝试、反馈及阶段标记 | 历史常规发布周期稳定，换块目标突变突出；出现过约 1.06 秒响应长尾 |
| RTC v2（2026-10-16） | 重规划 30→15 步，6 步局部过渡，修正计划回传前缀 | 41 项模拟测试通过，235 个历史窗口目标差分指标改善，属于开环证据 |
| RTC v3（2026-10-19） | 继承 v2，30 Hz 模型/90 Hz 发布插值，预留当前插值终点 | 67 项模拟测试、584 次历史回放通过；有旧机器人 Git 部署记录，90 Hz 真机改善未证实 |
| 本仓库 0.1.0（2026-10-29） | 当前 v3 与 RTC 服务统一整理，环境和路径可配置 | 本轮结果见报告，未替换设备上的旧目录 |

源码基线为机器人 `0a95d65be64cbac9dfff702ee927ae772c534253`、服务器 `183f4ba94d54e6c1c674ac0a90b7382a355db628`。这些是原本地仓库的溯源提交，不作为旧历史上传本仓库。

## 8. 本次整理的代码变化

- v3 脚本直接包含当前参数，不依赖旧版脚本；队列、插值器和诊断算法保留。
- 客户端按脚本目录启动，录像默认路径随目录变化；支持已有 Conda/ROS 环境配置。
- 服务支持 checkpoint、assets 基目录配置；模型和推理核心逻辑保留。
- 移除依赖旧 Git 提交的 baseline 回归入口，以及旧部署快照、旧脚本和编辑器/上游 CI 配置。
- 新增来源清单和验证记录，保留第三方许可证，见 [来源说明](THIRD_PARTY_NOTICES.md)。
