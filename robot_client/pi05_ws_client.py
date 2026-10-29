#!/usr/bin/env python3
"""通过 ROS 调用远端 OpenPI pi0.5，并支持键盘控制机械臂状态。"""

import argparse
import collections
import os
import select
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import msgpack
import numpy as np
import rospy
import websocket
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from rtc_action_queue import RTCActionQueue, QueueExhausted
from diagnostic_recorder import DiagnosticRecorder, NO_DIAGNOSTICS
from action_interpolator import LinearActionInterpolator, InterpolationTimingError


POLICY_URL = "ws://127.0.0.1:18000/"
PROMPT = "Insert the ethernet cable into the port."
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_CACHE_SIZE = 5
IMAGE_SYNC_TOLERANCE = 0.025

# A/B 均为 14 维：[左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪]。
# A为推理起始点位（键盘i控制），B为结束推理点位（键盘e控制）
PRESET_A = [
    -0.59310364, 0.77965511, -0.92053992, -0.24688176,
    1.08487739, 0.30557981, 0.05324802,
    0.55613620, 0.77440778, -0.90639485, 0.07693020,
    1.00660213, -0.08657428, 0.05196314,
]
PRESET_B = [
    -0.064, -0.022, 0.022, 0.069, 0.336, 0.099, 0.020,
    -0.019, -0.019, 0.016, 0.059, 0.364, 0.006, 0.054,
]


class ControlState:
    WAITING = "等待指令"
    MOVING_A = "正在前往 A"
    AT_A = "已到达 A"
    POLICY = "正在执行推理"
    POLICY_HOLD = "推理已暂停，保持最后目标"
    MOVING_B = "正在前往 B"
    AT_B = "已到达 B"



def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(value):
    return msgpack.Packer(default=_pack_array).pack(value)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=_unpack_array)


def validate_pose(name, value):
    """检查预设关节位置"""
    if value is None:
        raise ValueError(
            f"{name} 未设置。需填写 PRESET_{name}，"
            f"或通过 --pose-{name.lower()} 传入 14 个数值。"
        )
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape != (14,):
        raise ValueError(f"{name} 必须包含 14 个数值，当前形状为 {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} 包含 NaN 或无穷大")
    return pose


def resize_with_pad(image, size=224):
    """（处理头部相机）保持宽高比缩放 RGB 图像，并用黑色填充到指定尺寸。"""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    output[top:top + new_height, left:left + new_width] = resized
    return output


def preprocess_wrist(image):
    """（处理腕部相机）腕部图像先中心裁成正方形，再缩放到 224。"""
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return resize_with_pad(image[top:top + side, left:left + side])


class HeadVideoRecorder:
    """在独立线程中录制头部相机，避免视频编码阻塞 ROS 回调。"""

    def __init__(self, output_dir, fps, queue_size=30):
        self.output_dir = output_dir
        self.fps = fps
        self.frames = collections.deque(maxlen=queue_size)
        self.condition = threading.Condition()
        self.active = False
        self.shutdown = False
        self.session = 0
        self.output_path = None
        self.dropped_frames = 0
        self.thread = threading.Thread(target=self._run, name="head_video_recorder", daemon=True)
        self.thread.start()

    def is_active(self):
        with self.condition:
            return self.active

    def start(self):
        with self.condition:
            if self.active:
                return
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except OSError as exc:
                rospy.logerr("无法创建录像目录 %s：%s；本次推理继续运行但不录像", self.output_dir, exc)
                return
            now = time.time()
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            milliseconds = int((now % 1) * 1000)
            self.output_path = os.path.join(
                self.output_dir, f"inference_{timestamp}_{milliseconds:03d}.mp4"
            )
            self.frames.clear()
            self.dropped_frames = 0
            self.session += 1
            self.active = True
            self.condition.notify_all()

    def add_frame(self, rgb_image):
        with self.condition:
            if not self.active:
                return
            if len(self.frames) == self.frames.maxlen:
                self.frames.popleft()
                self.dropped_frames += 1
            self.frames.append(rgb_image.copy())
            self.condition.notify()

    def stop(self):
        with self.condition:
            if not self.active:
                return
            self.active = False
            self.frames.clear()
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.active = False
            self.frames.clear()
            self.shutdown = True
            self.condition.notify_all()
        self.thread.join(timeout=3.0)

    def _run(self):
        writer = None
        writer_session = -1
        written_frames = 0
        output_path = None

        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.shutdown
                    or self.session != writer_session
                    or not self.active
                    or bool(self.frames)
                )
                if self.shutdown:
                    break

                active = self.active
                session = self.session
                current_path = self.output_path
                dropped_frames = self.dropped_frames
                frame = self.frames.popleft() if active and self.frames else None

            if session != writer_session:
                if writer is not None:
                    writer.release()
                    rospy.loginfo("头部相机录像已保存：%s，共 %d 帧", output_path, written_frames)
                writer = None
                writer_session = session
                written_frames = 0
                output_path = current_path

            if not active:
                if writer is not None:
                    writer.release()
                    rospy.loginfo(
                        "头部相机录像已保存：%s，共 %d 帧，丢弃 %d 帧",
                        output_path, written_frames, dropped_frames,
                    )
                    writer = None
                with self.condition:
                    self.condition.wait_for(lambda: self.shutdown or self.active)
                continue

            if frame is None:
                continue

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    rospy.logerr("无法创建头部相机录像文件：%s；本次推理继续运行但不录像", output_path)
                    writer.release()
                    writer = None
                    with self.condition:
                        if self.session == session:
                            self.active = False
                            self.frames.clear()
                    continue
                rospy.loginfo("开始录制头部相机：%s", output_path)

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written_frames += 1

        if writer is not None:
            writer.release()
            rospy.loginfo("头部相机录像已保存：%s，共 %d 帧", output_path, written_frames)


class RosObservation:
    """保存最新关节反馈，并在推理会话中缓存按时间戳对齐的相机消息。"""

    def __init__(self, args):
        self.bridge = CvBridge()
        self.diagnostics = getattr(args, "_diagnostics", NO_DIAGNOSTICS)
        self.video_recorder = (
            HeadVideoRecorder(args.record_dir, args.video_fps)
            if args.record_video else None
        )
        self.lock = threading.Lock()
        self.image_queues = {
            name: collections.deque(maxlen=IMAGE_CACHE_SIZE)
            for name in CAMERA_NAMES
        }
        self.last_observation_stamp = None
        self.left = None
        self.right = None
        self.left_seq = 0
        self.right_seq = 0
        self.feedback_times = {"left": {}, "right": {}}
        self.capture_enabled = False

        rospy.Subscriber(args.img_front_topic, Image, self._image_callback, "cam_high", queue_size=1)
        rospy.Subscriber(args.img_left_topic, Image, self._image_callback, "cam_left_wrist", queue_size=1)
        rospy.Subscriber(args.img_right_topic, Image, self._image_callback, "cam_right_wrist", queue_size=1)
        rospy.Subscriber(args.puppet_arm_left_topic, JointState, self._left_callback, queue_size=1)
        rospy.Subscriber(args.puppet_arm_right_topic, JointState, self._right_callback, queue_size=1)

    def _image_callback(self, msg, name):
        with self.lock:
            capture_enabled = self.capture_enabled
        record_frame = (
            name == "cam_high"
            and self.video_recorder is not None
            and self.video_recorder.is_active()
        )
        if not capture_enabled and not record_frame:
            return

        if record_frame:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.video_recorder.add_frame(image)
        if capture_enabled:
            # 推理期间只缓存原始 ROS 消息；真正请求时才转换三张图，避免持续转换占用 CPU。
            stamp = msg.header.stamp.to_sec()
            if stamp <= 0:
                stamp = rospy.get_time()
            with self.lock:
                if self.capture_enabled:
                    self.image_queues[name].append((stamp, msg))

    def _left_callback(self, msg):
        received_ns = time.monotonic_ns()
        with self.lock:
            self.left = np.asarray(msg.position, dtype=np.float32)
            self.left_seq += 1
            self._record_feedback("left", msg, self.left_seq, received_ns)

    def _right_callback(self, msg):
        received_ns = time.monotonic_ns()
        with self.lock:
            self.right = np.asarray(msg.position, dtype=np.float32)
            self.right_seq += 1
            self._record_feedback("right", msg, self.right_seq, received_ns)

    def _record_feedback(self, side, msg, sequence, received_ns):
        """Called with observation lock; only a bounded memory enqueue, no disk I/O."""
        if not self.diagnostics.enabled:
            return
        stamp = msg.header.stamp.to_sec()
        timing = {"receive_monotonic_ns": received_ns, "ros_stamp_s": stamp,
                  "header_sequence": msg.header.seq, "feedback_sequence": sequence}
        self.feedback_times[side] = timing
        self.diagnostics.emit("feedback", side=side, **timing,
                              names=list(msg.name), position=list(msg.position),
                              velocity=list(msg.velocity), effort=list(msg.effort))

    def start_capture(self):
        """清除旧图像，保证第一次推理使用按下 s 之后的新帧。"""
        with self.lock:
            for queue in self.image_queues.values():
                queue.clear()
            self.last_observation_stamp = None
            self.capture_enabled = True
        rospy.loginfo("已开始采集推理所需的观测")

    def stop_capture(self):
        with self.lock:
            self.capture_enabled = False
            for queue in self.image_queues.values():
                queue.clear()
            self.last_observation_stamp = None

    def start_video_recording(self):
        if self.video_recorder is not None:
            self.video_recorder.start()

    def stop_video_recording(self):
        if self.video_recorder is not None:
            self.video_recorder.stop()

    def close(self):
        self.stop_capture()
        if self.video_recorder is not None:
            self.video_recorder.close()

    def get_qpos(self):
        """返回当前从臂 14 维真实关节位置。"""
        sample = self.get_qpos_with_version()
        return None if sample is None else sample[0]

    def get_qpos_with_version(self):
        """返回关节位置，以及左右臂各自的反馈序号。"""
        with self.lock:
            if self.left is None or self.right is None:
                return None
            if len(self.left) != 7 or len(self.right) != 7:
                return None
            qpos = np.concatenate((self.left, self.right)).astype(np.float32, copy=True)
            return qpos, (self.left_seq, self.right_seq)

    def get(self, request_snapshot=None):
        """选取时间戳最接近的三路新图像，并与最新关节反馈构造观测。"""
        with self.lock:
            if not self.capture_enabled:
                return None
            if (self.left is None or self.right is None
                    or any(not self.image_queues[name] for name in CAMERA_NAMES)):
                return None
            if len(self.left) != 7 or len(self.right) != 7:
                raise ValueError("左右从臂反馈均应为 7 维（6 个关节和 1 个夹爪）")

            # 以三路最新时间戳中的最小值为共同基准，各自选择最接近的一帧。
            frame_time = min(self.image_queues[name][-1][0] for name in CAMERA_NAMES)
            if (self.last_observation_stamp is not None
                    and frame_time <= self.last_observation_stamp):
                return None

            selected = {}
            selected_stamps = []
            for name in CAMERA_NAMES:
                stamp, msg = min(
                    self.image_queues[name],
                    key=lambda item: abs(item[0] - frame_time),
                )
                if abs(stamp - frame_time) > IMAGE_SYNC_TOLERANCE:
                    return None
                selected[name] = msg
                selected_stamps.append(stamp)

            state = np.concatenate((self.left, self.right)).astype(np.float32, copy=False)
            self.last_observation_stamp = frame_time
            image_skew_ms = (max(selected_stamps) - min(selected_stamps)) * 1000
            diagnostics = getattr(self, "diagnostics", NO_DIAGNOSTICS)
            if diagnostics.enabled:
                selection_ns = time.monotonic_ns()
                selection_ros_s = rospy.get_time()
                feedback_times = {side: dict(value) for side, value in self.feedback_times.items()}

        # 在图像处理前对齐控制 tick 和旧轨迹前缀。此时观测锁已经释放，
        # 避免与控制线程的 controller -> observation 锁顺序相互等待。
        contract = request_snapshot() if request_snapshot is not None else None
        if request_snapshot is not None and contract is None:
            return None

        # 图像转换和缩放放在锁外执行，避免阻塞相机与关节反馈回调。
        images = {
            name: self.bridge.imgmsg_to_cv2(selected[name], desired_encoding="rgb8")
            for name in CAMERA_NAMES
        }
        rospy.loginfo("三路推理图像最大时间差：%.1f ms", image_skew_ms)

        result = {
            "state": state,
            "images": {
                "cam_high": resize_with_pad(images["cam_high"]),
                "cam_left_wrist": preprocess_wrist(images["cam_left_wrist"]),
                "cam_right_wrist": preprocess_wrist(images["cam_right_wrist"]),
            },
            "prompt": PROMPT,
        }
        if contract is not None:
            result["rtc"] = contract
        if diagnostics.enabled:
            diagnostics.emit("observation", selection_monotonic_ns=selection_ns,
                             selection_ros_time_s=selection_ros_s,
                             state=state, feedback_times=feedback_times,
                             image_effective_stamps_s=dict(zip(CAMERA_NAMES, selected_stamps)),
                             image_header_stamps_s={name: selected[name].header.stamp.to_sec()
                                                    for name in CAMERA_NAMES},
                             image_skew_ms=image_skew_ms, rtc=contract)
        return result


class ArmPublisher:
    """向左右 master 话题发布一个 14 维关节目标。"""

    JOINT_NAMES = [f"joint{i}" for i in range(7)]

    def __init__(self, args):
        self.left_pub = rospy.Publisher(args.left_cmd_topic, JointState, queue_size=1)
        self.right_pub = rospy.Publisher(args.right_cmd_topic, JointState, queue_size=1)

    def publish(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,):
            raise ValueError(f"关节目标应为 14 维，当前形状为 {action.shape}")

        stamp = rospy.Time.now()
        left_message = JointState(
            header=Header(stamp=stamp),
            name=self.JOINT_NAMES,
            position=action[:7].tolist(),
        )
        right_message = JointState(
            header=Header(stamp=stamp),
            name=self.JOINT_NAMES,
            position=action[7:].tolist(),
        )
        self.left_pub.publish(left_message)
        self.right_pub.publish(right_message)
        return stamp.to_sec()


class ArmController:
    """固定频率发布动作，并按 e > i > 推理动作处理请求。"""

    def __init__(self, args, observation, publisher, pose_a, pose_b):
        self.args = args
        self.observation = observation
        self.publisher = publisher
        self.pose_a = pose_a
        self.pose_b = pose_b
        self.diagnostics = getattr(args, "_diagnostics", NO_DIAGNOSTICS)

        self.lock = threading.Lock()
        self.state = ControlState.WAITING
        self.target = None
        self.trajectory = None
        self.trajectory_index = 0
        self.stable_count = 0
        self.last_feedback_version = (0, 0)
        self.actions = collections.deque()
        self.epoch = 0
        self.policy_chunk_index = 0
        self.last_policy_action_time = None
        self.rtc_enabled = args.rtc_mode != "legacy"
        self.interpolation_enabled = args.action_interpolation == "linear"
        self.action_rate = args.action_rate or args.publish_rate
        self.interpolator = LinearActionInterpolator(self.action_rate)
        self.arrival_stable_cycles = (int(np.ceil(args.stable_cycles * args.publish_rate / self.action_rate))
                                      if self.interpolation_enabled else args.stable_cycles)
        self.rtc_queue = RTCActionQueue(transition_steps=args.rtc_transition_steps)
        self.rtc_replan_steps = args.chunk_steps or 30
        self._reset_rtc_locked()

        self.go_a_requested = False
        self.start_requested = False
        self.exit_requested = False

        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.at_b_event = threading.Event()
        self.error_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="arm_control", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready_event.wait(timeout=0.05):
                return True
            if self.error_event.is_set():
                return False
        return False

    def wait_at_b(self, timeout):
        return self.at_b_event.wait(timeout)

    def get_error(self):
        with self.lock:
            return self.error

    def request_go_a(self):
        with self.lock:
            if self.exit_requested:
                return False
            self.go_a_requested = True
            self.start_requested = False
            self.actions.clear()
            self.epoch += 1
            self._reset_rtc_locked()
        rospy.loginfo("收到 i：取消推理动作，准备前往预设位置 A")
        return True

    def request_start(self):
        with self.lock:
            if self.exit_requested:
                rospy.logwarn("正在退出，忽略 s")
                return False
            if self.state != ControlState.AT_A:
                rospy.logwarn("只有确认到达 A 后才能按 s 开始推理，当前状态：%s", self.state)
                return False
            self.start_requested = True
        rospy.loginfo("收到 s：准备采集新观测并开始推理")
        return True

    def request_exit(self, source="e"):
        with self.lock:
            if self.exit_requested:
                return False
            self.exit_requested = True
            self.go_a_requested = False
            self.start_requested = False
            self.actions.clear()
            self.epoch += 1
            self._reset_rtc_locked()
        rospy.loginfo("收到 %s：取消当前动作，准备前往预设位置 B 后退出", source)
        return True

    def policy_request(self):
        """动作队列为空时返回当前推理代次，否则返回 None。"""
        with self.lock:
            if self.rtc_enabled:
                if (self.state == ControlState.POLICY and not self.go_a_requested
                        and not self.exit_requested and not self.rtc_request_pending
                        and (not self.rtc_bootstrapped or self.rtc_queue.tick >= self.next_rtc_request_tick)):
                    return self.epoch
                return None
            if (self.state == ControlState.POLICY and not self.actions
                    and not self.go_a_requested and not self.exit_requested):
                return self.epoch
            return None

    def _reset_rtc_locked(self):
        """调用者持有控制锁，或尚未启动线程。"""
        self.rtc_queue.reset()
        self.rtc_bootstrapped = False
        self.rtc_request_pending = False
        self.next_rtc_request_tick = 0
        self._diagnostic_sources = collections.deque()
        self._diagnostic_source = None
        self.interpolator.reset()
        self._interpolation_start = None
        self._interpolation_end = None
        self._interpolation_phase = None
        self._interpolation_end_source = None

    def _pause_rtc_locked(self, reason):
        # 网络错误/前缀耗尽时停止消费推理目标，保持最后已发送目标。
        # 不自动规划前往 A/B 的运动；用户仍可显式按 i/e。
        self.diagnostics.emit("policy_paused", reason=reason, epoch=self.epoch,
                              session_id=self.rtc_queue.session_id, tick=self.rtc_queue.tick)
        self.epoch += 1
        self._reset_rtc_locked()
        self.state = ControlState.POLICY_HOLD
        self.observation.stop_capture()
        self.observation.stop_video_recording()
        rospy.logerr("RTC 已暂停：%s；保持最后目标，按 i 返回 A 或按 e 前往 B", reason)

    def pause_rtc(self, reason, epoch):
        with self.lock:
            if self.state == ControlState.POLICY and self.epoch == epoch:
                self._pause_rtc_locked(reason)

    def begin_rtc_request(self, epoch):
        with self.lock:
            if (self.state != ControlState.POLICY or self.epoch != epoch
                    or self.go_a_requested or self.exit_requested or self.rtc_request_pending):
                return None
            if self.rtc_bootstrapped and self.rtc_queue.tick < self.next_rtc_request_tick:
                return None
            contract = self.rtc_queue.begin(
                mode=self.args.rtc_mode, delay=self.args.rtc_delay,
                prefix_attention_horizon=self.args.rtc_prefix_attention_horizon or None)
            contract["smoothing_alpha"] = self.args.rtc_smoothing_alpha
            self.rtc_request_pending = True
            self.next_rtc_request_tick = contract["observation_tick"] + self.rtc_replan_steps
            if self.diagnostics.enabled:
                self.diagnostics.emit("rtc_snapshot", epoch=epoch, rtc=contract,
                                      queue_remaining=self.rtc_queue.remaining,
                                      interpolation_phase=self._interpolation_phase,
                                      action_rate=self.action_rate)
            return contract

    def discard_initial_rtc_result(self, epoch):
        with self.lock:
            if (self.state != ControlState.POLICY or self.epoch != epoch
                    or self.go_a_requested or self.exit_requested):
                return False
            self.diagnostics.emit("response_discarded", reason="first_chunk", epoch=epoch,
                                  session_id=self.rtc_queue.session_id)
            self._reset_rtc_locked()
            return True

    def submit_rtc_result(self, result, epoch):
        with self.lock:
            if (self.state != ControlState.POLICY or self.epoch != epoch
                    or self.go_a_requested or self.exit_requested):
                return False
            if not self.rtc_queue.merge(result):
                raise ValueError("RTC 返回的 session/request/tick 与在途请求不一致")
            self.rtc_request_pending = False
            self.rtc_bootstrapped = True
            self.policy_chunk_index += 1
            info = dict(self.rtc_queue.last_merge)
            remaining = self.rtc_queue.remaining
            chunk = self.policy_chunk_index
            if self.diagnostics.enabled:
                identity = result["rtc_info"]
                preserve = info["preserved_old_steps"]
                old_sources = list(self._diagnostic_sources)[:preserve]
                start = info["actual_delay"] + preserve
                new_sources = [{"session_id": identity["session_id"],
                                "request_id": identity["request_id"],
                                "observation_tick": identity["observation_tick"],
                                "action_index": index, "chunk_index": chunk,
                                "transition_applied": index - start < info["transition_steps"]}
                               for index in range(start, 50)]
                self._diagnostic_sources = collections.deque(old_sources + new_sources)
                self.diagnostics.emit("chunk_merged", epoch=epoch, rtc_info=identity,
                                      merge=info, queue_tick=self.rtc_queue.tick,
                                      queue_remaining=remaining, chunk_index=chunk,
                                      planned_targets=self.rtc_queue.targets())
        rospy.loginfo("RTC chunk#%d：实际返回延迟=%d步，预算=%d步，保留旧前缀=%d步，剩余=%d步，超预算=%s",
                      chunk, info["actual_delay"], info["budget"], info["preserved_old_steps"],
                      remaining, info["budget_overrun"])
        if info["transition_steps"]:
            rospy.loginfo("RTC 局部过渡=%d步，首步最大关节差 %.6f -> %.6f；夹爪不变",
                          info["transition_steps"], info["raw_join_max_joint_step"],
                          info["applied_join_max_joint_step"])
        return True

    def record_marker(self, label):
        """Annotation only: this method never changes a motion or state."""
        if not self.diagnostics.enabled:
            return
        with self.lock:
            self.diagnostics.emit("marker", label=label, state=self.state, epoch=self.epoch,
                                  session_id=self.rtc_queue.session_id, tick=self.rtc_queue.tick)
        rospy.loginfo("诊断标记：%s", label)

    def policy_epoch_is_current(self, epoch):
        """判断某次推理返回时，当前是否仍处于同一轮推理会话。"""
        with self.lock:
            return (self.state == ControlState.POLICY
                    and self.epoch == epoch
                    and not self.go_a_requested
                    and not self.exit_requested)

    def submit_actions(self, actions, epoch):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"推理动作应为 (N, 14)，当前形状为 {actions.shape}")
        if len(actions) == 0:
            raise ValueError("推理动作 chunk 为空")
        if not np.all(np.isfinite(actions)):
            raise ValueError("推理动作包含 NaN 或无穷大")

        current_qpos = self.observation.get_qpos()
        applied_actions, smooth_steps, smooth_reason = self._smooth_chunk_boundary(
            actions, current_qpos
        )
        with self.lock:
            if self.state != ControlState.POLICY or self.epoch != epoch or self.exit_requested:
                return False
            previous_target = self.target.copy()
            gap_ms = None
            if self.last_policy_action_time is not None:
                gap_ms = (time.monotonic() - self.last_policy_action_time) * 1000
            self.actions.extend(applied_actions)
            self.policy_chunk_index += 1
            chunk_index = self.policy_chunk_index
            if self.diagnostics.enabled:
                self._diagnostic_sources.extend(
                    {"chunk_index": chunk_index, "action_index": index, "legacy": True}
                    for index in range(len(applied_actions)))
                self.diagnostics.emit("legacy_chunk_merged", epoch=epoch, chunk_index=chunk_index,
                                      selected_actions=actions, applied_actions=applied_actions,
                                      smooth_steps=smooth_steps)

        self._log_action_continuity(
            chunk_index, previous_target, current_qpos, actions,
            applied_actions, smooth_steps, smooth_reason, gap_ms,
        )
        return True

    def _smooth_chunk_boundary(self, actions, current_qpos):
        """仅平滑新 chunk 开头的 12 个机械臂关节，夹爪动作保持不变。"""
        smoothed = actions.copy()
        steps = min(self.args.boundary_smooth_steps, len(actions))
        if current_qpos is None or len(current_qpos) != 14:
            return smoothed, 0, "真实关节反馈不可用"
        if steps <= 0:
            return smoothed, 0, "参数已关闭"

        joint_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        max_start_delta = float(np.max(
            np.abs(actions[0, joint_indices] - current_qpos[joint_indices])
        ))
        # float32 中的 0.01 可能略小于十进制 0.01，留极小容差保证“达到阈值”会启用。
        if max_start_delta < self.args.boundary_smooth_threshold - 1e-8:
            return (
                smoothed,
                0,
                f"最大关节差 {max_start_delta:.5f} rad 小于阈值 "
                f"{self.args.boundary_smooth_threshold:.5f} rad",
            )

        # 逐步消除“当前真实位置到模型首步”的偏移；最后一步完全恢复模型原轨迹。
        phase = np.arange(1, steps + 1, dtype=np.float32) / steps
        weights = 0.5 * (1.0 - np.cos(np.pi * phase))
        initial_offset = current_qpos[joint_indices] - actions[0, joint_indices]
        smoothed[:steps, joint_indices] = (
            actions[:steps, joint_indices]
            + (1.0 - weights[:, None]) * initial_offset[None, :]
        )
        return smoothed, steps, ""

    @staticmethod
    def _log_action_continuity(
            chunk_index, previous_target, current_qpos, actions,
            applied_actions, smooth_steps, smooth_reason, gap_ms):
        """每个 chunk 输出一条动作连续性日志，用于定位抖动发生位置。"""
        joint_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        gripper_indices = np.array([6, 13])

        def action_name(index):
            side = "左" if index < 7 else "右"
            local_index = index if index < 7 else index - 7
            return f"{side}夹爪" if local_index == 6 else f"{side}joint{local_index}"

        def max_delta_info(delta, indices):
            local_index = int(np.argmax(delta[indices]))
            action_index = int(indices[local_index])
            return float(delta[action_index]), action_name(action_index)

        boundary_delta = np.abs(actions[0] - previous_target)
        boundary_joint, boundary_joint_name = max_delta_info(boundary_delta, joint_indices)
        boundary_gripper, boundary_gripper_name = max_delta_info(boundary_delta, gripper_indices)

        if current_qpos is not None and len(current_qpos) == 14:
            first_to_actual = np.abs(actions[0] - current_qpos)
            previous_to_actual = np.abs(previous_target - current_qpos)
            actual_joint, actual_joint_name = max_delta_info(first_to_actual, joint_indices)
            actual_gripper, actual_gripper_name = max_delta_info(first_to_actual, gripper_indices)
            tracking_joint, tracking_joint_name = max_delta_info(previous_to_actual, joint_indices)
            tracking_gripper, tracking_gripper_name = max_delta_info(previous_to_actual, gripper_indices)
            feedback_text = (
                f"新首步距实际：关节={actual_joint:.5f} rad({actual_joint_name})、"
                f"夹爪={actual_gripper:.5f} m({actual_gripper_name})；"
                f"旧末端命令距实际：关节={tracking_joint:.5f} rad({tracking_joint_name})、"
                f"夹爪={tracking_gripper:.5f} m({tracking_gripper_name})"
            )
        else:
            feedback_text = "真实反馈不可用"

        if current_qpos is not None and len(current_qpos) == 14 and smooth_steps > 0:
            applied_first_delta = np.abs(applied_actions[0] - current_qpos)
            applied_joint, applied_joint_name = max_delta_info(
                applied_first_delta, joint_indices
            )
            smooth_path = np.vstack((current_qpos, applied_actions[:smooth_steps]))
            smooth_step_delta = np.abs(np.diff(smooth_path, axis=0))[:, joint_indices]
            smooth_max_step = float(np.max(smooth_step_delta))
            smooth_text = (
                f"关节余弦平滑={smooth_steps}步，平滑后首步距实际="
                f"{applied_joint:.5f} rad({applied_joint_name})，"
                f"平滑段最大单步={smooth_max_step:.5f} rad，夹爪未平滑"
            )
        else:
            smooth_text = f"关节边界平滑未应用（{smooth_reason}），夹爪未平滑"

        if len(actions) > 1:
            signed_step_delta = np.diff(actions, axis=0)
            step_delta = np.abs(signed_step_delta)
            joint_delta = step_delta[:, joint_indices]
            gripper_delta = step_delta[:, gripper_indices]
            intra_joint_p95 = float(np.percentile(joint_delta, 95))
            intra_joint_max = float(np.max(joint_delta))
            intra_gripper_p95 = float(np.percentile(gripper_delta, 95))
            intra_gripper_max = float(np.max(gripper_delta))

            max_step, max_joint_local = np.unravel_index(
                int(np.argmax(joint_delta)), joint_delta.shape
            )
            max_joint_index = int(joint_indices[max_joint_local])
            max_joint_name = action_name(max_joint_index)
            max_joint_direction = "+" if signed_step_delta[max_step, max_joint_index] >= 0 else "-"
        else:
            intra_joint_p95 = intra_joint_max = 0.0
            intra_gripper_p95 = intra_gripper_max = 0.0
            max_step = 0
            max_joint_name = "无"
            max_joint_direction = ""

        gap_text = "首次执行" if gap_ms is None else f"{gap_ms:.1f} ms"
        rospy.loginfo(
            "动作连续性 chunk#%d：chunk间保持=%s；"
            "首步跳变：关节=%.5f rad(%s)、夹爪=%.5f m(%s)；%s；"
            "%s；"
            "chunk内相邻步：关节P95=%.5f rad、最大=%.5f rad(%s，第%d→%d步，方向%s)，"
            "夹爪P95=%.5f m、最大=%.5f m",
            chunk_index,
            gap_text,
            boundary_joint,
            boundary_joint_name,
            boundary_gripper,
            boundary_gripper_name,
            feedback_text,
            smooth_text,
            intra_joint_p95,
            intra_joint_max,
            max_joint_name,
            max_step + 1,
            max_step + 2,
            max_joint_direction,
            intra_gripper_p95,
            intra_gripper_max,
        )

    def _make_trajectory(self, current, target):
        steps = max(1, int(self.args.move_duration * self.args.publish_rate))
        alpha = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, steps + 1) / steps))
        return current[None, :] + alpha[:, None] * (target - current)[None, :]

    def _start_motion(self, target, state):
        sample = self.observation.get_qpos_with_version()
        if sample is None:
            rospy.logwarn("尚未收到完整关节反馈，暂时不能规划运动")
            return False
        current, feedback_version = sample

        self.trajectory = self._make_trajectory(current, target)
        self.trajectory_index = 0
        self.target = target.copy()
        self.stable_count = 0
        self.last_feedback_version = feedback_version
        self.actions.clear()
        self.policy_chunk_index = 0
        self.last_policy_action_time = None
        self._reset_rtc_locked()
        self.state = state
        rospy.loginfo("%s，运动时间 %.1f 秒", state, self.args.move_duration)
        return True

    def _target_reached(self, target):
        sample = self.observation.get_qpos_with_version()
        if sample is None:
            return False, False
        current, feedback_version = sample

        # 左右臂都产生新反馈后，才进行一次到位判断。
        if (feedback_version[0] <= self.last_feedback_version[0]
                or feedback_version[1] <= self.last_feedback_version[1]):
            return False, False
        self.last_feedback_version = feedback_version

        joint_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        joint_error = np.max(np.abs(current[joint_indices] - target[joint_indices]))
        gripper_error = max(abs(current[6] - target[6]), abs(current[13] - target[13]))
        reached = joint_error <= self.args.joint_tolerance and gripper_error <= self.args.gripper_tolerance
        return True, reached

    def _process_requests(self):
        # 退出请求保持有效，后续 i/s 都不能覆盖它。
        if self.exit_requested:
            if self.state not in (ControlState.MOVING_B, ControlState.AT_B):
                self.observation.stop_video_recording()
                self._start_motion(self.pose_b, ControlState.MOVING_B)
            return

        if self.go_a_requested:
            self.go_a_requested = False
            self.observation.stop_video_recording()
            if self.state != ControlState.AT_A:
                self._start_motion(self.pose_a, ControlState.MOVING_A)
            return

        if self.start_requested:
            self.start_requested = False
            if self.state == ControlState.AT_A:
                self.actions.clear()
                self.epoch += 1
                self._reset_rtc_locked()
                self.policy_chunk_index = 0
                self.last_policy_action_time = None
                self.state = ControlState.POLICY
                self.target = self.pose_a.copy()
                self.observation.start_video_recording()
                rospy.loginfo("已进入推理状态，等待第一批动作")

    def _next_interpolated_target(self, now):
        """Controller lock serializes model pops, endpoint reservations and merge."""
        if not self.rtc_bootstrapped:
            return self.target
        try:
            if self.interpolator.origin is None:
                self.interpolator.start(now)
                advance, phase = 1, 0.0
            else:
                advance, phase = self.interpolator.sample(now)
                # At the final knot there was no endpoint to interpolate to.
                # Do not jump to a newly arriving uncommitted target at the next
                # boundary. Operator restart is required after lost lookahead.
                if advance and self._interpolation_end is None:
                    self._pause_rtc_locked("插值前瞻耗尽，保持最后目标，拒绝跳跃续接")
                    return self.target
            if advance:
                self._interpolation_start = self.rtc_queue.pop()
                if self.diagnostics.enabled:
                    self._diagnostic_source = (self._diagnostic_sources.popleft()
                                               if self._diagnostic_sources else None)
                self._interpolation_end = self.rtc_queue.reserve_next()
                self._interpolation_end_source = (self._diagnostic_sources[0]
                                                  if self._diagnostic_sources else None)
                self.last_policy_action_time = now
            self._interpolation_phase = phase
            self.target = self.interpolator.target(
                self._interpolation_start, self._interpolation_end, phase)
        except (InterpolationTimingError, QueueExhausted) as exc:
            self._pause_rtc_locked("高频插值暂停：" + str(exc))
        return self.target

    def _next_target(self, now=None):
        if self.state in (ControlState.MOVING_A, ControlState.MOVING_B):
            if self.trajectory_index < len(self.trajectory):
                self.target = self.trajectory[self.trajectory_index]
                self.trajectory_index += 1
            else:
                has_new_feedback, reached = self._target_reached(self.target)
                if has_new_feedback:
                    if reached:
                        self.stable_count += 1
                        if self.stable_count >= self.arrival_stable_cycles:
                            if self.state == ControlState.MOVING_A:
                                self.state = ControlState.AT_A
                                rospy.loginfo("已确认到达预设位置 A，可以按 s 开始推理")
                            else:
                                self.state = ControlState.AT_B
                                self.at_b_event.set()
                                rospy.loginfo("已确认到达预设位置 B")
                    else:
                        self.stable_count = 0

        elif self.state == ControlState.POLICY and self.rtc_enabled:
            if self.interpolation_enabled:
                return self._next_interpolated_target(time.monotonic() if now is None else now)
            if self.rtc_queue.remaining:
                self.target = self.rtc_queue.pop()
                if self.diagnostics.enabled:
                    self._diagnostic_source = (self._diagnostic_sources.popleft()
                                               if self._diagnostic_sources else None)
                self.last_policy_action_time = time.monotonic()
            elif self.rtc_bootstrapped:
                self._pause_rtc_locked("动作队列耗尽，拒绝重复播放旧轨迹")
        elif self.state == ControlState.POLICY and self.actions:
            self.target = self.actions.popleft()
            if self.diagnostics.enabled:
                self._diagnostic_source = (self._diagnostic_sources.popleft()
                                           if self._diagnostic_sources else None)
            self.last_policy_action_time = time.monotonic()

        return self.target

    def _run(self):
        try:
            rospy.loginfo("等待从臂关节反馈……")
            while not self.stop_event.is_set():
                current = self.observation.get_qpos()
                if current is not None:
                    with self.lock:
                        self.target = current
                    self.ready_event.set()
                    rospy.loginfo("已收到关节反馈，启动后保持当前位置；按 i 前往 A")
                    break
                time.sleep(0.02)

            rate = rospy.Rate(self.args.publish_rate)
            previous_state = None
            while not self.stop_event.is_set() and not rospy.is_shutdown():
                with self.lock:
                    self._process_requests()
                    target = self._next_target()
                    if self.diagnostics.enabled:
                        context = {"state": self.state, "epoch": self.epoch,
                                   "queue_tick": self.rtc_queue.tick,
                                   "queue_remaining": self.rtc_queue.remaining,
                                   "session_id": self.rtc_queue.session_id,
                                   "source": self._diagnostic_source}
                        if self.interpolation_enabled:
                            context.update(interpolation={
                                "method": "linear", "action_rate": self.action_rate,
                                "publish_rate": self.args.publish_rate,
                                "phase": self._interpolation_phase,
                                "model_index": self.interpolator.index,
                                "model_origin_monotonic_s": self.interpolator.origin,
                                "start_target": self._interpolation_start,
                                "end_target": self._interpolation_end,
                                "end_source": self._interpolation_end_source,
                                "grippers": "hold_start_until_next_model_boundary"})
                        state_key = (self.state, self.epoch, self.rtc_queue.session_id)
                        if state_key != previous_state:
                            self.diagnostics.emit("control_state", **context)
                            previous_state = state_key

                if target is not None:
                    if self.diagnostics.enabled:
                        publish_started_ns = time.monotonic_ns()
                        publish_ok = False
                        stamp = None
                        try:
                            stamp = self.publisher.publish(target)
                            publish_ok = True
                        finally:
                            self.diagnostics.emit("command", target=np.asarray(target, dtype=np.float32), **context,
                                                  publish_started_monotonic_ns=publish_started_ns,
                                                  publish_finished_monotonic_ns=time.monotonic_ns(),
                                                  ros_stamp_s=stamp, publish_ok=publish_ok)
                    else:
                        self.publisher.publish(target)
                rate.sleep()
        except Exception as exc:
            with self.lock:
                self.error = exc
            self.error_event.set()
            rospy.logerr("机械臂控制线程异常：%s", exc)


class KeyboardMonitor:
    """在启动终端中读取单键，不需要按回车。"""

    def __init__(self, controller):
        self.controller = controller
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="keyboard", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self):
        if not sys.stdin.isatty():
            rospy.logwarn("标准输入不是终端，i/s/e 键不可用；仍可使用 Ctrl+C")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                key = sys.stdin.read(1).lower()
                if key == "i":
                    self.controller.request_go_a()
                elif key == "s":
                    self.controller.request_start()
                elif key == "e":
                    self.controller.request_exit("e")
                elif key in "12345":
                    labels = {"1": "free_space", "2": "alignment", "3": "contact",
                              "4": "success", "5": "failure"}
                    self.controller.record_marker(labels[key])
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class PolicyClient:
    """远端策略客户端；第一次按 s 后才建立连接。"""

    def __init__(self, url, timeout, diagnostics=NO_DIAGNOSTICS):
        self.url = url
        self.timeout = timeout
        self.ws = None
        self.diagnostics = diagnostics
        self.diagnostic_rpc_index = 0

    def begin_session(self):
        """Discard an idle socket before a new s session; never retry stale actions."""
        try:
            self.close()
        except Exception as exc:
            rospy.logwarn("关闭旧会话连接失败，将使用新连接：%s", exc)
        self.diagnostics.emit("connection_reset", reason="new_policy_session")

    def connect(self):
        if self.ws is not None:
            return
        connect_start = time.perf_counter()
        self.ws = websocket.create_connection(self.url, timeout=self.timeout)
        handshake_done = time.perf_counter()
        metadata = self.ws.recv()
        metadata_done = time.perf_counter()
        if metadata == "" or metadata == b"":
            raise ConnectionError("策略连接在握手时关闭（空响应）")
        if isinstance(metadata, str):
            raise RuntimeError(f"策略服务握手失败：\n{metadata}")
        self.server_metadata = unpackb(metadata)
        self.diagnostics.emit("server_connected", metadata=self.server_metadata,
                              connect_ms=(metadata_done - connect_start) * 1000)
        rospy.loginfo(
            "已连接 pi0.5 策略服务：%s；WebSocket握手=%.1f ms，等待metadata=%.1f ms，总计=%.1f ms",
            self.url,
            (handshake_done - connect_start) * 1000,
            (metadata_done - handshake_done) * 1000,
            (metadata_done - connect_start) * 1000,
        )

    def infer(self, observation):
        self.diagnostic_rpc_index += 1
        rpc_index = self.diagnostic_rpc_index
        if self.diagnostics.enabled:
            self.diagnostics.emit("inference_request", rpc_index=rpc_index,
                                  rtc=observation.get("rtc"), state=observation.get("state"))
        pack_start = time.perf_counter()
        request = packb(observation)
        pack_done = time.perf_counter()

        self.ws.send(request, opcode=websocket.ABNF.OPCODE_BINARY)
        send_done = time.perf_counter()

        response = self.ws.recv()
        recv_done = time.perf_counter()
        if response == "" or response == b"":
            raise ConnectionError("策略连接已关闭（空响应），当前推理暂停；重新开始会建立新连接")
        if isinstance(response, str):
            raise RuntimeError(f"策略服务返回错误：\n{response}")

        result = unpackb(response)
        unpack_done = time.perf_counter()
        if self.diagnostics.enabled:
            self.diagnostics.emit("model_response", rpc_index=rpc_index,
                                  actions=result.get("actions"), rtc_info=result.get("rtc_info"),
                                  server_timing=result.get("server_timing"),
                                  timing_ms={"pack": (pack_done - pack_start) * 1000,
                                             "send": (send_done - pack_done) * 1000,
                                             "wait": (recv_done - send_done) * 1000,
                                             "unpack": (unpack_done - recv_done) * 1000,
                                             "total": (unpack_done - pack_start) * 1000},
                                  request_bytes=len(request), response_bytes=len(response))
        rospy.loginfo(
            "WebSocket请求耗时：序列化=%.1f ms，发送=%.1f ms，等待响应=%.1f ms，"
            "反序列化=%.1f ms；请求=%.1f KiB，响应=%.1f KiB",
            (pack_done - pack_start) * 1000,
            (send_done - pack_done) * 1000,
            (recv_done - send_done) * 1000,
            (unpack_done - recv_done) * 1000,
            len(request) / 1024,
            len(response) / 1024,
        )
        return result

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None


def run_rtc_policy_loop(args, controller, observation, policy):
    """主线程执行网络请求，已有控制线程持续消费带 tick 的动作队列。"""
    active_epoch = None
    discarded_epoch = None
    while (not controller.at_b_event.is_set() and not controller.error_event.is_set()
           and not rospy.is_shutdown()):
        epoch = controller.policy_request()
        if epoch is None:
            if active_epoch is not None and not controller.policy_epoch_is_current(active_epoch):
                observation.stop_capture()
                active_epoch = None
            time.sleep(0.005)
            continue
        try:
            if epoch != active_epoch:
                policy.begin_session()
                observation.start_capture()
                active_epoch = epoch
            # 在重处理图像之前保存旧前缀；图像处理与网络等待均计入实际 tick 延迟。
            obs = observation.get(request_snapshot=lambda: controller.begin_rtc_request(epoch))
            if obs is None:
                time.sleep(0.005)
                continue
            started = time.monotonic()
            policy.connect()
            metadata = policy.server_metadata
            if (metadata.get("rtc_protocol") != 1 or metadata.get("action_horizon") != 50
                    or metadata.get("action_dim") != 14):
                raise ValueError("服务未提供 RTC v1 / 50x14 接口，请连接 RTC 服务")
            result = policy.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError("RTC 服务必须返回有限的 (50,14) 动作块")
            info = result.get("rtc_info", {})
            if any(info.get(k) != obs["rtc"][k] for k in ("session_id", "request_id", "observation_tick")):
                raise ValueError("RTC 响应身份与请求不一致")
            if (args.rtc_mode == "guided" and "prefix" in obs["rtc"]
                    and not info.get("guidance_applied")):
                raise ValueError("请求了 RTC 前缀引导，但服务未启用 guidance")
            # 保留原客户端每轮 s 后丢弃首块的行为，同时释放该在途请求。
            if discarded_epoch != epoch:
                if controller.discard_initial_rtc_result(epoch):
                    discarded_epoch = epoch
                    rospy.loginfo("本轮首块已丢弃，下一次请求重新初始化 RTC 前缀")
                continue
            if controller.submit_rtc_result(result, epoch):
                rospy.loginfo("RTC 收到完整 50 步动作；重规划间隔=%d步，往返=%.1f ms",
                              controller.rtc_replan_steps, (time.monotonic()-started)*1000)
        except Exception as exc:
            # 先取消推理、保持最后目标，再关闭可能阻塞的网络连接。
            controller.pause_rtc(str(exc), epoch)
            observation.stop_capture()
            active_epoch = None
            try:
                policy.close()
            except Exception as close_error:
                rospy.logwarn("关闭 RTC 连接失败，继续保持暂停状态：%s", close_error)


def get_arguments():
    parser = argparse.ArgumentParser(description="运行 pi0.5 远端推理并通过键盘控制机械臂")
    parser.add_argument("--url", default=POLICY_URL)
    parser.add_argument("--rtc-mode", choices=("legacy", "off", "guided"), default="legacy",
                        help="legacy 保留原同步逻辑；off 为普通异步；guided 为异步 RTC")
    parser.add_argument("--rtc-delay", type=int, default=5, help="推理期间承诺继续执行的旧目标步数")
    parser.add_argument("--rtc-smoothing-alpha", type=float, default=1.0,
                        help="服务端关节 EMA，(0,1]；1 关闭，夹爪不参与")
    parser.add_argument("--rtc-transition-steps", type=int, default=0,
                        help="RTC 换块局部位置/速度过渡步数，0 关闭；仅 12 关节，原始尾段和夹爪不变")
    parser.add_argument("--rtc-prefix-attention-horizon", type=int, default=0,
                        help="RTC 旧轨迹软约束长度，0 使用全部剩余前缀；与重规划间隔独立")
    parser.add_argument("--publish-rate", type=float, default=40.0)
    parser.add_argument("--action-rate", type=float, default=0.0,
                        help="模型动作频率；0 与发布频率相同，保留原行为。插值入口固定为 30")
    parser.add_argument("--action-interpolation", choices=("none", "linear"), default="none",
                        help="RTC chunk 内关节插值；linear 将模型时间与高频发布分离，夹爪按模型时刻保持")
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=0,
        help="legacy: 只执行前 N 步；RTC: 重规划间隔，0 默认 30，保留完整 50 步前缀",
    )
    parser.add_argument(
        "--boundary-smooth-steps",
        type=int,
        default=5,
        help="新 chunk 开头用于关节余弦平滑的步数；0 表示关闭，夹爪始终不平滑",
    )
    parser.add_argument(
        "--boundary-smooth-threshold",
        type=float,
        default=0.01,
        help="模型首步距真实位置的最大关节差达到该值时才平滑（弧度）",
    )
    parser.add_argument("--move-duration", type=float, default=3.0, help="当前位置移动到 A/B 的插值时间（秒）")
    parser.add_argument("--joint-tolerance", type=float, default=0.05, help="到位关节误差阈值（弧度）")
    parser.add_argument("--gripper-tolerance", type=float, default=0.01, help="到位夹爪误差阈值")
    parser.add_argument("--stable-cycles", type=int, default=5, help="连续满足到位条件的控制周期数")
    parser.add_argument("--feedback-timeout", type=float, default=5.0, help="启动时等待关节反馈的时间（秒）")
    parser.add_argument("--return-timeout", type=float, default=30.0, help="异常退出时等待回到 B 的最长时间（秒）")
    parser.add_argument("--ws-timeout", type=float, default=30.0, help="策略服务连接和响应超时（秒）")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="推理期间录制头部相机；默认不录像",
    )
    parser.add_argument(
        "--record-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos"),
        help="推理期间头部相机录像的保存目录",
    )
    parser.add_argument("--video-fps", type=float, default=30.0, help="头部相机录像帧率")
    parser.add_argument("--diagnostics-dir", default=None,
                        help="启用异步诊断记录，在此目录下创建独立实验子目录；默认关闭")
    parser.add_argument("--diagnostics-label", default="",
                        help="本次实验说明，例如 rtc_no_ema_01；仅写入记录，不改变控制")
    parser.add_argument("--pose-a", type=float, nargs=14, default=PRESET_A, help="开始推理的点位")
    parser.add_argument("--pose-b", type=float, nargs=14, default=PRESET_B, help="结束推理的点位，回到平稳位置")
    parser.add_argument("--img-front-topic", default="/camera_f/color/image_raw")
    parser.add_argument("--img-left-topic", default="/camera_l/color/image_raw")
    parser.add_argument("--img-right-topic", default="/camera_r/color/image_raw")
    parser.add_argument("--puppet-arm-left-topic", default="/puppet/joint_left")
    parser.add_argument("--puppet-arm-right-topic", default="/puppet/joint_right")
    parser.add_argument("--left-cmd-topic", default="/master/joint_left")
    parser.add_argument("--right-cmd-topic", default="/master/joint_right")
    return parser.parse_args()


def validate_action_timing(args):
    """Reject an accidental speed change before initializing any ROS objects."""
    if (not np.isfinite(args.publish_rate) or args.publish_rate <= 0
            or not np.isfinite(args.action_rate) or args.action_rate < 0):
        raise ValueError("发布频率须有限且大于 0，动作频率须有限且非负")
    action_rate = args.action_rate or args.publish_rate
    if args.action_interpolation == "none":
        if not np.isclose(action_rate, args.publish_rate, rtol=0, atol=1e-9):
            raise ValueError("动作/发布频率不同时必须启用 --action-interpolation linear")
    elif args.rtc_mode == "legacy":
        raise ValueError("高频插值仅支持异步 RTC 模式 off/guided，暂不支持 legacy")
    elif args.publish_rate < action_rate:
        raise ValueError("插值发布频率不能低于模型动作频率")


def _run_client(args):
    validate_action_timing(args)
    if args.publish_rate <= 0 or args.move_duration <= 0:
        raise ValueError("发布频率和 A/B 运动时间必须大于 0")
    if args.chunk_steps < 0:
        raise ValueError("每次执行的 chunk 步数不能为负数；0 表示执行全部动作")
    if args.rtc_mode != "legacy":
        replan = args.chunk_steps or 30
        if not 1 <= replan < 50 or not 0 <= args.rtc_delay <= 50 - replan:
            raise ValueError("RTC 要求重规划间隔为 1..49，delay 不超过 50-重规划间隔")
        if not np.isfinite(args.rtc_smoothing_alpha) or not 0 < args.rtc_smoothing_alpha <= 1:
            raise ValueError("RTC smoothing alpha 必须在 (0,1] 内")
        if (args.rtc_transition_steps < 0 or args.rtc_transition_steps == 1
                or args.rtc_transition_steps > min(replan, 20)):
            raise ValueError("RTC 局部过渡须为 0 或 2..min(重规划间隔,20)")
        if (args.rtc_prefix_attention_horizon != 0
                and not args.rtc_delay <= args.rtc_prefix_attention_horizon <= 50):
            raise ValueError("RTC 软约束长度须为 0 或 delay..50")
    if args.boundary_smooth_steps < 0:
        raise ValueError("chunk 边界平滑步数不能为负数；0 表示关闭")
    if args.boundary_smooth_threshold < 0:
        raise ValueError("chunk 边界平滑阈值不能为负数")
    if args.video_fps <= 0:
        raise ValueError("录像帧率必须大于 0")
    if args.joint_tolerance < 0 or args.gripper_tolerance < 0 or args.stable_cycles < 1:
        raise ValueError("到位阈值不能为负，稳定周期数必须至少为 1")

    pose_a = validate_pose("A", args.pose_a)
    pose_b = validate_pose("B", args.pose_b)

    # 自行处理 Ctrl+C，保证先请求回到 B，再关闭 ROS 客户端。
    rospy.init_node("pi05_ws_client", anonymous=True, disable_signals=True)
    observation = RosObservation(args)
    publisher = ArmPublisher(args)
    controller = ArmController(args, observation, publisher, pose_a, pose_b)
    keyboard = KeyboardMonitor(controller)
    policy = PolicyClient(args.url, args.ws_timeout, diagnostics=args._diagnostics)

    rospy.loginfo("动作时间频率=%.1f Hz，目标发布频率=%.1f Hz，chunk 内插值=%s",
                  controller.action_rate, args.publish_rate, args.action_interpolation)
    controller.start()
    if not controller.wait_ready(args.feedback_timeout):
        controller.stop()
        if controller.error_event.is_set():
            raise RuntimeError(f"机械臂控制线程启动失败：{controller.get_error()}")
        raise RuntimeError("等待从臂关节反馈超时，请检查 mode=1 的 Piper 节点和话题")

    keyboard.start()

    def handle_sigint(_signum, _frame):
        controller.request_exit("Ctrl+C")

    signal.signal(signal.SIGINT, handle_sigint)
    rospy.loginfo("键盘控制已启动：i=前往 A，s=开始推理，e/Ctrl+C=前往 B 后退出")
    if args._diagnostics.enabled:
        rospy.loginfo("诊断记录目录：%s", args._diagnostics.run_dir)
        rospy.loginfo("仅标记、不改变运动：1=自由空间，2=对准，3=接触，4=成功，5=失败")

    active_epoch = None
    first_chunk_discarded_epoch = None
    normal_exit = False
    try:
        while (not controller.at_b_event.is_set()
               and not controller.error_event.is_set()
               and not rospy.is_shutdown()
               and args.rtc_mode == "legacy"):
            epoch = controller.policy_request()
            if epoch is None:
                # 动作执行期间继续缓存相机消息；仅在离开当前推理会话时停止。
                if (active_epoch is not None
                        and not controller.policy_epoch_is_current(active_epoch)):
                    observation.stop_capture()
                    active_epoch = None
                time.sleep(0.02)
                continue

            if active_epoch != epoch:
                observation.start_capture()
                active_epoch = epoch

            obs = observation.get()
            if obs is None:
                time.sleep(0.01)
                continue

            started = time.time()
            try:
                policy.connect()
                result = policy.infer(obs)
                all_actions = result["actions"]
                infer_ms = result.get("server_timing", {}).get("infer_ms")

                # 每次按 s 开启的新推理会话，都丢弃第一批动作。
                # 推理期间相机仍在更新缓存，因此可以直接使用下一组新观测。
                if first_chunk_discarded_epoch != epoch:
                    if controller.policy_epoch_is_current(epoch):
                        first_chunk_discarded_epoch = epoch
                        rospy.loginfo(
                            "本轮首次推理返回 %d 步动作，已全部丢弃；"
                            "服务端耗时=%s ms，往返耗时=%.1f ms，准备请求第二个 chunk",
                            len(all_actions), infer_ms, (time.time() - started) * 1000,
                        )
                    else:
                        rospy.loginfo("控制状态已改变，丢弃过期的首次推理结果")
                    continue

                actions = all_actions
                if args.chunk_steps > 0:
                    actions = all_actions[:args.chunk_steps]
                accepted = controller.submit_actions(actions, epoch)
                if accepted:
                    rospy.loginfo(
                        "收到 %d 步推理动作，本轮执行前 %d 步，服务端耗时=%s ms，往返耗时=%.1f ms",
                        len(all_actions), len(actions), infer_ms, (time.time() - started) * 1000,
                    )
                else:
                    observation.stop_capture()
                    active_epoch = None
                    rospy.loginfo("控制状态已改变，丢弃过期的推理结果")
            except Exception as exc:
                rospy.logerr("推理请求失败：%s；机械臂将返回 A", exc)
                policy.close()
                observation.stop_capture()
                active_epoch = None
                controller.request_go_a()

        if args.rtc_mode != "legacy":
            run_rtc_policy_loop(args, controller, observation, policy)

        if controller.error_event.is_set():
            raise RuntimeError(f"机械臂控制线程异常：{controller.get_error()}")
        normal_exit = controller.at_b_event.is_set()
    finally:
        observation.stop_video_recording()
        observation.stop_capture()
        if (not normal_exit and controller.ready_event.is_set()
                and not controller.error_event.is_set() and not rospy.is_shutdown()):
            if args.rtc_mode == "legacy" or controller.exit_requested:
                controller.request_exit("程序异常")
                if not controller.wait_at_b(args.return_timeout):
                    rospy.logerr("等待回到 B 超时，请立即检查机械臂状态")
            else:
                controller.pause_rtc("主循环异常退出", controller.epoch)
                rospy.logerr("RTC 主循环退出；未自动规划 A/B 运动")
        elif controller.error_event.is_set():
            rospy.logerr("控制线程已经停止，无法自动执行返回 B，请立即检查机械臂状态")

        policy.close()
        keyboard.stop()
        controller.stop()
        observation.close()
        rospy.signal_shutdown("客户端已完成退出流程")

    rospy.loginfo("机械臂已到达 B，程序正常退出")


def main():
    args = get_arguments()
    recorder = (DiagnosticRecorder(args.diagnostics_dir, vars(args), args.diagnostics_label)
                if args.diagnostics_dir else NO_DIAGNOSTICS)
    args._diagnostics = recorder
    try:
        _run_client(args)
    finally:
        summary = recorder.close()
        if recorder.enabled:
            if summary["complete"]:
                rospy.loginfo("诊断数据已保存：%s；共 %d 条，记录队列无丢失",
                              recorder.run_dir, summary["written_events"])
            else:
                rospy.logerr("诊断记录不完整：%s；状态=%s", recorder.run_dir, summary)


if __name__ == "__main__":
    main()
