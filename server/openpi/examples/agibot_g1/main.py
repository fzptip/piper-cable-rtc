import contextlib
import dataclasses
import signal
import time
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

from openpi.policies import g1_policy

try:
    from a2d_sdk.robot import CosineCamera as Camera
    from a2d_sdk.robot import RobotController
    from a2d_sdk.robot import RobotDds
except ImportError:
    Camera = None
    RobotController = None
    RobotDds = None


DEFAULT_CAMERA_NAMES = ("head", "hand_left", "hand_right")
SERVO_MODE = 1
OBSERVATION_RETRY_SLEEP_S = 0.02

# 用于获取机器人观测值后向服务端发起推理请求，获取响应结果后向机器人下发动作指令

@dataclasses.dataclass
class Args:
    instruction: str | None = None

    # Remote policy server.
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Loop configuration.
    max_policy_calls: int = 0
    open_loop_horizon: int = 0
    control_frequency_hz: float = 30.0
    loop_time_margin_s: float = 0.05

    # Image preprocessing.
    image_height: int = 224
    image_width: int = 224

    # Robot runtime.
    dry_run: bool = False
    robot_link: str = "base_link"
    sensor_warmup_s: float = 1.0
    sensor_ready_timeout_s: float = 3.0
    sensor_poll_interval_s: float = 0.1

    # Some SDK builds report gripper positions in raw units. If values are already in [0, 1],
    # this normalization is skipped automatically.
    gripper_raw_min: float = 35.0
    gripper_raw_max: float = 120.0


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until the protected block completes."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _unwrap_packet(packet) -> tuple[Any, Any]:
    if packet is None:
        return None, None
    if isinstance(packet, tuple) and len(packet) == 2:
        return packet
    return packet, None


def _poll_state(getter, length: int, name: str, *, timeout_s: float, interval_s: float) -> list[float]:
    deadline = time.time() + timeout_s
    last_values = None
    while time.time() < deadline:
        values, _ = _unwrap_packet(getter())
        last_values = values
        if values is not None and len(values) == length:
            return list(values)
        time.sleep(interval_s)
    raise RuntimeError(f"{name} was not ready within {timeout_s}s. Last value: {last_values!r}")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 image, got shape {image.shape}")

    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.max(image)) <= 1.0:
            image = (255.0 * image).astype(np.uint8)
        else:
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def _prepare_policy_image(image, *, height: int, width: int) -> np.ndarray:
    image = _parse_image(image)
    image = image_tools.convert_to_uint8(image)
    return image_tools.resize_with_pad(image, height, width)


def _normalize_gripper_observation(gripper_q: np.ndarray, *, raw_min: float, raw_max: float) -> np.ndarray:
    gripper_q = np.asarray(gripper_q, dtype=np.float32)
    if gripper_q.size != 2:
        raise ValueError(f"Expected 2 gripper values, got shape {gripper_q.shape}")

    if np.max(gripper_q) > 1.0:
        if raw_max <= raw_min:
            raise ValueError(f"Invalid gripper range: min={raw_min}, max={raw_max}")
        gripper_q = (gripper_q - raw_min) / (raw_max - raw_min)

    return np.clip(gripper_q, 0.0, 1.0).astype(np.float32)


class AgibotG1Client:
    def __init__(self, args: Args):
        if Camera is None or RobotController is None or RobotDds is None:
            raise ImportError("a2d_sdk is required to run examples/agibot_g1/main.py on the robot.")

        self._args = args
        self._policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        self._robot_dds = RobotDds()
        self._robot_controller = None if args.dry_run else RobotController()
        self._camera = Camera(list(DEFAULT_CAMERA_NAMES))

        print(f"Connected to policy server. metadata={self._policy_client.get_server_metadata()}")
        self._setup_robot()

    def _setup_robot(self) -> None:
        print("Waiting for AgiBot G1 sensors to come online...")
        time.sleep(self._args.sensor_warmup_s)

        _poll_state(
            self._robot_dds.head_joint_states,
            2,
            "head_joint_states",
            timeout_s=self._args.sensor_ready_timeout_s,
            interval_s=self._args.sensor_poll_interval_s,
        )
        _poll_state(
            self._robot_dds.waist_joint_states,
            2,
            "waist_joint_states",
            timeout_s=self._args.sensor_ready_timeout_s,
            interval_s=self._args.sensor_poll_interval_s,
        )
        _poll_state(
            self._robot_dds.arm_joint_states,
            14,
            "arm_joint_states",
            timeout_s=self._args.sensor_ready_timeout_s,
            interval_s=self._args.sensor_poll_interval_s,
        )
        _poll_state(
            self._robot_dds.gripper_states,
            2,
            "gripper_states",
            timeout_s=self._args.sensor_ready_timeout_s,
            interval_s=self._args.sensor_poll_interval_s,
        )

        if self._robot_controller is None:
            print("dry_run=True: observations and inference only; no motion commands will be sent.")
        else:
            self._robot_controller.set_motion_control_mode(SERVO_MODE)
            print("Robot switched to SERVO mode.")

    def get_observation(self) -> dict[str, Any] | None:
        head_image, _ = _unwrap_packet(self._camera.get_latest_image("head"))
        left_wrist_image, _ = _unwrap_packet(self._camera.get_latest_image("hand_left"))
        right_wrist_image, _ = _unwrap_packet(self._camera.get_latest_image("hand_right"))

        arm_q, arm_ts = _unwrap_packet(self._robot_dds.arm_joint_states())
        gripper_q, _ = _unwrap_packet(self._robot_dds.gripper_states())
        waist_q, _ = _unwrap_packet(self._robot_dds.waist_joint_states())
        head_q, _ = _unwrap_packet(self._robot_dds.head_joint_states())

        if any(x is None for x in (head_image, left_wrist_image, right_wrist_image, arm_q, gripper_q, waist_q, head_q)):
            return None

        joint_position = np.asarray(arm_q, dtype=np.float32)
        if joint_position.size != 14:
            raise ValueError(f"Expected 14 arm joints, got shape {joint_position.shape}")

        effector_position = _normalize_gripper_observation(
            gripper_q,
            raw_min=self._args.gripper_raw_min,
            raw_max=self._args.gripper_raw_max,
        )

        return {
            g1_policy.HEAD_IMAGE_KEY: _prepare_policy_image(
                head_image,
                height=self._args.image_height,
                width=self._args.image_width,
            ),
            g1_policy.LEFT_WRIST_IMAGE_KEY: _prepare_policy_image(
                left_wrist_image,
                height=self._args.image_height,
                width=self._args.image_width,
            ),
            g1_policy.RIGHT_WRIST_IMAGE_KEY: _prepare_policy_image(
                right_wrist_image,
                height=self._args.image_height,
                width=self._args.image_width,
            ),
            g1_policy.JOINT_POSITION_KEY: joint_position,
            g1_policy.EFFECTOR_POSITION_KEY: effector_position,
            "ref_states": {
                "head": list(np.asarray(head_q, dtype=np.float32)),
                "waist": list(np.asarray(waist_q, dtype=np.float32)),
                "arm": list(joint_position),
                "gripper": list(effector_position),
            },
            "obs_timestamp": int(arm_ts if arm_ts is not None else time.time_ns()),
        }

    def request_action_chunk(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        request_data = {
            g1_policy.HEAD_IMAGE_KEY: observation[g1_policy.HEAD_IMAGE_KEY],
            g1_policy.LEFT_WRIST_IMAGE_KEY: observation[g1_policy.LEFT_WRIST_IMAGE_KEY],
            g1_policy.RIGHT_WRIST_IMAGE_KEY: observation[g1_policy.RIGHT_WRIST_IMAGE_KEY],
            g1_policy.JOINT_POSITION_KEY: observation[g1_policy.JOINT_POSITION_KEY],
            g1_policy.EFFECTOR_POSITION_KEY: observation[g1_policy.EFFECTOR_POSITION_KEY],
            g1_policy.PROMPT_KEY: instruction,
        }

        with prevent_keyboard_interrupt():
            response = self._policy_client.infer(request_data)

        action_chunk = np.asarray(response["actions"], dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[1] < g1_policy.STATE_DIM:
            raise ValueError(f"Expected action chunk shape (T, >=16), got {action_chunk.shape}")
        if not np.isfinite(action_chunk).all():
            raise ValueError("Policy output contains NaN or Inf values.")

        return action_chunk[:, : g1_policy.STATE_DIM]

    def execute_action_chunk(self, action_chunk: np.ndarray, observation: dict[str, Any]) -> tuple[float, float]:
        num_frames = len(action_chunk)
        if self._args.open_loop_horizon > 0:
            num_frames = min(num_frames, self._args.open_loop_horizon)
        if num_frames <= 0:
            raise ValueError("No actions available to execute.")

        action_chunk = action_chunk[:num_frames]
        arm_targets = np.concatenate([action_chunk[:, 0:7], action_chunk[:, 8:15]], axis=1)
        gripper_targets = np.clip(action_chunk[:, [7, 15]], 0.0, 1.0)

        robot_actions = []
        for arm_target, gripper_target in zip(arm_targets, gripper_targets, strict=True):
            robot_actions.append(
                {
                    "left_arm": {
                        "action_data": arm_target[:7].tolist(),
                        "control_type": "ABS_JOINT",
                    },
                    "right_arm": {
                        "action_data": arm_target[7:].tolist(),
                        "control_type": "ABS_JOINT",
                    },
                    "gripper": {
                        "action_data": gripper_target.tolist(),
                        "control_type": "ABS_JOINT",
                    },
                }
            )

        trajectory_reference_time = num_frames / self._args.control_frequency_hz
        if self._robot_controller is None:
            print(f"dry_run: skipping trajectory command for {num_frames} frames")
        else:
            self._robot_controller.trajectory_tracking_control(
                infer_timestamp=int(observation["obs_timestamp"]),
                robot_states=observation["ref_states"],
                robot_actions=robot_actions,
                robot_link=self._args.robot_link,
                trajectory_reference_time=trajectory_reference_time,
            )

        print(
            "Executed chunk: "
            f"frames={num_frames}, "
            f"reference_time={trajectory_reference_time:.3f}s, "
            f"left_final={np.round(robot_actions[-1]['left_arm']['action_data'], 4).tolist()}, "
            f"right_final={np.round(robot_actions[-1]['right_arm']['action_data'], 4).tolist()}, "
            f"gripper_final={np.round(robot_actions[-1]['gripper']['action_data'], 4).tolist()}"
        )

        return trajectory_reference_time, time.monotonic()

    def run(self, instruction: str) -> None:
        print(f"Starting AgiBot G1 rollout. instruction={instruction!r}")
        policy_calls = 0

        try:
            while self._args.max_policy_calls <= 0 or policy_calls < self._args.max_policy_calls:
                loop_start = time.monotonic()
                observation = self.get_observation()
                if observation is None:
                    time.sleep(OBSERVATION_RETRY_SLEEP_S)
                    continue

                action_chunk = self.request_action_chunk(observation, instruction)
                print(f"Policy returned action chunk with shape {action_chunk.shape}")

                reference_time, command_sent_time = self.execute_action_chunk(action_chunk, observation)
                policy_calls += 1

                next_loop_time = command_sent_time + reference_time + self._args.loop_time_margin_s
                sleep_s = max(0.0, next_loop_time - time.monotonic())
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif time.monotonic() - loop_start < 1.0 / self._args.control_frequency_hz:
                    time.sleep(max(0.0, 1.0 / self._args.control_frequency_hz - (time.monotonic() - loop_start)))

        except KeyboardInterrupt:
            print("Interrupted by user. Stopping robot motion...")
            if self._robot_controller is not None:
                self._robot_controller.set_motion_stop()
        finally:
            self._robot_dds.shutdown()
            self._camera.close()
            print("Released AgiBot G1 resources.")


def main(args: Args) -> None:
    instruction = args.instruction or input("Enter instruction: ").strip()
    if not instruction:
        raise ValueError("Instruction must not be empty.")

    client = AgibotG1Client(args)
    client.run(instruction)


if __name__ == "__main__":
    main(tyro.cli(Args))
