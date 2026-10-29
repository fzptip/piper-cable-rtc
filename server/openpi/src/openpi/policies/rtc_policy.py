"""Validated stateless RTC wire contract and optional joint-only causal smoothing."""
import time

import numpy as np

HORIZON = 50
ACTION_DIM = 14
MODEL_DIM = 32
JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return int(value)


def make_weights(delay, end):
    index = np.arange(HORIZON, dtype=np.float32)
    weight = np.clip((delay - 1 - index) / (end - delay + 1) + 1, 0, 1)
    weight *= np.expm1(weight) / np.expm1(1.0)
    weight[index < delay] = 1.0
    weight[index >= end] = 0
    result = np.zeros((HORIZON, MODEL_DIM), dtype=np.float32)
    result[:, :ACTION_DIM] = weight[:, None]
    return result


def smooth_actions(actions, alpha, start=0, anchor=None):
    """Causal EMA on 12 joints. Grippers and committed prefix are unchanged.

    alpha=1 is exact identity. anchor is the last committed absolute target;
    there is no hidden cross-session filter state.
    """
    result = np.array(actions, dtype=np.float32, copy=True)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("smoothing_alpha must be in (0,1]")
    integer(start, "smoothing start", 0, len(result))
    if alpha == 1 or start == len(result):
        return result
    previous = result[start, JOINT_INDICES] if anchor is None else np.asarray(anchor)[JOINT_INDICES]
    for i in range(start, len(result)):
        result[i, JOINT_INDICES] = alpha * result[i, JOINT_INDICES] + (1 - alpha) * previous
        previous = result[i, JOINT_INDICES]
    return result


def validate_observation(obs):
    state = np.asarray(obs.get("state"))
    if state.shape != (ACTION_DIM,) or not np.issubdtype(state.dtype, np.number) or not np.isfinite(state).all():
        raise ValueError("state must contain 14 finite raw dataset values")
    images = obs.get("images", {})
    for camera in CAMERAS:
        image = np.asarray(images.get(camera))
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f"{camera} must be RGB uint8 HWC (224,224,3)")
    if "prompt" in obs and not isinstance(obs["prompt"], str):
        raise ValueError("prompt must be a string")


class RTCPolicy:
    def __init__(self, policy, *, smoothing_alpha=1.0, max_guidance=10.0):
        if not np.isfinite(max_guidance) or not 0 < max_guidance <= 100:
            raise ValueError("max_guidance must be in (0,100]")
        if not np.isfinite(smoothing_alpha) or not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0,1]")
        self.policy = policy
        self.smoothing_alpha = smoothing_alpha
        self.max_guidance = max_guidance
        self.metadata = {**policy.metadata, "rtc_protocol": 1, "action_horizon": HORIZON,
                         "action_dim": ACTION_DIM, "rtc_modes": ["off", "guided"],
                         "smoothing_alpha": smoothing_alpha, "max_guidance": max_guidance}

    def infer(self, observation, *, noise=None):
        started = time.perf_counter()
        if not isinstance(observation, dict):
            raise ValueError("observation must be a dict")
        obs = dict(observation)
        request = obs.pop("rtc", None)
        validate_observation(obs)
        rtc = None
        delay = 0
        anchor = None
        mode = "off"
        alpha = self.smoothing_alpha
        identity = {}
        if request is not None:
            if not isinstance(request, dict):
                raise ValueError("rtc must be a dict")
            mode = request.get("mode", "guided")
            alpha = request.get("smoothing_alpha", alpha)
            if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.integer, np.floating)):
                raise ValueError("smoothing_alpha must be numeric")
            if not np.isfinite(alpha) or not 0 < alpha <= 1:
                raise ValueError("smoothing_alpha must be in (0,1]")
            alpha = float(alpha)
            if mode not in ("off", "guided"):
                raise ValueError("rtc.mode must be off or guided")
            session = request.get("session_id")
            if not isinstance(session, str) or not 1 <= len(session) <= 128:
                raise ValueError("rtc.session_id must be a nonempty string, at most 128 characters")
            identity = {"session_id": session,
                        "request_id": integer(request.get("request_id"), "request_id", 0, 2**53 - 1),
                        "observation_tick": integer(request.get("observation_tick"), "observation_tick", 0, 2**53 - 1)}
            delay = integer(request.get("delay", 0), "delay", 0, HORIZON - 1)
            prefix = request.get("prefix")
            if prefix is not None:
                prefix = np.asarray(prefix, dtype=np.float32)
                if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM or not 1 <= len(prefix) <= HORIZON:
                    raise ValueError("prefix must have shape (1..50,14)")
                if not np.isfinite(prefix).all():
                    raise ValueError("prefix contains non-finite values")
                end = integer(request.get("prefix_attention_horizon", len(prefix)), "prefix_attention_horizon", 0, len(prefix))
                if delay > end:
                    raise ValueError("prefix attention must cover delay")
                anchor = prefix[delay - 1] if delay else None
                if mode == "guided" and end:
                    padded = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
                    padded[:len(prefix)] = prefix
                    rtc = {"prefix": padded, "weights": make_weights(delay, end), "max_guidance": self.max_guidance}
            elif delay:
                raise ValueError("nonzero delay requires a prefix")
        result = self.policy.infer(obs, noise=noise, rtc=rtc)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ValueError("model returned invalid actions")
        result["actions"] = smooth_actions(actions, alpha, delay, anchor)
        result["rtc_info"] = {**identity, "mode": mode, "guidance_applied": rtc is not None,
                              "delay": delay, "smoothing_alpha": alpha,
                              "total_ms": (time.perf_counter() - started) * 1000}
        return result
