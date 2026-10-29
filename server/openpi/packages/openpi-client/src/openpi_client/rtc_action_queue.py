"""Thread-safe tick queue for a single asynchronous RTC inference worker.

This module issues absolute dataset-space targets only; it never drives hardware.
The consumer must stop/hold explicitly on QueueExhausted, never replay old targets.
"""
import threading
import uuid

import numpy as np


class QueueExhausted(RuntimeError):
    pass


class RTCActionQueue:
    def __init__(self, horizon=50, action_dim=14):
        self.horizon, self.action_dim = horizon, action_dim
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.session_id = uuid.uuid4().hex
            self.tick = 0
            self._next_id = 0
            self._pending = None
            self.last_merge = None
            self._actions = np.empty((0, self.action_dim), dtype=np.float32)

    def begin(self, *, mode="guided", delay=0):
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("only one inference request may be in flight")
            if mode not in ("off", "guided"):
                raise ValueError("invalid RTC mode")
            if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)) or delay < 0:
                raise ValueError("delay must be a nonnegative integer")
            if len(self._actions) and delay > len(self._actions):
                raise QueueExhausted("insufficient old trajectory for inference budget")
            request = {"session_id": self.session_id, "request_id": self._next_id,
                       "observation_tick": self.tick, "mode": mode,
                       "delay": int(delay) if len(self._actions) else 0}
            if len(self._actions):
                request["prefix"] = self._actions.copy()
                request["prefix_attention_horizon"] = len(self._actions)
            self._next_id += 1
            self._pending = {**request, "coverage": len(self._actions)}
            return request

    def merge(self, response):
        with self._lock:
            info = response.get("rtc_info", {})
            pending = self._pending
            if pending is None or any(info.get(k) != pending[k] for k in ("session_id", "request_id", "observation_tick")):
                return False
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.shape != (self.horizon, self.action_dim) or not np.isfinite(actions).all():
                raise ValueError("invalid returned chunk")
            elapsed = self.tick - pending["observation_tick"]
            if elapsed >= self.horizon or (pending["coverage"] and elapsed > pending["coverage"]):
                self._pending = None
                raise QueueExhausted("response arrived outside old trajectory coverage")
            # A response can arrive before the promised delay prefix has been
            # consumed. Keep those remaining old targets EXACTLY, then switch
            # to the guided/filtered future at the promised tick.
            preserve = max(0, pending["delay"] - elapsed)
            carry = self._actions[:preserve]
            self._actions = np.concatenate((carry, actions[elapsed+preserve:]), axis=0)
            self.last_merge = {"actual_delay": elapsed, "budget": pending["delay"],
                               "preserved_old_steps": preserve, "budget_overrun": elapsed > pending["delay"]}
            self._pending = None
            return True

    def pop(self):
        with self._lock:
            if not len(self._actions):
                raise QueueExhausted("no target for current tick; caller must stop/hold")
            action = self._actions[0].copy()
            self._actions = self._actions[1:]
            self.tick += 1
            return action

    @property
    def remaining(self):
        with self._lock:
            return len(self._actions)
