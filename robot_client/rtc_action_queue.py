"""Thread-safe tick queue for a single asynchronous RTC inference worker.

This module issues absolute dataset-space targets only; it never drives hardware.
The consumer must stop/hold explicitly on QueueExhausted, never replay old targets.
"""
import collections
import threading
import uuid

import numpy as np


class QueueExhausted(RuntimeError):
    pass


JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def bridge_joint_targets(future, anchor, previous_step, steps):
    """Cubic Hermite bridge in action-tick time. Grippers/tail stay exact.

    At tick 0 start from the last committed target and its last finite-difference
    velocity. Join the model again at tick N with the following model velocity.
    This is a short boundary transition, not a filter on the entire chunk, and
    does not assert hardware velocity/acceleration limits.
    """
    result = np.asarray(future, dtype=np.float32).copy()
    count = min(int(steps), len(result) - 1)
    if count < 2:
        return result, 0
    start = np.asarray(anchor, dtype=np.float64)[JOINT_INDICES]
    start_step = np.asarray(previous_step, dtype=np.float64)[JOINT_INDICES]
    goal = result[count - 1, JOINT_INDICES].astype(np.float64)
    end_step = (result[count, JOINT_INDICES].astype(np.float64) - goal)
    phase = (np.arange(1, count + 1, dtype=np.float64) / count)[:, None]
    h00 = 2 * phase ** 3 - 3 * phase ** 2 + 1
    h10 = phase ** 3 - 2 * phase ** 2 + phase
    h01 = -2 * phase ** 3 + 3 * phase ** 2
    h11 = phase ** 3 - phase ** 2
    result[:count, JOINT_INDICES] = (
        h00 * start + h10 * count * start_step + h01 * goal + h11 * count * end_step)
    # Preserve the join point bit-for-bit, not just within floating-point tolerance.
    result[count - 1] = future[count - 1]
    return result, count


class RTCActionQueue:
    def __init__(self, horizon=50, action_dim=14, transition_steps=0):
        if (isinstance(transition_steps, bool) or not isinstance(transition_steps, (int, np.integer))
                or transition_steps < 0 or transition_steps == 1 or transition_steps >= horizon):
            raise ValueError("transition_steps must be 0 or an integer in 2..horizon-1")
        if transition_steps and action_dim != 14:
            raise ValueError("joint-only transition requires 14-dimensional Piper targets")
        self.horizon, self.action_dim = horizon, action_dim
        self.transition_steps = int(transition_steps)
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
            self._history = collections.deque(maxlen=2)
            self._reserved_next = False

    def begin(self, *, mode="guided", delay=0, prefix_attention_horizon=None):
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("only one inference request may be in flight")
            if mode not in ("off", "guided"):
                raise ValueError("invalid RTC mode")
            if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)) or delay < 0:
                raise ValueError("delay must be a nonnegative integer")
            # An interpolation endpoint is already in use even though its model
            # tick has not been consumed. Expose that commitment to the server.
            delay = max(int(delay), int(self._reserved_next))
            if len(self._actions) and delay > len(self._actions):
                raise QueueExhausted("insufficient old trajectory for inference budget")
            if prefix_attention_horizon is not None:
                if (isinstance(prefix_attention_horizon, bool)
                        or not isinstance(prefix_attention_horizon, (int, np.integer))
                        or not delay <= prefix_attention_horizon <= self.horizon):
                    raise ValueError("prefix attention horizon must cover delay and fit horizon")
            request = {"session_id": self.session_id, "request_id": self._next_id,
                       "observation_tick": self.tick, "mode": mode,
                       "delay": int(delay) if len(self._actions) else 0}
            if len(self._actions):
                request["prefix"] = self._actions.copy()
                request["prefix_attention_horizon"] = (len(self._actions) if prefix_attention_horizon is None
                                                       else min(len(self._actions), int(prefix_attention_horizon)))
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
            preserve = max(int(self._reserved_next), pending["delay"] - elapsed, 0)
            carry = self._actions[:preserve]
            future = actions[elapsed+preserve:].copy()
            transition_count = 0
            join_before, join_after = None, None
            history = list(self._history) + list(carry)
            if self.transition_steps and pending["coverage"] and history and len(future):
                anchor = history[-1]
                previous_step = anchor - history[-2] if len(history) >= 2 else np.zeros_like(anchor)
                join_before = float(np.max(np.abs(future[0, JOINT_INDICES] - anchor[JOINT_INDICES])))
                future, transition_count = bridge_joint_targets(future, anchor, previous_step, self.transition_steps)
                if not np.isfinite(future).all():
                    raise ValueError("boundary transition produced a non-finite target")
                join_after = float(np.max(np.abs(future[0, JOINT_INDICES] - anchor[JOINT_INDICES])))
            self._actions = np.concatenate((carry, future), axis=0)
            self.last_merge = {"actual_delay": elapsed, "budget": pending["delay"],
                               "preserved_old_steps": preserve, "budget_overrun": elapsed > pending["delay"],
                               "reserved_interpolation_steps": int(self._reserved_next),
                               "transition_steps": transition_count,
                               "transition_start_tick": self.tick + preserve,
                               "raw_join_max_joint_step": join_before,
                               "applied_join_max_joint_step": join_after}
            self._pending = None
            return True

    def pop(self):
        with self._lock:
            if not len(self._actions):
                raise QueueExhausted("no target for current tick; caller must stop/hold")
            action = self._actions[0].copy()
            self._actions = self._actions[1:]
            self._history.append(action.copy())
            self._reserved_next = False
            self.tick += 1
            return action

    def reserve_next(self):
        """Peek and commit the next knot until pop/reset, without advancing time.

        Called under the controller lock. merge must preserve this endpoint even
        when the response exceeds its originally promised delay prefix.
        """
        with self._lock:
            if not len(self._actions):
                return None
            self._reserved_next = True
            return self._actions[0].copy()

    def targets(self):
        """Copy of the actual future plan, for logging and offline verification."""
        with self._lock:
            return self._actions.copy()

    @property
    def remaining(self):
        with self._lock:
            return len(self._actions)
