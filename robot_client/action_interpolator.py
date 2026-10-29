"""Monotonic model clock and joint-only linear sampling; no ROS or networking."""

import math

import numpy as np

from rtc_action_queue import JOINT_INDICES


class InterpolationTimingError(RuntimeError):
    pass


class LinearActionInterpolator:
    """Advance at model time, never at the number of publisher callbacks.

    The caller owns/commits the two endpoints. Grippers use zero-order hold on
    the current knot and change only at model boundaries. A scheduling gap of
    more than one whole model period is a fault, not permission to skip targets.
    """

    def __init__(self, action_rate):
        if not math.isfinite(action_rate) or action_rate <= 0:
            raise ValueError("action rate must be finite and positive")
        self.rate = float(action_rate)
        self.reset()

    def reset(self):
        self.origin = None
        self.last_time = None
        self.index = 0

    def start(self, now):
        if not math.isfinite(now):
            raise InterpolationTimingError("non-finite monotonic time")
        self.origin = self.last_time = now
        self.index = 0

    def sample(self, now):
        if self.origin is None:
            raise InterpolationTimingError("model clock not started")
        if not math.isfinite(now) or now < self.last_time:
            raise InterpolationTimingError("invalid or backwards monotonic time")
        if now - self.last_time > (1.0 + 1e-6) / self.rate:
            raise InterpolationTimingError("publisher gap exceeded one model period; refusing catch-up")
        elapsed = (now - self.origin) * self.rate
        index = int(math.floor(elapsed + 1e-7))
        advance = index - self.index
        if not 0 <= advance <= 1:
            raise InterpolationTimingError("model boundary skipped; refusing catch-up")
        phase = max(0.0, min(1.0, elapsed - index))
        self.last_time, self.index = now, index
        return advance, phase

    @staticmethod
    def target(start, end, phase):
        result = np.asarray(start, dtype=np.float32).copy()
        if end is not None and phase > 0:
            left = result[JOINT_INDICES].astype(np.float64)
            right = np.asarray(end)[JOINT_INDICES].astype(np.float64)
            result[JOINT_INDICES] = left + phase * (right - left)
        return result
