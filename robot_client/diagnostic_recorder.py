"""Optional, bounded asynchronous telemetry. This module never imports ROS."""

import collections
import datetime
import hashlib
import json
import os
import queue
import threading
import time
import uuid

import numpy as np


class NullRecorder:
    enabled = False

    def emit(self, event, **fields):
        return False

    def close(self):
        return {"enabled": False}


NO_DIAGNOSTICS = NullRecorder()


def _snapshot(value):
    # Own mutable data before the producer resumes. Convert arrays on the writer.
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    return value


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("unsupported telemetry type: " + type(value).__name__)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  default=_json_default, allow_nan=False)
        stream.write("\n")


class DiagnosticRecorder:
    enabled = True

    def __init__(self, output_dir, config, label="", queue_size=8192):
        if queue_size < 1:
            raise ValueError("diagnostic queue size must be positive")
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(os.path.abspath(os.path.expanduser(output_dir)),
                                    now + "_" + uuid.uuid4().hex[:8])
        os.makedirs(self.run_dir, exist_ok=False)
        self._queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accepting = True
        self._sequence = 0
        self._dropped = 0
        self._written = 0
        self._counts = collections.Counter()
        self.error = None
        config = {key: value for key, value in config.items() if not key.startswith("_")}
        manifest = {
            "schema_version": 1,
            "created_wall_time_ns": time.time_ns(),
            "created_monotonic_ns": time.monotonic_ns(),
            "label": label,
            "config": config,
            "action_order": [side + "_" + name for side in ("left", "right")
                             for name in ("joint0", "joint1", "joint2", "joint3",
                                          "joint4", "joint5", "gripper")],
            "joint_indices": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
            "gripper_indices": [6, 13],
            "units": "Original client/ROS values; no unit conversion or hardware unit verification.",
            "clocks": {
                "monotonic_ns": "Robot host monotonic clock. Use for local durations and alignment.",
                "wall_time_ns": "Robot host wall clock. May jump; not used for durations.",
                "ros_stamp_s": "ROS message clock. Do not subtract from monotonic or server clocks.",
            },
            "model_actions": "Server-returned absolute targets after RTC and existing output transforms. "
                             "They are before client queueing, but after server EMA if alpha < 1. "
                             "Use alpha=1 to record RTC predictions without EMA.",
            "command": "One client publish attempt; publish_ok means both ROS publish calls returned, "
                       "not hardware receipt or execution confirmation.",
            "interpolation": "When enabled, command.interpolation records the two committed model knots "
                             "and monotonic-clock phase. Joint targets are linearly sampled; grippers hold "
                             "the start knot until the next model boundary. source identifies the start "
                             "knot, end_source identifies the endpoint. queue_tick counts model knots, "
                             "not high-frequency publishes. None endpoints mean hold, not extrapolation.",
            "feedback": "Independent left/right JointState callbacks, not resampled copies of commands. "
                        "Velocity/effort fields are unverified driver fields, not certified force measurements.",
            "queue_capacity_events": queue_size,
            "completion": "Read summary.json for loss/error status. A missing summary is incomplete.",
            "source_sha256": {},
        }
        for name in ("pi05_ws_client.py", "rtc_action_queue.py", "diagnostic_recorder.py", "action_interpolator.py"):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
            with open(path, "rb") as source:
                manifest["source_sha256"][name] = hashlib.sha256(source.read()).hexdigest()
        _write_json(os.path.join(self.run_dir, "manifest.json"), manifest)
        # Open before starting control, so an invalid output path fails at startup.
        self._stream = open(os.path.join(self.run_dir, "events.jsonl"), "w",
                            encoding="utf-8", newline="\n")
        self._thread = threading.Thread(target=self._run, name="diagnostic_writer", daemon=True)
        self._thread.start()
        self.emit("recorder_started")

    def emit(self, event, **fields):
        # No disk I/O and no waiting for queue capacity on the control/callback threads.
        try:
            record = {"event": event, "monotonic_ns": time.monotonic_ns(),
                      "wall_time_ns": time.time_ns(), **_snapshot(fields)}
            with self._lock:
                if not self._accepting:
                    return False
                self._sequence += 1
                record["sequence"] = self._sequence
                if self.error is not None:
                    self._dropped += 1
                    return False
                try:
                    self._queue.put_nowait(record)
                    return True
                except queue.Full:
                    self._dropped += 1
                    return False
        except Exception as exc:
            # A telemetry failure must not trigger a robot state transition.
            with self._lock:
                self._dropped += 1
                self.error = "telemetry snapshot: " + str(exc)
            return False

    def _run(self):
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if time.monotonic() - last_flush >= 0.5:
                        self._stream.flush()
                        last_flush = time.monotonic()
                    continue
                self._stream.write(json.dumps(record, ensure_ascii=False,
                                              default=_json_default, allow_nan=False) + "\n")
                self._written += 1
                self._counts[record["event"]] += 1
                if time.monotonic() - last_flush >= 0.5:
                    self._stream.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            with self._lock:
                self.error = "telemetry writer: " + str(exc)
        finally:
            try:
                self._stream.close()
            except Exception as exc:
                self.error = "telemetry close: " + str(exc)

    def close(self):
        with self._lock:
            self._accepting = False
        self._stop.set()
        self._thread.join(timeout=5.0)
        summary = {
            "schema_version": 1, "closed_wall_time_ns": time.time_ns(),
            "writer_stopped": not self._thread.is_alive(),
            "attempted_events": self._sequence, "written_events": self._written,
            "dropped_events": self._dropped, "unwritten_queue_events": self._queue.qsize(),
            "event_counts": dict(self._counts), "error": self.error,
        }
        summary["complete"] = (summary["writer_stopped"] and self.error is None
                               and self._dropped == 0 and self._written == self._sequence)
        # Closing happens after the controller stops. A slow disk cannot delay a command.
        try:
            if summary["writer_stopped"]:
                _write_json(os.path.join(self.run_dir, "summary.json"), summary)
        except Exception as exc:
            summary["error"] = "telemetry summary: " + str(exc)
            summary["complete"] = False
        return summary
