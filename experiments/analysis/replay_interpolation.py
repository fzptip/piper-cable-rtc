"""Replay the archived 2026-10-16 dataset only; ROS is mocked, no robot calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot_client"))
from action_interpolator import LinearActionInterpolator
from rtc_action_queue import JOINT_INDICES, RTCActionQueue
from test_action_interpolation import interpolated_control, bootstrap
from test_rtc_client import client


def stats(values):
    return {"p50": float(np.median(values)), "p99": float(np.percentile(values, 99)),
            "maximum": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True,
                        help="Archived 2026-10-16 diagnostics directory (292 chunks)")
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Matching historical boundaries.csv and latency.csv directory")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/interpolation_replay")
    args = parser.parse_args()
    analysis = args.analysis_dir
    for required in (analysis / "boundaries.csv", analysis / "latency.csv"):
        if not required.is_file():
            parser.error("Missing historical analysis input: " + str(required))
    if not args.diagnostics.is_dir():
        parser.error("Missing historical diagnostics directory: " + str(args.diagnostics))
    args.output.mkdir(parents=True, exist_ok=True)
    with (analysis / "boundaries.csv").open(encoding="utf-8-sig") as stream:
        boundaries = {(r["run"], int(r["epoch"]), int(r["request_id"])): r
                      for r in csv.DictReader(stream)}
    with (analysis / "latency.csv").open(encoding="utf-8-sig") as stream:
        delays = {(r["run"], int(r["epoch"]), int(r["request_id"])): int(r["actual_delay_steps"])
                  for r in csv.DictReader(stream) if r["actual_delay_steps"]}
    chunks, windows = [], []
    for folder in sorted(args.diagnostics.iterdir()):
        if not folder.is_dir():
            continue
        models, snapshots = {}, {}
        with (folder / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.startswith(('{"event": "model_response"', '{"event": "rtc_snapshot"')):
                    continue
                event = json.loads(line)
                info = event.get("rtc_info", event.get("rtc"))
                key = info["session_id"], info["request_id"]
                (models if event["event"] == "model_response" else snapshots)[key] = event
        for key, event in models.items():
            raw = np.asarray(event["actions"], dtype=np.float32)
            chunks.append(raw)
            snapshot = snapshots[key]
            identity = (folder.name, snapshot["epoch"], key[1])
            record = boundaries.get(identity)
            if record is None:
                continue
            prefix = np.asarray(snapshot["rtc"]["prefix"], dtype=np.float32)
            previous = np.asarray(models[(key[0], int(record["old_request_id"]))]["actions"], np.float32)
            queue = RTCActionQueue(transition_steps=6)
            request = queue.begin()
            queue.merge({"actions": previous, "rtc_info": request})
            for _ in range(50 - len(prefix)): queue.pop()
            np.testing.assert_array_equal(queue.targets(), prefix)
            request = queue.begin(delay=snapshot["rtc"]["delay"], prefix_attention_horizon=20)
            elapsed = delays[identity]
            for _ in range(elapsed): queue.pop()
            # Simulate a reply arriving while interpolation is using next knot.
            endpoint = queue.reserve_next()
            queue.merge({"actions": raw, "rtc_info": request})
            if endpoint is not None:
                np.testing.assert_array_equal(queue.targets()[0], endpoint)
            head = prefix[elapsed - 2:elapsed]
            if len(head) < 2:
                raise AssertionError("Expected at least two executed old knots")
            windows.append(np.vstack((head, queue.targets()[:10])))
    assert len(chunks) == 292 and len(windows) == 235

    elapsed_us, worst_error = [], 0.0
    for rate in (90, 100):
        for raw in chunks:
            _, _, publisher, control = interpolated_control(rate, transition=6)
            bootstrap(control, raw)
            count = int(np.ceil(50 * rate / 30))
            times = np.arange(count) / rate
            actual = []
            for now in times:
                started = time.perf_counter_ns()
                actual.append(control._next_target(float(now)).copy())
                elapsed_us.append((time.perf_counter_ns() - started) / 1000)
                assert control.state == client.ControlState.POLICY
            actual = np.asarray(actual)
            # Independent reference: NumPy interpolation on fixed 30 Hz times.
            expected = np.column_stack([np.interp(times, np.arange(50) / 30, raw[:, joint])
                                        for joint in JOINT_INDICES])
            error = float(np.max(np.abs(actual[:, JOINT_INDICES] - expected)))
            worst_error = max(worst_error, error)
            np.testing.assert_allclose(actual[:, JOINT_INDICES], expected, rtol=0, atol=3e-7)
            indices = np.minimum(np.floor(times * 30 + 1e-7).astype(int), 49)
            np.testing.assert_array_equal(actual[:, [6, 13]], raw[indices][:, [6, 13]])
            if rate == 90:
                np.testing.assert_array_equal(actual[::3], raw)
            last = actual[-1].copy()
            control._next_target(float(count / rate))
            assert control.state == client.ControlState.POLICY_HOLD
            np.testing.assert_array_equal(control.target, last)
            publisher.publish.assert_not_called()

    # These are corrected static windows, not model re-inference or physical rollouts.
    max_step_30, max_step_90 = [], []
    for window in windows:
        high = []
        for index in range(len(window) - 1):
            for phase in (0, 1 / 3, 2 / 3):
                high.append(LinearActionInterpolator.target(window[index], window[index + 1], phase))
        high.append(window[-1])
        high = np.asarray(high)
        np.testing.assert_array_equal(high[::3], window)
        expected_grips = np.repeat(window[:-1, [6, 13]], 3, axis=0)
        np.testing.assert_array_equal(high[:-1, [6, 13]], expected_grips)
        max_step_30.append(float(np.max(np.abs(np.diff(window[:, JOINT_INDICES], axis=0)))))
        max_step_90.append(float(np.max(np.abs(np.diff(high[:, JOINT_INDICES], axis=0)))))
    np.testing.assert_allclose(np.asarray(max_step_90) * 3, max_step_30, atol=5e-7, rtol=0)

    result = {
        "reference_validation_date": "2026-10-19", "scope": "Local offline replay; no network, ROS publishers, GPU inference or physical actuation",
        "historical_chunks": len(chunks), "publisher_rates_tested_hz": [90, 100],
        "chunk_rate_replays": len(chunks) * 2, "model_rate_hz": 30,
        "model_horizon_duration_s": 50 / 30, "model_knots_at_90hz_exact": True,
        "max_joint_difference_vs_independent_numpy_reference": worst_error,
        "gripper_samples_and_timing_exact": True, "exhaustion_hold_checks_passed": True,
        "historical_boundary_windows": len(windows), "reserved_endpoints_preserved": True,
        "full_window_30hz_max_joint_step": stats(max_step_30),
        "full_window_90hz_max_joint_step": stats(max_step_90),
        "local_sampling_cpu_us": stats(elapsed_us),
        "step_metric_caveat": "Step approximately divides by three by construction of resampling; not evidence of lower physical acceleration, model oscillation or improved success rate.",
        "limitations": ["Recorded outputs reused; no new predictions under updated prefixes",
                        "No hardware publication frequency/load validation",
                        "Linear interpolation has velocity changes at model knots",
                        "First model target remains unchanged and can differ from initial pose"],
        "source_sha256": {name: hashlib.sha256((ROOT / "robot_client" / name).read_bytes()).hexdigest()
                          for name in ("pi05_ws_client.py", "rtc_action_queue.py", "action_interpolator.py", "diagnostic_recorder.py")},
    }
    (args.output / "validation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
