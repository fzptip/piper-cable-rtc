"""Offline diagnostics tests. Never connect a socket, initialize ROS or move a robot."""

import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diagnostic_recorder import DiagnosticRecorder
from test_rtc_client import bootstrap, build, client, response, ros


def records(recorder):
    return [json.loads(line) for line in (Path(recorder.run_dir) / "events.jsonl")
            .read_text(encoding="utf-8").splitlines()]


class RecorderTests(unittest.TestCase):
    def test_owns_arrays_and_drains_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {"publish_rate": 30}, "test")
            value = np.arange(14, dtype=np.float32)
            recorder.emit("sample", nested={"position": value})
            value[:] = -1
            summary = recorder.close()
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["dropped_events"], 0)
            saved = records(recorder)
            self.assertEqual(saved[1]["nested"]["position"], list(range(14)))
            self.assertEqual(saved[1]["sequence"], 2)
            self.assertIsInstance(saved[1]["monotonic_ns"], int)
            self.assertIsInstance(saved[1]["wall_time_ns"], int)
            self.assertFalse(recorder.emit("after_close"))
            manifest = json.loads((Path(recorder.run_dir) / "manifest.json").read_text("utf-8"))
            self.assertEqual(set(manifest["source_sha256"]),
                             {"pi05_ws_client.py", "rtc_action_queue.py", "diagnostic_recorder.py", "action_interpolator.py"})
            self.assertEqual(manifest["action_order"][6], "left_gripper")

    def test_full_queue_drops_instead_of_waiting_for_writer(self):
        release = threading.Event()
        actual_run = DiagnosticRecorder._run

        def delayed_run(recorder):
            release.wait(timeout=2)
            actual_run(recorder)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(DiagnosticRecorder, "_run", delayed_run):
                recorder = DiagnosticRecorder(directory, {}, queue_size=1)
            try:
                self.assertFalse(recorder.emit("must_not_block"))
            finally:
                release.set()
                summary = recorder.close()
            self.assertEqual(summary["dropped_events"], 1)
            self.assertFalse(summary["complete"])
            self.assertTrue(summary["writer_stopped"])

    def test_serialization_failure_does_not_escape_to_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            recorder.emit("invalid", value=object())
            summary = recorder.close()
            self.assertFalse(summary["complete"])
            self.assertIn("telemetry writer", summary["error"])
            self.assertTrue(summary["writer_stopped"])

    def test_sessions_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            first = DiagnosticRecorder(directory, {})
            second = DiagnosticRecorder(directory, {})
            first.close()
            second.close()
            self.assertNotEqual(first.run_dir, second.run_dir)


class IntegrationTests(unittest.TestCase):
    def test_telemetry_preserves_targets_and_tracks_carry_prefix_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            _, _, _, enabled = build()
            enabled.diagnostics = recorder
            _, _, _, disabled = build()
            for controller in (enabled, disabled):
                bootstrap(controller)
                for _ in range(30):
                    controller._next_target()
            requests = [controller.begin_rtc_request(controller.epoch)
                        for controller in (enabled, disabled)]
            for controller, request in zip((enabled, disabled), requests):
                for _ in range(3):
                    controller._next_target()
                controller.submit_rtc_result(response(request, 1000), controller.epoch)
            enabled_sources = []
            for _ in range(3):
                np.testing.assert_array_equal(enabled._next_target(), disabled._next_target())
                enabled_sources.append(enabled._diagnostic_source)
            self.assertEqual([source["action_index"] for source in enabled_sources], [33, 34, 5])
            self.assertEqual([source["request_id"] for source in enabled_sources], [0, 0, 1])
            self.assertEqual(enabled.rtc_queue.tick, disabled.rtc_queue.tick)
            self.assertTrue(recorder.close()["complete"])
            self.assertEqual(sum(row["event"] == "chunk_merged" for row in records(recorder)), 2)

    def test_command_records_are_the_published_values(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            _, _, publisher, controller = build()
            controller.diagnostics = recorder
            bootstrap(controller)
            publisher.publish.return_value = 123.25
            rate = Mock()
            rate.sleep.side_effect = controller.stop_event.set
            with patch.object(ros, "Rate", return_value=rate, create=True):
                controller._run()
            self.assertFalse(controller.error_event.is_set())
            published = publisher.publish.call_args.args[0]
            self.assertTrue(recorder.close()["complete"])
            commands = [row for row in records(recorder) if row["event"] == "command"]
            self.assertEqual(len(commands), 1)
            np.testing.assert_array_equal(commands[0]["target"], published)
            self.assertTrue(commands[0]["publish_ok"])
            self.assertEqual(commands[0]["ros_stamp_s"], 123.25)
            self.assertEqual(commands[0]["source"]["action_index"], 0)

    def test_markers_do_not_change_control_or_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            _, _, publisher, controller = build()
            controller.diagnostics = recorder
            bootstrap(controller)
            before = (controller.state, controller.epoch, controller.rtc_queue.tick,
                      controller.rtc_queue.remaining)
            controller.record_marker("contact")
            self.assertEqual(before, (controller.state, controller.epoch, controller.rtc_queue.tick,
                                      controller.rtc_queue.remaining))
            publisher.publish.assert_not_called()
            recorder.close()
            marker = next(row for row in records(recorder) if row["event"] == "marker")
            self.assertEqual(marker["label"], "contact")

    def test_command_uses_publishers_float32_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            _, observation, publisher, controller = build()
            controller.diagnostics = recorder
            controller.state = client.ControlState.AT_A
            observation.get_qpos = lambda: np.full(14, 0.123456789, dtype=np.float64)
            publisher.publish.return_value = 123.25
            rate = Mock()
            rate.sleep.side_effect = controller.stop_event.set
            with patch.object(ros, "Rate", return_value=rate, create=True):
                controller._run()
            self.assertTrue(recorder.close()["complete"])
            command = next(row for row in records(recorder) if row["event"] == "command")
            np.testing.assert_array_equal(command["target"],
                                          np.asarray(publisher.publish.call_args.args[0], dtype=np.float32))

    def test_feedback_records_both_callbacks_without_inventing_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            observation = object.__new__(client.RosObservation)
            observation.lock = threading.Lock()
            observation.diagnostics = recorder
            observation.left_seq = observation.right_seq = 0
            observation.feedback_times = {"left": {}, "right": {}}
            for side, value, stamp in (("left", 1, 123.0), ("right", 2, 123.02)):
                message = types.SimpleNamespace(
                    header=types.SimpleNamespace(stamp=Mock(to_sec=Mock(return_value=stamp)), seq=7),
                    name=["joint" + str(index) for index in range(7)],
                    position=[value] * 7, velocity=[], effort=[])
                getattr(observation, "_" + side + "_callback")(message)
            recorder.close()
            rows = [row for row in records(recorder) if row["event"] == "feedback"]
            self.assertEqual([row["side"] for row in rows], ["left", "right"])
            self.assertEqual([row["ros_stamp_s"] for row in rows], [123.0, 123.02])
            np.testing.assert_array_equal(observation.get_qpos(), [1] * 7 + [2] * 7)

    def test_policy_records_complete_returned_chunk_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            policy = client.PolicyClient("ws://offline.invalid", 1, diagnostics=recorder)
            policy.ws = Mock()
            policy.ws.recv.return_value = b"response"
            request = {"rtc": {"session_id": "test", "request_id": 3, "observation_tick": 30},
                       "state": np.zeros(14)}
            result = {"actions": np.arange(700, dtype=np.float32).reshape(50, 14),
                      "rtc_info": request["rtc"]}
            with patch.object(client, "packb", return_value=b"request"), \
                 patch.object(client, "unpackb", return_value=result), \
                 patch.object(client.websocket, "ABNF", types.SimpleNamespace(OPCODE_BINARY=2), create=True):
                returned = policy.infer(request)
            self.assertIs(returned, result)
            self.assertTrue(recorder.close()["complete"])
            saved = next(row for row in records(recorder) if row["event"] == "model_response")
            np.testing.assert_array_equal(saved["actions"], result["actions"])
            self.assertEqual(saved["rtc_info"]["request_id"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
