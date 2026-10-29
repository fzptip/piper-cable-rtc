"""Offline regression for opt-in RTC boundary transitions and reconnects."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rtc_action_queue import JOINT_INDICES, QueueExhausted, RTCActionQueue, bridge_joint_targets
from diagnostic_recorder import DiagnosticRecorder
from test_rtc_client import build, client, response, ros


def bootstrap(queue):
    request = queue.begin()
    queue.merge(response(request))


class BridgeTests(unittest.TestCase):
    def test_disabled_path_is_exact_and_copies_input(self):
        raw = np.arange(280, dtype=np.float32).reshape(20, 14)
        actual, count = bridge_joint_targets(raw, np.zeros(14), np.zeros(14), 0)
        np.testing.assert_array_equal(actual, raw)
        self.assertEqual(count, 0)
        self.assertFalse(np.shares_memory(raw, actual))

    def test_step_change_is_spread_and_endpoint_tail_grippers_stay_exact(self):
        raw = np.ones((20, 14), dtype=np.float32)
        raw[:, 6] = np.arange(20)
        raw[:, 13] = -np.arange(20)
        actual, count = bridge_joint_targets(raw, np.zeros(14), np.zeros(14), 6)
        self.assertEqual(count, 6)
        self.assertLess(float(actual[0, 0]), 0.1)
        self.assertLess(float(np.max(np.abs(np.diff(np.vstack((np.zeros(14), actual)), axis=0)[:, JOINT_INDICES]))), 0.3)
        np.testing.assert_array_equal(actual[:, [6, 13]], raw[:, [6, 13]])
        np.testing.assert_array_equal(actual[5:], raw[5:])
        np.testing.assert_array_equal(raw[:, JOINT_INDICES], np.ones((20, 12)))

    def test_consistent_linear_motion_is_not_changed(self):
        anchor = np.linspace(-1, 1, 14, dtype=np.float32)
        velocity = np.linspace(0.001, 0.01, 14, dtype=np.float32)
        raw = anchor + np.arange(1, 21)[:, None] * velocity
        actual, _ = bridge_joint_targets(raw.astype(np.float32), anchor, velocity, 6)
        np.testing.assert_allclose(actual, raw, atol=2e-7, rtol=1e-6)

    def test_constant_goal_has_no_overshoot(self):
        raw = np.tile(np.linspace(-2, 2, 14, dtype=np.float32), (20, 1))
        actual, _ = bridge_joint_targets(raw, np.zeros(14), np.zeros(14), 6)
        lo, hi = np.minimum(raw[0], 0), np.maximum(raw[0], 0)
        self.assertTrue(np.all(actual[:6] >= lo - 1e-7))
        self.assertTrue(np.all(actual[:6] <= hi + 1e-7))

    def test_short_tail_does_not_invent_or_repeat_actions(self):
        raw = np.ones((3, 14), dtype=np.float32)
        actual, count = bridge_joint_targets(raw, np.zeros(14), np.zeros(14), 6)
        self.assertEqual(actual.shape, raw.shape)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(actual[1:], raw[1:])

    def test_invalid_configuration_is_rejected(self):
        for steps in (-1, 1, 50, True, 2.5):
            with self.assertRaises(ValueError): RTCActionQueue(transition_steps=steps)
        with self.assertRaises(ValueError): RTCActionQueue(action_dim=1, transition_steps=6)


class QueueV2Tests(unittest.TestCase):
    def test_earlier_request_has_more_coverage_without_longer_attention(self):
        queue = RTCActionQueue(transition_steps=6)
        bootstrap(queue)
        for _ in range(15): queue.pop()
        request = queue.begin(delay=5, prefix_attention_horizon=20)
        self.assertEqual(request["prefix"].shape, (35, 14))
        self.assertEqual(request["prefix_attention_horizon"], 20)

    def test_committed_prefix_exact_and_next_request_sees_actual_transition(self):
        queue = RTCActionQueue(transition_steps=6)
        bootstrap(queue)
        for _ in range(30): queue.pop()
        request = queue.begin(delay=5, prefix_attention_horizon=20)
        for _ in range(3): queue.pop()
        raw = response(request, 1000)
        queue.merge(raw)
        planned = queue.targets()
        np.testing.assert_array_equal(planned[:2], np.tile([[33], [34]], (1, 14)))
        np.testing.assert_array_equal(planned[2:, [6, 13]], raw["actions"][5:, [6, 13]])
        np.testing.assert_array_equal(planned[7:], raw["actions"][10:])
        self.assertEqual(len(planned), 47)
        self.assertEqual(queue.last_merge["transition_steps"], 6)
        self.assertLess(queue.last_merge["applied_join_max_joint_step"], queue.last_merge["raw_join_max_joint_step"])
        next_request = queue.begin(delay=5, prefix_attention_horizon=20)
        np.testing.assert_array_equal(next_request["prefix"], planned)

    def test_observed_1062ms_delay_fits_new_coverage(self):
        for replan, expected in ((30, False), (15, True)):
            queue = RTCActionQueue(transition_steps=6)
            bootstrap(queue)
            for _ in range(replan): queue.pop()
            request = queue.begin(delay=5, prefix_attention_horizon=20)
            survived = True
            try:
                for _ in range(32): queue.pop()
                self.assertTrue(queue.merge(response(request, 1)))
            except QueueExhausted:
                survived = False
            self.assertEqual(survived, expected)

    def test_unbounded_delay_still_exhausts_and_stale_reply_cannot_resume(self):
        queue = RTCActionQueue(transition_steps=6)
        bootstrap(queue)
        for _ in range(15): queue.pop()
        request = queue.begin(delay=5, prefix_attention_horizon=20)
        for _ in range(35): queue.pop()
        with self.assertRaises(QueueExhausted): queue.pop()
        queue.reset()
        self.assertFalse(queue.merge(response(request)))
        self.assertEqual(queue.remaining, 0)

    def test_reset_drops_transition_history(self):
        queue = RTCActionQueue(transition_steps=6)
        bootstrap(queue)
        queue.pop(); queue.pop(); queue.reset(); bootstrap(queue)
        self.assertEqual(queue.last_merge["transition_steps"], 0)
        np.testing.assert_array_equal(queue.pop(), np.zeros(14))

    def test_attention_length_must_cover_committed_prefix(self):
        queue = RTCActionQueue(); bootstrap(queue)
        with self.assertRaises(ValueError): queue.begin(delay=5, prefix_attention_horizon=4)


class ClientV2Tests(unittest.TestCase):
    def test_new_session_discards_idle_connection(self):
        policy = client.PolicyClient("ws://offline.invalid", 1)
        socket = Mock()
        policy.ws = socket
        policy.begin_session()
        socket.close.assert_called_once()
        self.assertIsNone(policy.ws)

    def test_close_failure_does_not_keep_stale_socket(self):
        policy = client.PolicyClient("ws://offline.invalid", 1)
        policy.ws = Mock()
        policy.ws.close.side_effect = RuntimeError("already closed")
        policy.begin_session()
        self.assertIsNone(policy.ws)

    def test_empty_response_is_connection_error_not_server_error(self):
        for empty in ("", b""):
            policy = client.PolicyClient("ws://offline.invalid", 1)
            policy.ws = Mock(); policy.ws.recv.return_value = empty
            with patch.object(client, "packb", return_value=b"request"), \
                 patch.object(client.websocket, "ABNF", types.SimpleNamespace(OPCODE_BINARY=2), create=True):
                with self.assertRaises(ConnectionError): policy.infer({})

    def test_new_session_happens_once_before_each_policy_epoch(self):
        args, observation, _, controller = build()
        observation.get = lambda request_snapshot=None: {"rtc": request_snapshot()}
        policy = Mock()
        policy.server_metadata = {"rtc_protocol": 1, "action_horizon": 50, "action_dim": 14}
        policy.infer.side_effect = lambda obs: response(obs["rtc"])
        with patch.object(ros, "is_shutdown", side_effect=lambda: policy.infer.call_count >= 2):
            client.run_rtc_policy_loop(args, controller, observation, policy)
        policy.begin_session.assert_called_once()

    def test_modified_plan_and_origin_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {})
            _, _, _, controller = build()
            controller.diagnostics = recorder
            controller.rtc_queue = RTCActionQueue(transition_steps=6)
            request = controller.begin_rtc_request(controller.epoch)
            controller.submit_rtc_result(response(request), controller.epoch)
            for _ in range(30): controller._next_target()
            request = controller.begin_rtc_request(controller.epoch)
            for _ in range(3): controller._next_target()
            controller.submit_rtc_result(response(request, 1000), controller.epoch)
            for _ in range(3): controller._next_target()
            self.assertTrue(controller._diagnostic_source["transition_applied"])
            self.assertTrue(recorder.close()["complete"])
            events = [json.loads(line) for line in (Path(recorder.run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            merge = [event for event in events if event["event"] == "chunk_merged"][-1]
            self.assertEqual(merge["merge"]["transition_steps"], 6)
            self.assertEqual(len(merge["planned_targets"]), merge["queue_remaining"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
