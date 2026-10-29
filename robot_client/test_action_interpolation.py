"""Offline interpolation/timing regressions. ROS, publishers and network mocked."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from action_interpolator import LinearActionInterpolator, InterpolationTimingError
from diagnostic_recorder import DiagnosticRecorder
from rtc_action_queue import JOINT_INDICES, RTCActionQueue
from test_rtc_client import build, client, response, ros


def interpolated_control(publish_rate=90, diagnostics=None, transition=6):
    args, observation, publisher, _ = build()
    args.publish_rate = publish_rate
    args.action_rate = 30
    args.action_interpolation = "linear"
    args.chunk_steps = 15
    args.rtc_transition_steps = transition
    args.rtc_prefix_attention_horizon = 20
    client.validate_action_timing(args)
    if diagnostics is not None:
        args._diagnostics = diagnostics
    control = client.ArmController(args, observation, publisher, np.zeros(14), np.ones(14))
    control.state = client.ControlState.POLICY
    control.target = np.zeros(14, np.float32)
    return args, observation, publisher, control


def bootstrap(control, actions=None):
    request = control.begin_rtc_request(control.epoch)
    result = response(request)
    if actions is not None:
        result["actions"] = np.asarray(actions, np.float32)
    control.submit_rtc_result(result, control.epoch)
    return request


class ClockTests(unittest.TestCase):
    def test_subframes_do_not_advance_model_clock(self):
        clock = LinearActionInterpolator(30)
        clock.start(100)
        for subframe in range(1, 10):
            advance, phase = clock.sample(100 + subframe / 90)
            self.assertEqual(advance, int(subframe % 3 == 0))
            self.assertAlmostEqual(phase, (subframe % 3) / 3)

    def test_noninteger_frequency_ratio_and_publication_jitter(self):
        clock = LinearActionInterpolator(30)
        clock.start(50)
        rng = np.random.default_rng(123)
        for frame in range(1, 301):
            now = 50 + frame / 100 + rng.uniform(-0.001, 0.001)
            old_index = clock.index
            advance, phase = clock.sample(now)
            self.assertEqual(clock.index, int((now - 50) * 30))
            self.assertEqual(advance, clock.index - old_index)
            self.assertAlmostEqual(phase + clock.index, (now - 50) * 30)

    def test_stall_does_not_skip_points_or_accelerate_to_catch_up(self):
        clock = LinearActionInterpolator(30)
        clock.start(0)
        clock.sample(0.01)
        with self.assertRaises(InterpolationTimingError): clock.sample(0.1)
        self.assertEqual(clock.index, 0)

    def test_backwards_nonfinite_and_unstarted_clock_rejected(self):
        clock = LinearActionInterpolator(30)
        with self.assertRaises(InterpolationTimingError): clock.sample(0)
        for now in (float("nan"), float("inf")):
            with self.assertRaises(InterpolationTimingError): clock.start(now)
        clock.start(3)
        with self.assertRaises(InterpolationTimingError): clock.sample(2)
        with self.assertRaises(InterpolationTimingError): clock.sample(float("nan"))

    def test_reset_clears_the_old_time_origin(self):
        clock = LinearActionInterpolator(30)
        clock.start(4)
        clock.sample(4.02)
        clock.reset()
        self.assertIsNone(clock.origin)
        clock.start(100)
        self.assertEqual(clock.sample(100.01)[0], 0)

    def test_joint_interpolation_is_bounded_and_grippers_hold(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            start, end = rng.normal(size=(2, 14)).astype(np.float32)
            for phase in (0, 0.1, 0.5, 0.9):
                actual = LinearActionInterpolator.target(start, end, phase)
                self.assertTrue(np.all(actual >= np.minimum(start, end)))
                self.assertTrue(np.all(actual <= np.maximum(start, end)))
                np.testing.assert_array_equal(actual[[6, 13]], start[[6, 13]])
            np.testing.assert_array_equal(LinearActionInterpolator.target(start, None, 0.9), start)


class ReservationTests(unittest.TestCase):
    def test_reply_at_each_phase_early_and_late_preserves_interpolated_endpoint(self):
        for delay_frames in range(0, 34):
            _, _, _, control = interpolated_control()
            bootstrap(control)
            for frame in range(45): control._next_target(frame / 90)
            request = control.begin_rtc_request(control.epoch)
            for frame in range(45, 45 + delay_frames): control._next_target(frame / 90)
            endpoint = control._interpolation_end.copy()
            start = control._interpolation_start.copy()
            phase = control._interpolation_phase
            before = control.target.copy()
            control.submit_rtc_result(response(request, 1000), control.epoch)
            np.testing.assert_array_equal(control._interpolation_start, start)
            np.testing.assert_array_equal(control._interpolation_end, endpoint)
            np.testing.assert_array_equal(control.target, before)
            np.testing.assert_array_equal(control.rtc_queue.targets()[0], endpoint)
            self.assertEqual(control._interpolation_phase, phase)

    def test_reservation_is_in_prefix_without_consuming_a_tick(self):
        queue = RTCActionQueue()
        request = queue.begin()
        queue.merge(response(request))
        queue.pop()
        end = queue.reserve_next()
        request = queue.begin(delay=0)
        self.assertEqual(queue.tick, 1)
        self.assertEqual(queue.remaining, 49)
        self.assertEqual(request["delay"], 1)
        np.testing.assert_array_equal(request["prefix"][0], end)
        end[:] = -100
        np.testing.assert_array_equal(queue.targets()[0], np.ones(14))

    def test_late_response_cannot_replace_the_endpoint_in_use(self):
        queue = RTCActionQueue(transition_steps=6)
        request = queue.begin()
        queue.merge(response(request))
        for _ in range(15): queue.pop()
        request = queue.begin(delay=5, prefix_attention_horizon=20)
        for _ in range(7): queue.pop()
        endpoint = queue.reserve_next()
        queue.merge(response(request, 1000))
        self.assertEqual(queue.last_merge["preserved_old_steps"], 1)
        self.assertEqual(queue.last_merge["reserved_interpolation_steps"], 1)
        np.testing.assert_array_equal(queue.pop(), endpoint)
        # The bridge starts after the protected endpoint and feeds back as plan.
        next_request = queue.begin(delay=5, prefix_attention_horizon=20)
        np.testing.assert_array_equal(next_request["prefix"], queue.targets())

    def test_end_of_response_horizon_can_preserve_only_the_endpoint(self):
        queue = RTCActionQueue(transition_steps=6)
        request = queue.begin(); queue.merge(response(request))
        request = queue.begin(delay=0)
        for _ in range(49): queue.pop()
        endpoint = queue.reserve_next()
        self.assertTrue(queue.merge(response(request, 1000)))
        self.assertEqual(queue.remaining, 1)
        np.testing.assert_array_equal(queue.pop(), endpoint)

    def test_reset_releases_endpoint_reservation(self):
        queue = RTCActionQueue()
        request = queue.begin(); queue.merge(response(request))
        queue.reserve_next(); queue.reset()
        request = queue.begin(delay=0)
        self.assertEqual(request["delay"], 0)
        self.assertNotIn("prefix", request)


class ControllerInterpolationTests(unittest.TestCase):
    def test_50_model_points_take_50_model_periods_at_90hz(self):
        _, _, publisher, control = interpolated_control()
        bootstrap(control)
        samples = []
        for frame in range(150):
            now = 100 + frame / 90
            target = control._next_target(now).copy()
            samples.append(target)
            self.assertEqual(control.rtc_queue.tick, min(frame // 3 + 1, 50))
            self.assertEqual(control.state, client.ControlState.POLICY)
            self.assertAlmostEqual(target[0], min(frame / 3, 49), places=5)
            self.assertEqual(target[6], min(frame // 3, 49))
        # 49th interval holds the last point, then exhaustion pauses at 50/30 s.
        control._next_target(100 + 150 / 90)
        self.assertEqual(control.state, client.ControlState.POLICY_HOLD)
        np.testing.assert_array_equal(control.target, samples[-1])
        publisher.publish.assert_not_called()

    def test_100hz_does_not_change_duration_or_playback_speed(self):
        _, _, _, control = interpolated_control(100)
        bootstrap(control)
        for frame in range(167):
            now = frame / 100
            target = control._next_target(now)
            self.assertAlmostEqual(target[0], min(now * 30, 49), places=5)
        self.assertEqual(control.state, client.ControlState.POLICY)
        control._next_target(1.67)
        self.assertEqual(control.state, client.ControlState.POLICY_HOLD)

    def test_replanning_delay_and_prefix_are_in_30hz_steps(self):
        _, _, _, control = interpolated_control()
        bootstrap(control)
        for frame in range(42):
            control._next_target(frame / 90)
        self.assertEqual(control.rtc_queue.tick, 14)
        self.assertIsNone(control.policy_request())
        control._next_target(42 / 90)
        request = control.begin_rtc_request(control.epoch)
        self.assertEqual(request["observation_tick"], 15)
        self.assertEqual(request["prefix"].shape, (35, 14))
        self.assertEqual(request["delay"], 5)
        self.assertEqual(request["prefix_attention_horizon"], 20)
        # After 80 ms, only two more model boundaries, not seven publisher ticks.
        for frame in range(43, 50): control._next_target(frame / 90)
        control.submit_rtc_result(response(request, 1000), control.epoch)
        self.assertEqual(control.rtc_queue.last_merge["actual_delay"], 2)

    def test_late_merge_during_segment_never_changes_its_endpoint(self):
        _, _, _, control = interpolated_control()
        bootstrap(control)
        for frame in range(45): control._next_target(frame / 90)
        request = control.begin_rtc_request(control.epoch)
        for frame in range(45, 67): control._next_target(frame / 90)
        endpoint = control._interpolation_end.copy()
        control.submit_rtc_result(response(request, 1000), control.epoch)
        self.assertTrue(control.rtc_queue.last_merge["budget_overrun"])
        self.assertEqual(control.rtc_queue.last_merge["preserved_old_steps"], 1)
        for frame in (67, 68):
            actual = control._next_target(frame / 90)
            self.assertAlmostEqual(actual[0], frame / 3, places=5)
        np.testing.assert_array_equal(control._next_target(69 / 90), endpoint)

    def test_loop_stall_holds_the_last_actually_sampled_target(self):
        _, _, _, control = interpolated_control()
        bootstrap(control)
        control._next_target(0)
        last = control._next_target(1 / 90).copy()
        epoch = control.epoch
        control._next_target(0.2)
        self.assertEqual(control.state, client.ControlState.POLICY_HOLD)
        self.assertGreater(control.epoch, epoch)
        np.testing.assert_array_equal(control.target, last)
        self.assertIsNone(control.interpolator.origin)
        self.assertIsNone(control.trajectory)

    def test_network_pause_clears_segment_and_stale_reply_cannot_restart(self):
        _, _, _, control = interpolated_control()
        bootstrap(control)
        for frame in range(45): control._next_target(frame / 90)
        epoch = control.epoch
        request = control.begin_rtc_request(epoch)
        last = control.target.copy()
        control.pause_rtc("simulated timeout", epoch)
        self.assertFalse(control.submit_rtc_result(response(request, 1000), epoch))
        np.testing.assert_array_equal(control._next_target(1), last)
        self.assertIsNone(control._interpolation_end)

    def test_i_and_e_drop_partial_segment_and_cancel_old_response(self):
        for action in ("request_go_a", "request_exit"):
            _, _, _, control = interpolated_control()
            bootstrap(control)
            for frame in range(45): control._next_target(frame / 90)
            epoch = control.epoch
            request = control.begin_rtc_request(epoch)
            getattr(control, action)()
            self.assertIsNone(control.interpolator.origin)
            self.assertFalse(control.submit_rtc_result(response(request), epoch))
            control._process_requests()
            self.assertIn(control.state, (client.ControlState.MOVING_A, client.ControlState.MOVING_B))

    def test_new_session_starts_fresh_without_reusing_old_phase(self):
        _, _, _, control = interpolated_control()
        bootstrap(control)
        control._next_target(0); control._next_target(0.01)
        control.pause_rtc("test", control.epoch)
        control.state = client.ControlState.AT_A
        control.request_start(); control._process_requests()
        bootstrap(control)
        np.testing.assert_array_equal(control._next_target(100), np.zeros(14))
        self.assertEqual(control._interpolation_phase, 0)

    def test_late_tail_response_does_not_restart_from_unseen_endpoint(self):
        _, _, _, control = interpolated_control()
        bootstrap(control)
        for frame in range(45): control._next_target(frame / 90)
        epoch = control.epoch
        request = control.begin_rtc_request(epoch)
        for frame in range(45, 149): control._next_target(frame / 90)
        self.assertIsNone(control._interpolation_end)
        control.submit_rtc_result(response(request, 1000), epoch)
        last = control.target.copy()
        control._next_target(149 / 90)
        np.testing.assert_array_equal(control.target, last)
        control._next_target(150 / 90)
        self.assertEqual(control.state, client.ControlState.POLICY_HOLD)
        np.testing.assert_array_equal(control.target, last)

    def test_disabled_path_has_identical_v2_knots(self):
        _, _, _, plain = build()
        _, _, _, high = interpolated_control(transition=0)
        plain.args.publish_rate = 30
        bootstrap(plain); bootstrap(high)
        for index in range(50):
            original = plain._next_target().copy()
            for subframe in range(3):
                actual = high._next_target((index * 3 + subframe) / 90)
                if subframe == 0: np.testing.assert_array_equal(actual, original)

    def test_ab_duration_and_confirmation_window_do_not_shrink(self):
        args, _, _, control = interpolated_control()
        args.move_duration = 2
        path = control._make_trajectory(np.zeros(14), np.ones(14))
        self.assertEqual(len(path), 180)
        self.assertEqual(control.arrival_stable_cycles, args.stable_cycles * 3)
        np.testing.assert_array_equal(path[-1], np.ones(14))

    def test_no_model_actions_means_hold_not_extrapolation(self):
        _, _, _, control = interpolated_control()
        last = control.target.copy()
        for now in (0, 0.01, 100):
            np.testing.assert_array_equal(control._next_target(now), last)
        self.assertIsNone(control.interpolator.origin)


class ConfigurationAndLoggingTests(unittest.TestCase):
    def test_launcher_defaults_and_overrides_are_parsed_as_expected(self):
        # The wrappers append "$@", so user overrides are deliberately last.
        defaults = ["client", "--rtc-mode", "guided", "--chunk-steps", "15",
                    "--rtc-delay", "5", "--rtc-transition-steps", "6",
                    "--rtc-prefix-attention-horizon", "20", "--publish-rate", "30",
                    "--action-rate", "30", "--publish-rate", "90", "--action-interpolation", "linear"]
        for overrides, publish, method in (([], 90, "linear"),
                                            (["--publish-rate", "100"], 100, "linear"),
                                            (["--publish-rate", "30", "--action-interpolation", "none"], 30, "none")):
            with patch.object(client.sys, "argv", defaults + overrides): args = client.get_arguments()
            client.validate_action_timing(args)
            self.assertEqual(args.action_rate, 30)
            self.assertEqual(args.publish_rate, publish)
            self.assertEqual(args.action_interpolation, method)

    def test_invalid_config_rejected_before_ros_or_publishers(self):
        cases = [dict(action_interpolation="none", action_rate=30, publish_rate=90),
                 dict(action_interpolation="linear", rtc_mode="legacy"),
                 dict(publish_rate=20), dict(publish_rate=float("nan")),
                 dict(action_rate=-1), dict(action_rate=float("inf"))]
        for overrides in cases:
            args, _, _, _ = interpolated_control()
            vars(args).update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError): client._run_client(args)
        ros.Publisher.assert_not_called()

    def test_command_log_reconstructs_joint_targets_and_endpoint_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, {"action_rate": 30, "publish_rate": 90})
            _, _, publisher, control = interpolated_control(diagnostics=recorder)
            bootstrap(control)
            now, published = [100.0], []
            publisher.publish.side_effect = lambda value: published.append(value.copy()) or 123.0
            rate = Mock()
            def sleep():
                now[0] += 1 / 90
                if len(published) >= 12: control.stop_event.set()
            rate.sleep.side_effect = sleep
            with patch.object(client.time, "monotonic", side_effect=lambda: now[0]), \
                 patch.object(ros, "Rate", return_value=rate, create=True):
                control._run()
            self.assertFalse(control.error_event.is_set())
            summary = recorder.close()
            self.assertTrue(summary["complete"])
            with (Path(recorder.run_dir) / "events.jsonl").open(encoding="utf-8") as stream:
                commands = [row for row in map(json.loads, stream) if row["event"] == "command"]
            self.assertEqual(len(commands), 12)
            for row, sent in zip(commands, published):
                info = row["interpolation"]
                rebuilt = LinearActionInterpolator.target(info["start_target"], info["end_target"], info["phase"])
                np.testing.assert_array_equal(rebuilt, sent)
                np.testing.assert_array_equal(row["target"], sent)
                self.assertEqual(info["end_source"]["action_index"], row["source"]["action_index"] + 1)
            self.assertEqual(commands[-1]["queue_tick"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
