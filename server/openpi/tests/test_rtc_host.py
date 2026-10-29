import unittest

import numpy as np

from openpi.policies.rtc_policy import RTCPolicy, make_weights, smooth_actions
from openpi_client.rtc_action_queue import QueueExhausted, RTCActionQueue


def observation():
    return {"state": np.zeros(14), "images": {k: np.zeros((224, 224, 3), np.uint8)
            for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}


class FakePolicy:
    metadata = {}

    def infer(self, obs, *, noise=None, rtc=None):
        self.rtc = rtc
        return {"actions": np.tile(np.arange(50, dtype=np.float32)[:, None], (1, 14))}


class HostTests(unittest.TestCase):
    def test_mask_and_padding(self):
        weight = make_weights(3, 20)
        np.testing.assert_array_equal(weight[:3, :14], 1)
        self.assertTrue(np.all(np.diff(weight[3:20, 0]) < 0))
        np.testing.assert_array_equal(weight[20:], 0)
        np.testing.assert_array_equal(weight[:, 14:], 0)
        np.testing.assert_array_equal(make_weights(0, 0), 0)

    def test_filter_preserves_grippers_and_committed_prefix(self):
        rng = np.random.default_rng(4)
        actions = rng.normal(size=(50, 14)).astype(np.float32)
        filtered = smooth_actions(actions, 0.5, 3, np.zeros(14))
        np.testing.assert_array_equal(filtered[:3], actions[:3])
        np.testing.assert_array_equal(filtered[:, [6, 13]], actions[:, [6, 13]])
        self.assertLess(np.mean(np.diff(filtered[3:, 0])**2), np.mean(np.diff(actions[3:, 0])**2))
        np.testing.assert_array_equal(smooth_actions(actions, 1), actions)

    def test_wire_contract(self):
        base = FakePolicy()
        policy = RTCPolicy(base)
        queue = RTCActionQueue()
        req = queue.begin(delay=3)
        first = policy.infer({**observation(), "rtc": req})
        self.assertFalse(first["rtc_info"]["guidance_applied"])
        self.assertTrue(queue.merge(first))
        for _ in range(30): queue.pop()
        req = queue.begin(delay=3)
        saved = req["prefix"].copy()
        second = policy.infer({**observation(), "rtc": req})
        np.testing.assert_array_equal(req["prefix"], saved)
        self.assertTrue(second["rtc_info"]["guidance_applied"])
        np.testing.assert_array_equal(base.rtc["prefix"][:20], saved)
        for _ in range(3): queue.pop()
        self.assertTrue(queue.merge(second))
        self.assertEqual(queue.tick, 33)
        np.testing.assert_array_equal(queue.pop(), np.full(14, 3))
        self.assertFalse(queue.merge(second))

    def test_no_cross_session_or_stale_results(self):
        queue = RTCActionQueue()
        old = queue.begin()
        queue.reset()
        new = queue.begin()
        self.assertFalse(queue.merge({"actions": np.ones((50, 14)), "rtc_info": old}))
        self.assertTrue(queue.merge({"actions": np.ones((50, 14)), "rtc_info": new}))

    def test_early_response_preserves_remaining_commitment(self):
        queue = RTCActionQueue()
        req = queue.begin()
        old = np.tile(np.arange(50, dtype=np.float32)[:, None], (1,14))
        queue.merge({"actions": old, "rtc_info": req})
        for _ in range(30): queue.pop()
        req = queue.begin(delay=5)
        for _ in range(3): queue.pop()
        queue.merge({"actions": old+1000, "rtc_info": req})
        self.assertEqual(queue.last_merge['preserved_old_steps'],2)
        self.assertFalse(queue.last_merge['budget_overrun'])
        self.assertEqual(queue.remaining,47)
        np.testing.assert_array_equal(queue.pop(),np.full(14,33))
        np.testing.assert_array_equal(queue.pop(),np.full(14,34))
        np.testing.assert_array_equal(queue.pop(),np.full(14,1005))

    def test_one_inflight_and_no_replay_on_underrun(self):
        queue = RTCActionQueue()
        request = queue.begin()
        with self.assertRaises(RuntimeError): queue.begin()
        queue.merge({"actions": np.zeros((50, 14)), "rtc_info": request})
        for _ in range(50): queue.pop()
        with self.assertRaises(QueueExhausted): queue.pop()

    def test_invalid_inputs(self):
        policy = RTCPolicy(FakePolicy())
        q = RTCActionQueue()
        req = q.begin()
        for bad in ({**req, "delay": True}, {**req, "delay": 3},
                    {**req, "prefix": np.zeros((2, 14)), "delay": 3},
                    {**req, "prefix": np.full((20, 14), np.nan)},
                    {**req, "smoothing_alpha": 0}, {**req, "smoothing_alpha": "0.5"},
                    {**req, "mode": "trained"}):
            with self.subTest(bad=bad.keys()), self.assertRaises(ValueError):
                policy.infer({**observation(), "rtc": bad})
        bad_obs = observation(); bad_obs["state"][0] = np.nan
        with self.assertRaises(ValueError): policy.infer(bad_obs)

    def test_request_filter_override(self):
        policy = RTCPolicy(FakePolicy())
        req = RTCActionQueue().begin()
        result = policy.infer({**observation(), "rtc": {**req, "smoothing_alpha": 0.5}})
        self.assertEqual(result["rtc_info"]["smoothing_alpha"], 0.5)
        self.assertEqual(result["actions"][1, 0], 0.5)
        self.assertEqual(result["actions"][1, 6], 1.0)

    def test_delay_budget_and_expired_chunk(self):
        queue = RTCActionQueue()
        req = queue.begin()
        queue.merge({"actions": np.zeros((50, 14)), "rtc_info": req})
        for _ in range(30): queue.pop()
        with self.assertRaises(QueueExhausted): queue.begin(delay=21)
        req = queue.begin(delay=3)
        for _ in range(20): queue.pop()
        with self.assertRaises(QueueExhausted): queue.pop()
        self.assertTrue(queue.merge({"actions": np.zeros((50, 14)), "rtc_info": req}))


if __name__ == "__main__": unittest.main(verbosity=2)
