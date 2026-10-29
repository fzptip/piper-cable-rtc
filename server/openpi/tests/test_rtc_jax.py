import unittest

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.rtc import corrected_velocity, prefix_weights
from openpi.policies.rtc_policy import make_weights


class GradientTests(unittest.TestCase):
    def test_reference_weights_match(self):
        for delay, end in ((0, 0), (0, 20), (3, 20), (20, 20), (49, 50)):
            np.testing.assert_allclose(prefix_weights(delay, end, 50), make_weights(delay, end)[:, 0], atol=1e-6)

    def test_vjp_sign_and_exact_linear_solution(self):
        x = jnp.array([[[0.4, -0.7]]])
        target = jnp.array([[[1.0, 2.0]]])
        weights = jnp.array([[[1.0, 0.0]]])
        time = 0.4
        velocity = lambda a: 0.3 * a + 0.1
        result = corrected_velocity(velocity, x, time, target, weights, 10)
        v = velocity(x)
        clean = x - time * v
        strength = (time**2 + (1-time)**2) / (time*(1-time))
        expected = v - strength * (1 - time*0.3) * (target-clean) * weights
        np.testing.assert_allclose(result, expected, atol=1e-6)
        # Reverse-time Euler must move the constrained component toward target.
        self.assertGreater(float((x - 0.1*result)[0, 0, 0]), float(x[0, 0, 0]))
        np.testing.assert_allclose(result[..., 1], v[..., 1])

    def test_nonlinear_finite_difference_and_endpoints(self):
        x = jnp.array([[[0.2, -0.4]]])
        target = jnp.ones_like(x)
        time = 0.6
        velocity = lambda a: jnp.sin(a)
        loss = lambda a: 0.5*jnp.sum((target-(a-time*velocity(a)))**2)
        epsilon = 1e-3
        finite = []
        for i in range(2):
            basis = jnp.zeros_like(x).at[0, 0, i].set(epsilon)
            finite.append(float((loss(x+basis)-loss(x-basis))/(2*epsilon)))
        result = corrected_velocity(velocity, x, time, target, jnp.ones_like(x))
        strength = (time**2 + (1-time)**2)/(time*(1-time))
        np.testing.assert_allclose(result, velocity(x)+strength*np.array(finite).reshape(x.shape), atol=2e-4)
        for t in (0.0, 1e-8, 0.5, 1.0):
            self.assertTrue(np.isfinite(corrected_velocity(velocity, x, t, target, jnp.ones_like(x))).all())
        np.testing.assert_allclose(corrected_velocity(velocity, x, time, target, jnp.zeros_like(x)), velocity(x))

    def test_actual_action_transform_rebases_current_state(self):
        from openpi import transforms
        from openpi.shared.normalize import NormStats
        mask = transforms.make_bool_mask(6, -1, 6, -1)
        stats = {"actions": NormStats(mean=np.arange(14)*0.01, std=np.ones(14)*0.5)}
        absolute = np.arange(50*14, dtype=np.float32).reshape(50, 14)*0.001
        for shift in (0.0, 0.8):
            state = np.arange(14, dtype=np.float32)*0.03 + shift
            data = transforms.DeltaActions(mask)({"state": state.copy(), "actions": absolute.copy()})
            normalized = transforms.Normalize(stats)(data)
            self.assertEqual(normalized["actions"].shape, (50, 14))
            recovered = transforms.AbsoluteActions(mask)(transforms.Unnormalize(stats)(normalized))
            np.testing.assert_allclose(recovered["actions"], absolute, atol=2e-7)

    def test_policy_prefix_uses_actual_quantile_pipeline_with_padding(self):
        from openpi import transforms
        from openpi.policies.cobot_magic_policy import CobotMagicInputs, CobotMagicOutputs
        from openpi.policies.policy import Policy
        from openpi.shared.normalize import NormStats
        mask = transforms.make_bool_mask(6, -1, 6, -1)
        stats = {k: NormStats(mean=np.zeros(14), std=np.ones(14), q01=-np.ones(14)*3, q99=np.ones(14)*3)
                 for k in ("state", "actions")}
        policy = object.__new__(Policy)
        policy._is_pytorch_model = False
        policy._sample_kwargs = {}
        policy._rng = jax.random.key(0)
        policy._input_transform = transforms.compose([CobotMagicInputs(), transforms.DeltaActions(mask),
                                  transforms.Normalize(stats, use_quantiles=True), transforms.PadStatesAndActions(32)])
        policy._output_transform = transforms.compose([transforms.Unnormalize(stats, use_quantiles=True),
                                   transforms.AbsoluteActions(mask), CobotMagicOutputs()])
        captured = []
        def sample(rng, obs, **kwargs):
            captured.append(np.array(kwargs["rtc_prefix"]))
            return kwargs["rtc_prefix"]
        policy._sample_actions = sample
        prefix = np.ones((50, 14), dtype=np.float32)*0.3
        for offset in (0.0, 0.8):
            obs = {"state": np.ones(14, np.float32)*offset,
                   "images": {k: np.zeros((224,224,3), np.uint8) for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
            result = policy.infer(obs, rtc={"prefix": prefix, "weights": make_weights(3,20), "max_guidance": 10})
            np.testing.assert_allclose(result["actions"], prefix, atol=5e-7)
            np.testing.assert_array_equal(prefix, np.full((50,14), 0.3, np.float32))
        self.assertEqual(captured[0].shape, (1,50,32))
        self.assertGreater(abs(captured[0][0,0,0]-captured[1][0,0,0]), 0.1)
        np.testing.assert_array_equal(captured[0][..., [6,13]], captured[1][..., [6,13]])


if __name__ == "__main__": unittest.main(verbosity=2)
