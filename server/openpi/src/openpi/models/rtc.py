"""Inference-time RTC for OpenPI's noise-at-one flow convention.

Adapted from Physical-Intelligence/real-time-chunking-kinetix (MIT).
See deployment/RTC_THIRD_PARTY_LICENSE.txt. No model parameter gradients.
"""
import jax
import jax.numpy as jnp


def prefix_weights(delay, end, horizon):
    """Full commitment before delay, exponentially decaying attention until end."""
    start = jnp.minimum(delay, end)
    index = jnp.arange(horizon)
    weight = jnp.clip((start - 1 - index) / (end - start + 1) + 1, 0, 1)
    weight = weight * jnp.expm1(weight) / jnp.expm1(1.0)
    weight = jnp.where(index < start, 1.0, weight)
    return jnp.where(index < end, weight, 0.0)


def corrected_velocity(velocity, x, time, target, weights, max_guidance=10.0):
    """Return v - guidance in reverse time; estimated clean action is x - t*v."""
    def denoiser(value):
        v = velocity(value)
        return value - time * v, v

    clean, pullback, v = jax.vjp(denoiser, x, has_aux=True)
    correction = pullback((target - clean) * weights)[0]
    # Equivalent to the author's forward-time formula under tau = 1 - time.
    # Algebraically simplify before evaluating the endpoints to avoid 0*inf.
    t = jnp.clip(time, 1e-6, 1.0 - 1e-6)
    strength = jnp.minimum((t * t + (1 - t) ** 2) / (t * (1 - t)), max_guidance)
    return v - strength * correction
