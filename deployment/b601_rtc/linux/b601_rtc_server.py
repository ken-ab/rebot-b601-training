"""Inference-time RTC for the pinned JAX OpenPI pi0.5 B601 checkpoint.

Algorithm: Black et al., https://arxiv.org/abs/2506.07339, equations 2-4.
The PI reference uses noise-time 0 -> 1; OpenPI uses 1 -> 0, hence the
minus sign in corrected velocity and x0 = x_t - t*v_t below.
Uses the existing checkpoint's full input/output transforms, including
normalization and any action-space transforms; no weights are changed.
"""

import argparse
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np

RTC_KEY = "b601_rtc"
RTC_PROTOCOL = "pi05-inference-rtc-v1"


def prefix_weights(start, end, total):
    """PI reference exponential soft mask, src/model.py get_prefix_weights."""
    import jax.numpy as jnp
    start = jnp.minimum(start, end)
    w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
    w = w * jnp.expm1(w) / (jnp.e - 1)
    return jnp.where(jnp.arange(total) >= end, 0, w)


def sample_actions_rtc(self, rng, observation, *, previous_actions, inference_delay,
                       prefix_horizon, num_steps=10, noise=None, max_guidance_weight=5.0):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask,
        positions=jnp.cumsum(prefix_mask, axis=1) - 1)
    weights = prefix_weights(inference_delay, prefix_horizon, self.action_horizon)
    # The other 25 dimensions are model padding, not robot actuators.
    weights = weights[None, :, None] * (jnp.arange(self.action_dim) < 7)[None, None, :]

    def velocity(x_t, t):
        tokens, mask, ar_mask, cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(t, batch_size))
        suffix_attn = make_attn_mask(mask, ar_mask)
        prefix_attn = jnp.broadcast_to(prefix_mask[:, None, :],
                                      (batch_size, tokens.shape[1], prefix_tokens.shape[1]))
        full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (_, output), _ = self.PaliGemma.llm(
            [None, tokens], mask=full_attn, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, cond])
        return self.action_out_proj(output[:, -self.action_horizon:])

    def step(carry):
        x_t, t = carry

        def denoiser(x):
            v = velocity(x, t)
            return x - t * v, v

        predicted, pullback, v = jax.vjp(denoiser, x_t, has_aux=True)
        error = (previous_actions - predicted) * weights
        correction = pullback(error)[0]
        # Same beta clipping as PI's inference-time RTC reference.
        denominator = t * (1 - t)
        scale = (t * t + (1 - t) ** 2) / denominator
        gain = jnp.minimum(jnp.nan_to_num(scale, posinf=max_guidance_weight), max_guidance_weight)
        corrected = v - gain * correction
        dt = -1.0 / num_steps
        return x_t + dt * corrected, t + dt

    actions, _ = jax.lax.while_loop(lambda carry: carry[1] >= 0.5 / num_steps,
                                   step, (noise, 1.0))
    return actions


class RTCPolicy:
    def __init__(self, policy):
        from openpi.shared import nnx_utils
        self.base = policy
        self.sampler = nnx_utils.module_jit(MethodType(sample_actions_rtc, policy._model))

    @property
    def metadata(self):
        return {**self.base.metadata, RTC_KEY: RTC_PROTOCOL,
                "rtc_mask": "exp", "rtc_max_guidance_weight": 5.0,
                "rtc_algorithm_source": "arxiv:2506.07339"}

    def infer(self, obs, *, noise=None):
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation
        obs = dict(obs)
        rtc = obs.pop(RTC_KEY, None)
        if rtc is None:
            return self.base.infer(obs, noise=noise)
        if rtc.get("protocol") != RTC_PROTOCOL:
            raise ValueError("Unsupported RTC protocol")
        previous = np.asarray(rtc["previous_actions"], dtype=np.float32)
        delay, horizon = int(rtc["inference_delay"]), int(rtc["prefix_horizon"])
        if (previous.shape != (25, 7) or not np.isfinite(previous).all()
                or not 0 <= delay <= 25 or not 0 <= horizon <= 25):
            raise ValueError("Invalid RTC prefix/indices")
        response_context = {"protocol": RTC_PROTOCOL, "request_id": rtc["request_id"],
                            "observation_tick": rtc["observation_tick"],
                            "inference_delay": delay, "prefix_horizon": horizon}
        if horizon == 0:
            # No overlapping committed actions: use the original sampler
            # exactly (also avoids paying for an unnecessary VJP).
            outputs = self.base.infer(obs, noise=noise)
            outputs[RTC_KEY] = response_context
            return outputs
        # B601Inputs consumes 'action', not 'actions'. Carry it through the
        # exact training transform pipeline, then separate it from observation.
        inputs = self.base._input_transform({**obs, "action": previous.copy()})
        normalized_previous = inputs.pop("actions")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self.base._rng, rng = jax.random.split(self.base._rng)
        kwargs = dict(self.base._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            kwargs["noise"] = noise[None] if noise.ndim == 2 else noise
        started = time.perf_counter()
        actions = self.sampler(rng, Observation.from_dict(inputs),
            previous_actions=jnp.asarray(normalized_previous)[None],
            inference_delay=jnp.asarray(delay, dtype=jnp.int32),
            prefix_horizon=jnp.asarray(horizon, dtype=jnp.int32), **kwargs)
        actions = np.asarray(actions[0])  # Synchronizes GPU before timing.
        outputs = self.base._output_transform({"state": np.asarray(inputs["state"][0]),
                                               "actions": actions})
        outputs["policy_timing"] = {"infer_ms": (time.perf_counter() - started) * 1000}
        outputs[RTC_KEY] = response_context
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle / "code"))
    from b601_policy import load_policy  # Configures XLA before JAX imports.
    policy = RTCPolicy(load_policy(args.bundle / "checkpoint", denoise_steps=10))
    # Compile both branches while motors are still disconnected. Service
    # readiness means RTC is actually GPU-tested, not merely advertised.
    example = np.load(args.bundle / "example_observation.npz")
    obs = {"observation.images.overhead": example["overhead"],
           "observation.images.wrist": example["wrist"],
           "observation.state": example["state"],
           "prompt": "拿起蓝色餐巾纸，把它放进黑色框里"}
    baseline = policy.infer(obs, noise=example["noise"])["actions"]
    np.testing.assert_allclose(baseline, example["expected_actions"], atol=1e-4, rtol=1e-5)
    previous = np.asarray(baseline).copy()
    previous[:, 0] += 10  # Offline challenge only, never sent to any robot.
    rtc = {"protocol": RTC_PROTOCOL, "request_id": "startup-test", "observation_tick": 0,
           "previous_actions": previous, "inference_delay": 7, "prefix_horizon": 18}
    begun = time.perf_counter()
    guided = policy.infer({**obs, RTC_KEY: rtc}, noise=example["noise"])["actions"]
    if guided.shape != (25, 7) or not np.isfinite(guided).all():
        raise RuntimeError("RTC GPU smoke test failed")
    plain_error = float(np.linalg.norm((baseline - previous)[:7]))
    guided_error = float(np.linalg.norm((guided - previous)[:7]))
    if guided_error >= plain_error:
        raise RuntimeError(f"RTC guidance moved away from prefix: {plain_error} -> {guided_error}")
    rtc["prefix_horizon"] = 0
    unguided = policy.infer({**obs, RTC_KEY: rtc}, noise=example["noise"])["actions"]
    np.testing.assert_allclose(unguided, baseline, atol=1e-4, rtol=1e-5)
    # Explicit noise (reproducibility tests) and omitted noise (normal RPC)
    # form distinct JIT call signatures. Warm the actual production path too.
    policy.infer(obs)
    rtc["prefix_horizon"] = 18
    production_times = []
    for _ in range(2):
        before = time.perf_counter()
        production = policy.infer({**obs, RTC_KEY: rtc})["actions"]
        if production.shape != (25, 7) or not np.isfinite(production).all():
            raise RuntimeError("RTC production-path warm-up failed")
        production_times.append((time.perf_counter() - before) * 1000)
    import json
    print(json.dumps({"rtc_startup_test": "passed", "golden_matches": True,
                      "prefix_error_before": plain_error, "prefix_error_after": guided_error,
                      "production_path_warmed": True, "production_call_ms": production_times,
                      "compile_and_test_seconds": time.perf_counter() - begun}), flush=True)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
