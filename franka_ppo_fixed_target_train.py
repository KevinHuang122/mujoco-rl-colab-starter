"""
Franka PPO training demo (fixed target first).

This script is designed for Colab + MuJoCo Playground.
It includes:
1) State/action inspection
2) PPO training
3) Evaluation rollout video export
"""

from __future__ import annotations

import functools
import os
from datetime import datetime
from typing import Any, Dict

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mediapy as media
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params


# =========================
# Module 1: Runtime backend setup
# Purpose: keep training stable on Colab G4 by defaulting to CPU backend.
# =========================
def setup_runtime() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")


# =========================
# Module 2: Environment construction
# Purpose: load Franka task, and try to enforce a fixed goal/target if config fields exist.
# =========================
def build_env_with_fixed_target(env_name: str = "PandaPickCubeOrientation"):
    env_cfg = registry.get_default_config(env_name)
    fixed_target = np.array([0.55, 0.00, 0.12], dtype=np.float32)

    # Try common goal field names used by different env versions.
    config_overrides: Dict[str, Any] = {}
    candidate_goal_fields = [
        "goal_position",
        "goal_pos",
        "target_position",
        "target_pos",
        "goal_xyz",
        "object_goal_pos",
    ]
    for key in candidate_goal_fields:
        if hasattr(env_cfg, key):
            config_overrides[key] = fixed_target
            break

    # Try common randomization switches to keep target fixed.
    for key in ["randomize_goal", "randomize_target", "target_randomization"]:
        if hasattr(env_cfg, key):
            config_overrides[key] = False

    if config_overrides:
        try:
            env = registry.load(env_name, config_overrides=config_overrides)
            print("[ENV] Loaded with fixed-target overrides:", config_overrides)
            return env, env_cfg, fixed_target
        except TypeError:
            # Older/newer APIs may not support config_overrides directly.
            pass

    env = registry.load(env_name)
    print("[ENV] Loaded default config (fixed-target override not available in this version).")
    return env, env_cfg, fixed_target


# =========================
# Module 3: State/action extraction
# Purpose: show exactly how to read state(obs) and action values at current timestep.
# =========================
def inspect_state_action(env) -> None:
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    obs = np.asarray(state.obs)
    print("[STATE/ACTION] obs_dim:", obs.shape[0])
    print("[STATE/ACTION] state(t=0) first 10:", np.round(obs[:10], 4))

    key, subkey = jax.random.split(key)
    action = jax.random.uniform(
        subkey, shape=(env.action_size,), minval=-1.0, maxval=1.0
    )
    action_np = np.asarray(action)
    print("[STATE/ACTION] action_dim:", action_np.shape[0])
    print("[STATE/ACTION] action(t=0):", np.round(action_np, 4))

    next_state = env.step(state, action)
    next_obs = np.asarray(next_state.obs)
    print("[STATE/ACTION] state(t=1) first 10:", np.round(next_obs[:10], 4))
    print("[STATE/ACTION] reward(t=1):", float(np.asarray(next_state.reward)))
    print("[STATE/ACTION] done(t=1):", bool(np.asarray(next_state.done)))


# =========================
# Module 4: PPO hyper-parameter preparation
# Purpose: start from official playground PPO config, then set practical demo values.
# =========================
def build_ppo_params(env_name: str) -> Dict[str, Any]:
    ppo_params = dict(manipulation_params.brax_ppo_config(env_name))

    # You can increase these later for better performance.
    ppo_params["num_timesteps"] = int(200_000)
    ppo_params["num_evals"] = int(10)
    ppo_params["reward_scaling"] = float(1.0)

    return ppo_params


# =========================
# Module 5: PPO training loop assembly
# Purpose: create train_fn and run optimization on actor/critic.
# =========================
def train_ppo(env, env_name: str):
    ppo_params = build_ppo_params(env_name)

    x_data, y_data = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        # Keep textual logs for script execution; no notebook plotting required.
        times.append(datetime.now())
        x_data.append(int(num_steps))
        y_data.append(float(metrics.get("eval/episode_reward", 0.0)))
        print(
            f"[TRAIN] steps={num_steps} "
            f"eval_reward={metrics.get('eval/episode_reward', 0.0):.3f} "
            f"eval_std={metrics.get('eval/episode_reward_std', 0.0):.3f}"
        )

    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params["network_factory"],
        )

    train_fn = functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress,
        seed=1,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print("[TRAIN] jit_time:", times[1] - times[0] if len(times) > 1 else "n/a")
    print("[TRAIN] final_metrics:", metrics)
    return make_inference_fn, params, metrics


# =========================
# Module 6: Evaluation rollout + visualization export
# Purpose: run trained policy and save a video file for visual inspection.
# =========================
def evaluate_and_save_video(env, env_cfg, make_inference_fn, params) -> None:
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(42)
    rollout = []

    state = jit_reset(rng)
    rollout.append(state)
    for _ in range(int(env_cfg.episode_length)):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    render_every = 1
    frames = env.render(rollout[::render_every])
    fps = 1.0 / float(env.dt) / render_every
    media.write_video("franka_ppo_eval.mp4", frames, fps=fps)
    print("[EVAL] Saved video: franka_ppo_eval.mp4")


def main() -> None:
    setup_runtime()

    env_name = "PandaPickCubeOrientation"
    env, env_cfg, fixed_target = build_env_with_fixed_target(env_name)
    print("[ENV] desired fixed target:", fixed_target.tolist())

    inspect_state_action(env)
    make_inference_fn, params, _ = train_ppo(env, env_name)
    evaluate_and_save_video(env, env_cfg, make_inference_fn, params)
    print("[DONE] PPO training script completed.")


if __name__ == "__main__":
    main()
