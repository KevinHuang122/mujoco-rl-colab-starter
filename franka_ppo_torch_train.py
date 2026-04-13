"""
Franka PPO training with JAX/Brax (active version).

Notes:
- The previous PyTorch version is preserved (commented out) in
  `franka_ppo_torch_train_legacy_commented.py`.
- This script uses MuJoCo Playground + Brax PPO for faster JAX-native training.
"""

from __future__ import annotations

import argparse
import functools
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import jax
import matplotlib.pyplot as plt
import mediapy as media
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params


def setup_runtime(seed: int = 1) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MJX_GPU_DEFAULT_WARP", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print("[INFO] jax.default_backend()=", jax.default_backend())
    print("[INFO] seed=", seed)


def build_env_with_fixed_target(env_name: str = "PandaPickCubeOrientation"):
    env_cfg = registry.get_default_config(env_name)
    fixed_target = [0.55, 0.00, 0.12]

    config_overrides: Dict[str, Any] = {"impl": "jax"}
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

    for key in ["randomize_goal", "randomize_target", "target_randomization"]:
        if hasattr(env_cfg, key):
            config_overrides[key] = False

    try:
        env = registry.load(env_name, config_overrides=config_overrides)
        print("[ENV] Loaded with overrides:", config_overrides)
        return env, env_cfg
    except TypeError:
        env = registry.load(env_name)
        print("[ENV] Loaded default config (override API not available).")
        return env, env_cfg


def train_and_eval(
    save_path: str = "franka_ppo_jax_params.pkl",
    video_path: str = "franka_ppo_jax_eval.mp4",
    reward_plot_path: str = "franka_ppo_jax_reward_curve.png",
    num_timesteps: int = 80_000,
    seed: int = 1,
) -> None:
    setup_runtime(seed=seed)
    env_name = "PandaPickCubeOrientation"
    env, env_cfg = build_env_with_fixed_target(env_name)

    ppo_params = dict(manipulation_params.brax_ppo_config(env_name))
    ppo_params["num_timesteps"] = int(num_timesteps)
    ppo_params["num_evals"] = int(8)

    # reward curve containers
    x_data: List[int] = []
    y_data: List[float] = []
    y_err: List[float] = []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(int(num_steps))
        y_data.append(float(metrics.get("eval/episode_reward", 0.0)))
        y_err.append(float(metrics.get("eval/episode_reward_std", 0.0)))
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
        seed=seed,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    print("[TRAIN] final_metrics:", metrics)
    if len(times) > 1:
        print("[TRAIN] jit_time:", times[1] - times[0])
        print("[TRAIN] train_time:", times[-1] - times[1])

    # Save reward curve.
    plot_dir = os.path.dirname(reward_plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.errorbar(x_data, y_data, yerr=y_err, color="tab:blue")
    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Episode Reward")
    plt.title("JAX PPO Reward Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reward_plot_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved reward curve: {reward_plot_path}")

    # Evaluate and render video.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_infer = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(42)
    rollout = []
    state = jit_reset(rng)
    rollout.append(state)
    for _ in range(int(env_cfg.episode_length)):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_infer(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        if bool(state.done):
            break

    frames = env.render(rollout)
    video_dir = os.path.dirname(video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    media.write_video(video_path, frames, fps=1.0 / float(env.dt))
    print(f"[EVAL] Saved video: {video_path}")

    # Save JAX params checkpoint.
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"[SAVE] Saved JAX params: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="franka_ppo_jax_params.pkl")
    parser.add_argument("--video_path", type=str, default="franka_ppo_jax_eval.mp4")
    parser.add_argument("--reward_plot_path", type=str, default="franka_ppo_jax_reward_curve.png")
    parser.add_argument("--num_timesteps", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    train_and_eval(
        save_path=args.save_path,
        video_path=args.video_path,
        reward_plot_path=args.reward_plot_path,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
