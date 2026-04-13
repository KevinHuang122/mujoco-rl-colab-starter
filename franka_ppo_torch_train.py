"""
Franka PPO training with PyTorch (no Brax PPO trainer).

This script keeps the MuJoCo Playground environment, but the RL algorithm
is implemented with PyTorch actor-critic + PPO update.

Run:
  python franka_ppo_torch_train.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Force GPU-only execution for both JAX and PyTorch.
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm.auto import tqdm

from mujoco_playground import registry


# =========================
# Module 1: Runtime setup
# Purpose: make Colab runtime stable (especially G4) and set seeds.
# =========================
def setup_runtime(seed: int = 1) -> None:
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Module 2: Environment setup
# Purpose: load Franka task and optionally fix target config if fields exist.
# =========================
def build_env_with_fixed_target(env_name: str = "PandaPickCubeOrientation"):
    env_cfg = registry.get_default_config(env_name)
    fixed_target = np.array([0.55, 0.00, 0.12], dtype=np.float32)

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

    for key in ["randomize_goal", "randomize_target", "target_randomization"]:
        if hasattr(env_cfg, key):
            config_overrides[key] = False

    # Prefer MJX JAX implementation over warp backend.
    config_overrides["impl"] = "jax"

    if config_overrides:
        try:
            env = registry.load(env_name, config_overrides=config_overrides)
            print("[ENV] Loaded with fixed-target overrides:", config_overrides)
            return env, env_cfg
        except TypeError:
            pass

    # Fallback: still try forcing impl=jax even if other keys are unsupported.
    try:
        env = registry.load(env_name, config_overrides={"impl": "jax"})
        print("[ENV] Loaded with impl=jax fallback.")
        return env, env_cfg
    except TypeError:
        env = registry.load(env_name)
        print("[ENV] Loaded default config (impl override unavailable in this version).")
    return env, env_cfg


# =========================
# Module 3: Policy/Value networks (PyTorch)
# Purpose: define Actor-Critic that outputs action distribution and state value.
# =========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist(self, obs: torch.Tensor) -> Normal:
        mu = self.actor(obs)
        log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mu)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        action = d.sample()
        log_prob = d.log_prob(action).sum(-1)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        log_prob = d.log_prob(action).sum(-1)
        entropy = d.entropy().sum(-1)
        value = self.value(obs)
        return log_prob, entropy, value


# =========================
# Module 4: Rollout storage + GAE
# Purpose: collect (s, a, r, done, logp_old, v) and compute advantage/targets.
# =========================
@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


def collect_rollout(
    env,
    model: ActorCritic,
    horizon: int,
    gamma: float,
    lam: float,
    device: torch.device,
    rng: jax.Array,
):
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    logp_list: List[float] = []
    val_list: List[float] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    rollout_states = []

    env_state = env.reset(rng)
    rollout_states.append(env_state)

    for _ in range(horizon):
        obs_np = np.asarray(env_state.obs, dtype=np.float32).copy()
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_t, logp_t, value_t = model.act(obs_t)

        action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        # Environment expects bounded control; we clip to [-1, 1].
        action_np = np.clip(action_np, -1.0, 1.0)

        next_state = env.step(env_state, jnp.asarray(action_np))
        reward = float(np.asarray(next_state.reward))
        done = float(np.asarray(next_state.done))

        obs_list.append(obs_np)
        act_list.append(action_np)
        logp_list.append(float(logp_t.item()))
        val_list.append(float(value_t.item()))
        rew_list.append(reward)
        done_list.append(done)

        env_state = next_state
        rollout_states.append(env_state)

        if done > 0.5:
            rng, sub = jax.random.split(rng)
            env_state = env.reset(sub)
            rollout_states.append(env_state)

    last_obs = np.asarray(env_state.obs, dtype=np.float32).copy()
    last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_value = float(model.value(last_obs_t).item())

    rewards = np.asarray(rew_list, dtype=np.float32)
    dones = np.asarray(done_list, dtype=np.float32)
    values = np.asarray(val_list, dtype=np.float32)
    advantages, returns = compute_gae(rewards, dones, values, last_value, gamma, lam)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = RolloutBatch(
        obs=torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.asarray(act_list), dtype=torch.float32, device=device),
        old_log_probs=torch.as_tensor(np.asarray(logp_list), dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
    )
    return batch, rollout_states, rng


# =========================
# Module 5: PPO update
# Purpose: train actor/critic using clipped objective + value loss + entropy bonus.
# =========================
def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    mini_batch_size: int,
) -> Dict[str, float]:
    n = batch.obs.shape[0]
    idx = np.arange(n)

    loss_pi_last = 0.0
    loss_v_last = 0.0
    entropy_last = 0.0

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mini_batch_size):
            mb_idx = idx[start : start + mini_batch_size]

            obs = batch.obs[mb_idx]
            actions = batch.actions[mb_idx]
            old_logp = batch.old_log_probs[mb_idx]
            returns = batch.returns[mb_idx]
            adv = batch.advantages[mb_idx]

            logp, entropy, value = model.evaluate_actions(obs, actions)
            ratio = torch.exp(logp - old_logp)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
            loss_pi = -torch.min(surr1, surr2).mean()

            loss_v = 0.5 * ((returns - value) ** 2).mean()
            loss_ent = entropy.mean()

            loss = loss_pi + value_coef * loss_v - entropy_coef * loss_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_pi_last = float(loss_pi.item())
            loss_v_last = float(loss_v.item())
            entropy_last = float(loss_ent.item())

    return {
        "loss_pi": loss_pi_last,
        "loss_v": loss_v_last,
        "entropy": entropy_last,
    }


# =========================
# Module 6: End-to-end train + evaluate
# Purpose: orchestrate PPO training and export a video for visualization.
# =========================
def save_checkpoint(
    save_path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs_dim: int,
    act_dim: int,
) -> None:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        },
        save_path,
    )
    print(f"[SAVE] Checkpoint saved: {save_path}")


def train_and_eval(
    save_path: str = "franka_ppo_torch_ckpt.pt",
    video_path: str = "franka_ppo_torch_eval.mp4",
    updates: int = 8,
) -> None:
    setup_runtime(seed=1)
    env_name = "PandaPickCubeOrientation"
    env, env_cfg = build_env_with_fixed_target(env_name)

    rng = jax.random.PRNGKey(42)
    init_state = env.reset(rng)
    obs_dim = int(np.asarray(init_state.obs).shape[0])
    act_dim = int(env.action_size)
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}")
    print("[INFO] mode=GPU_ONLY")
    print(
        "[INFO] JAX_PLATFORMS=",
        os.environ.get("JAX_PLATFORMS"),
        "jax.default_backend()=",
        jax.default_backend(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if jax.default_backend() != "cuda":
        raise RuntimeError(
            f"JAX backend is '{jax.default_backend()}', expected 'cuda'. "
            "Please switch Colab runtime to GPU and restart runtime."
        )
    if device.type != "cuda":
        raise RuntimeError(
            "PyTorch CUDA is unavailable. Please switch Colab runtime to GPU and restart runtime."
        )
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training hyperparameters (fast demo defaults).
    horizon = 512
    total_steps = int(updates * horizon)
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.0
    epochs = 5
    mini_batch_size = 128

    print(f"[TRAIN] total_steps={total_steps}, updates={updates}")
    pbar = tqdm(range(1, updates + 1), desc="PPO Updates", unit="update")
    for u in pbar:
        batch, rollout_states, rng = collect_rollout(
            env=env,
            model=model,
            horizon=horizon,
            gamma=gamma,
            lam=lam,
            device=device,
            rng=rng,
        )
        stats = ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
        )
        mean_ret = float(batch.returns.mean().item())
        pbar.set_postfix(
            mean_return=f"{mean_ret:.3f}",
            loss_pi=f"{stats['loss_pi']:.4f}",
            loss_v=f"{stats['loss_v']:.4f}",
        )

    # Simple evaluation rollout for visualization.
    eval_states = []
    state = env.reset(rng)
    eval_states.append(state)
    for _ in range(int(env_cfg.episode_length)):
        obs_np = np.asarray(state.obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = model.dist(obs_t)
            action = dist.mean.squeeze(0).cpu().numpy().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        state = env.step(state, jnp.asarray(action))
        eval_states.append(state)
        if float(np.asarray(state.done)) > 0.5:
            break

    frames = env.render(eval_states)
    fps = 1.0 / float(env.dt)
    video_dir = os.path.dirname(video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    media.write_video(video_path, frames, fps=fps)
    print(f"[EVAL] Saved video: {video_path}")

    save_checkpoint(
        save_path=save_path,
        model=model,
        optimizer=optimizer,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="franka_ppo_torch_ckpt.pt",
        help="Where to save the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="franka_ppo_torch_eval.mp4",
        help="Where to save the evaluation video.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=8,
        help="Number of PPO update iterations.",
    )
    args = parser.parse_args()
    train_and_eval(save_path=args.save_path, video_path=args.video_path, updates=args.updates)
