"""
Franka state/action setup demo for MuJoCo Playground.

What this script does:
1) Loads PandaPickCubeOrientation.
2) Defines action as continuous vector in [-1, 1]^action_size.
3) Defines state as the observation vector returned by env state (state.obs).
4) Prints state/action values at current timestep.

Run:
  python franka_state_action_demo.py
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry


def setup_backend_for_colab() -> None:
    # Keep this demo stable on Colab G4 runtimes.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")


def to_numpy(x):
    return np.asarray(x)


def read_state_vector(env_state) -> np.ndarray:
    """
    Default RL state:
    - Use env_state.obs as the state vector for policy input.
    """
    if not hasattr(env_state, "obs"):
        raise RuntimeError("Current env state does not contain `obs`.")
    return to_numpy(env_state.obs)


def print_state_action(tag: str, state_vec: np.ndarray, action_vec: np.ndarray | None) -> None:
    print(f"\n[{tag}]")
    print("state_dim:", state_vec.shape[0])
    print("state(first 10):", np.round(state_vec[:10], 4))
    if action_vec is None:
        print("action: <none at reset>")
    else:
        print("action_dim:", action_vec.shape[0])
        print("action:", np.round(action_vec, 4))


def main() -> None:
    setup_backend_for_colab()

    env_name = "PandaPickCubeOrientation"
    env = registry.load(env_name)
    print("Loaded env:", env_name)
    print("action_size:", env.action_size)

    # t=0: reset and read current state.
    key = jax.random.PRNGKey(0)
    env_state = env.reset(key)
    state_t0 = read_state_vector(env_state)
    print_state_action("t=0 (after reset)", state_t0, action_vec=None)

    # Define one action and read its value.
    key, subkey = jax.random.split(key)
    action_t0 = jax.random.uniform(
        subkey,
        shape=(env.action_size,),
        minval=-1.0,
        maxval=1.0,
    )
    action_t0_np = to_numpy(action_t0)
    print_state_action("action at t=0", state_t0, action_vec=action_t0_np)

    # Step once and read next state.
    env_state_next = env.step(env_state, action_t0)
    state_t1 = read_state_vector(env_state_next)
    print_state_action("t=1 (after one step)", state_t1, action_vec=action_t0_np)

    # Optional extras often used in training logs.
    reward = float(to_numpy(env_state_next.reward))
    done = bool(to_numpy(env_state_next.done))
    print("\nreward(t=1):", round(reward, 6))
    print("done(t=1):", done)


if __name__ == "__main__":
    main()
