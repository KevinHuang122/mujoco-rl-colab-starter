"""
Minimal Franka Emika Panda import demo for MuJoCo Playground.

Run:
  python franka_import_demo.py
"""

from __future__ import annotations

from mujoco_playground import registry


def main() -> None:
    # List manipulation environments that contain "Panda".
    panda_envs = [name for name in registry.manipulation.ALL_ENVS if "Panda" in name]
    print("Available Panda environments:")
    for name in panda_envs:
        print(f"  - {name}")

    # Load one Franka Panda manipulation task.
    env_name = "PandaPickCubeOrientation"
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    print("\nLoaded Franka environment successfully.")
    print("env_name:", env_name)
    print("env_type:", type(env))
    print("episode_length:", getattr(env_cfg, "episode_length", "unknown"))
    print("dt:", getattr(env, "dt", "unknown"))


if __name__ == "__main__":
    main()
