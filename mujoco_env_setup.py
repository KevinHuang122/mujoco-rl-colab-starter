"""
MuJoCo + MuJoCo Playground environment setup helper.

Usage (recommended in Colab):
  python mujoco_env_setup.py
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_pip_packages() -> None:
    # Minimal stack for MuJoCo manipulation workflow.
    packages = [
        "mujoco",
        "mujoco_mjx",
        "brax",
        "playground",
        "mediapy",
        "torch",
        "warp-lang",
    ]
    run([sys.executable, "-m", "pip", "install", "-U", *packages])


def check_gpu() -> list[str]:
    print("\n[CHECK] GPU availability")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    gpu_names: list[str] = []
    if result.returncode == 0:
        print("GPU detected by nvidia-smi.")
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if query.returncode == 0:
            gpu_names = [line.strip() for line in query.stdout.splitlines() if line.strip()]
            if gpu_names:
                print("GPU model(s):", ", ".join(gpu_names))
    else:
        print("No GPU detected by nvidia-smi (this is okay for CPU-only tests).")
    return gpu_names


def configure_env_vars(gpu_names: list[str]) -> None:
    # Colab usually needs this for MuJoCo EGL rendering.
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Colab G4 runtimes can hit CUDA/PTXAS incompatibilities with JAX GPU.
    # We default to CPU backend on G4 to keep setup stable for import/visualization.
    if any("G4" in name.upper() for name in gpu_names):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        print("[FIX] G4 runtime detected, defaulting JAX backend to CPU for compatibility.")

    print("\n[CHECK] Environment variables")
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    # Keep XLA_FLAGS untouched by default to avoid Colab/CUDA/PTXAS mismatch.
    print("XLA_FLAGS =", os.environ.get("XLA_FLAGS", "<not set>"))
    print("JAX_PLATFORMS =", os.environ.get("JAX_PLATFORMS", "<auto>"))


def validate_imports() -> None:
    print("\n[CHECK] Import validation")
    required_modules = [
        "mujoco",
        "mujoco.mjx",
        "brax",
        "jax",
        "mujoco_playground",
    ]
    for name in required_modules:
        importlib.import_module(name)
        print(f"OK: import {name}")


def validate_playground_registry() -> None:
    print("\n[CHECK] MuJoCo Playground registry")
    from mujoco_playground import registry

    env_name = "PandaPickCubeOrientation"
    env = registry.load(env_name)
    cfg = registry.get_default_config(env_name)
    print(f"Loaded env: {env_name}")
    print("Env type:", type(env))
    print("Episode length:", getattr(cfg, "episode_length", "unknown"))


def ensure_egl_icd_file_if_colab() -> None:
    # Mirrors the official notebook's workaround for some Colab runtimes.
    icd_path = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if not Path("/content").exists():
        return
    if icd_path.exists():
        return

    print("\n[FIX] Adding Nvidia EGL ICD file for Colab runtime")
    icd_path.parent.mkdir(parents=True, exist_ok=True)
    icd_path.write_text(
        '{\n'
        '  "file_format_version" : "1.0.0",\n'
        '  "ICD" : {\n'
        '    "library_path" : "libEGL_nvidia.so.0"\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=== MuJoCo Environment Setup ===")
    ensure_pip_packages()
    gpu_names = check_gpu()
    configure_env_vars(gpu_names)
    ensure_egl_icd_file_if_colab()
    validate_imports()
    validate_playground_registry()
    print("\nSetup completed. You can move on to training code.")


if __name__ == "__main__":
    main()
