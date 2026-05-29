"""
modal_verify_subgoal_space.py
------------------------------
Runs hiro/verify_subgoal_space.py on a GPU container.

Usage:
    uv run modal run modal_verify_subgoal_space.py
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("hiro-verify-subgoal-space")
PROJECT_ROOT = Path(__file__).parent
REMOTE_ROOT  = "/root/project"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "libvulkan1", "vulkan-tools", "libegl1", "libgles2", "libglvnd0",
        "libxext6", "libxrender1", "libsm6", "libxi6", "libgl1",
        "libglib2.0-0", "git", "clang", "build-essential",
    )
    .pip_install("modal>=1.4.3", "torch", "mani_skill", "numpy", "pytest")
    .env({"MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[".git", ".pytest_cache", ".venv", "__pycache__", "checkpoints", "*.mp4"],
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_env_verification() -> str:
    import io, sys
    import mani_skill.envs  # noqa: F401

    from hiro.verify_subgoal_space import verify_pick_cube, verify_stack_cube

    buf = io.StringIO()
    sys.stdout = buf
    try:
        verify_pick_cube(sim_backend="physx_cpu")
        verify_stack_cube(sim_backend="physx_cpu")
    finally:
        sys.stdout = sys.__stdout__
    return buf.getvalue()


@app.function(image=image, gpu="T4", timeout=600)
def run_unit_tests() -> str:
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "hiro/tests/test_subgoal_space.py", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=REMOTE_ROOT,
    )
    return result.stdout + (result.stderr if result.returncode != 0 else "")


@app.local_entrypoint()
def main() -> None:
    print("\n--- Unit tests (no env) ---")
    print(run_unit_tests.remote())

    print("\n--- Env verification (ManiSkill) ---")
    print(run_env_verification.remote())
