"""
modal_inspect_hiro_env.py
-------------------------
Modal wrapper that runs hiro/inspect_env.py on a GPU container.

Run:
    uv run modal run modal_inspect_hiro_env.py

Runs both tasks (PickCube-v1, StackCube-v1) and both control modes
(pd_joint_delta_pos, pd_ee_delta_pos) and prints the full report.
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("hiro-inspect-env")
PROJECT_ROOT = Path(__file__).parent
REMOTE_ROOT = "/root/project"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "libvulkan1",
        "vulkan-tools",
        "libegl1",
        "libgles2",
        "libglvnd0",
        "libxext6",
        "libxrender1",
        "libsm6",
        "libxi6",
        "libgl1",
        "libglib2.0-0",
        "git",
        "clang",
        "build-essential",
    )
    .pip_install("modal>=1.4.3", "torch", "mani_skill", "numpy")
    .env({"MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            ".pytest_cache",
            ".venv",
            "__pycache__",
            "checkpoints",
            "*.mp4",
        ],
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_inspect(env_id: str, control_mode: str) -> str:
    """
    Runs inside the GPU container. Returns the full printed report as a string
    so the local entrypoint can display it.
    """
    import io
    import sys

    import mani_skill.envs  # noqa: F401 — registers tasks with gymnasium

    from hiro.inspect_env import inspect

    # Capture stdout so we can return it as a string to the local caller.
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        # physx_cpu gives structured dict obs with named fields.
        # physx_gpu pre-flattens everything, losing field names.
        inspect(env_id, control_mode, n_random_steps=50, sim_backend="physx_cpu")
    finally:
        sys.stdout = old_stdout

    return buf.getvalue()


@app.function(image=image, gpu="T4", timeout=600)
def read_task_source() -> str:
    """
    Reads the ManiSkill3 source files for PickCube and StackCube so we can
    see exactly how the 42/48-dim state vectors are constructed.
    """
    import glob
    import os

    parts = []
    root = "/usr/local/lib/python3.11/site-packages/mani_skill"
    patterns = ["*pick_cube*", "*stack_cube*", "*pickcube*", "*stackcube*"]
    found: set[str] = set()
    for pat in patterns:
        for path in glob.glob(os.path.join(root, "**", pat), recursive=True):
            if path.endswith(".py"):
                found.add(path)

    for path in sorted(found):
        with open(path) as f:
            content = f.read()
        parts.append(f"\n{'='*60}\nFILE: {path}\n{'='*60}\n{content}")

    return "\n".join(parts) if parts else "No task source files found."


@app.local_entrypoint()
def main(
    env_id: str = "",
    control_mode: str = "",
    source: bool = False,
):
    if source:
        print(read_task_source.remote())
        return

    tasks = [env_id] if env_id else ["PickCube-v1", "StackCube-v1"]
    modes = [control_mode] if control_mode else ["pd_joint_delta_pos", "pd_ee_delta_pos"]

    # Fan out all (task, mode) pairs in parallel — Modal runs them concurrently.
    calls = {
        (t, m): run_inspect.spawn(t, m)
        for t in tasks
        for m in modes
    }

    for (t, m), fc in calls.items():
        result = fc.get()
        print(result)
