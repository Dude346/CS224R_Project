from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("maniskill-check")
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
    .pip_install("torch", "gymnasium", "mani_skill", "pillow")
    .env(
        {
            "PYTHONPATH": REMOTE_ROOT,
            "MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1",
        }
    )
    .workdir(REMOTE_ROOT)
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            ".pytest_cache",
            ".venv",
            "__pycache__",
        ],
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def smoke_test(
    env_id: str = "StackCube-v1",
    robot_uids: str = "weakpanda",
    obs_mode: str = "state+rgb",
    seed: int = 0,
    num_steps: int = 20,
) -> bytes:
    import io
    import numpy as np
    import gymnasium as gym
    import mani_skill  # noqa: F401  -- import registers ManiSkill envs with gym
    from PIL import Image
    import cs224r_project.mani_skill_ext  # noqa: F401 - registers weak Panda robots

    print(f"mani_skill version: {mani_skill.__version__}")
    print(f"task: {env_id}, robot_uids: {robot_uids}, obs_mode: {obs_mode}")

    env = gym.make(
        env_id,
        robot_uids=robot_uids,
        obs_mode=obs_mode,
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=seed)
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    print(f"frame shape: {frame.shape}, dtype: {frame.dtype}")

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    env.close()
    return buf.getvalue()


@app.local_entrypoint()
def main(
    env_id: str = "StackCube-v1",
    robot_uids: str = "weakpanda",
    obs_mode: str = "state+rgb",
    seed: int = 0,
    num_steps: int = 20,
):
    png_bytes = smoke_test.remote(
        env_id=env_id,
        robot_uids=robot_uids,
        obs_mode=obs_mode,
        seed=seed,
        num_steps=num_steps,
    )
    out_path = f"{env_id}_{robot_uids}_smoke.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"saved {len(png_bytes)} bytes to {out_path}")
