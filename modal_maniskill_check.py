import modal

app = modal.App("maniskill-check")

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
    .pip_install("torch", "mani_skill", "pillow")
)


@app.function(image=image, gpu="T4", timeout=600)
def smoke_test(env_id: str = "StackCube-v1") -> bytes:
    import io
    import numpy as np
    import gymnasium as gym
    import mani_skill  # noqa: F401  -- import registers ManiSkill envs with gym
    from PIL import Image

    print(f"mani_skill version: {mani_skill.__version__}")
    print(f"env: {env_id}")

    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

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
def main():
    for env_id in ["StackCube-v1", "PickCube-v1"]:
        png_bytes = smoke_test.remote(env_id)
        out_path = f"{env_id.lower().replace('-', '_')}_smoke.png"
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"saved {len(png_bytes)} bytes to {out_path}")
