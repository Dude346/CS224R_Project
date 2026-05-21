import modal

app = modal.App("train-sac")

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
    .pip_install("torch", "mani_skill", "pillow", "wandb", "tensorboard")
    .run_commands(
        "git clone --depth 1 https://github.com/haosulab/ManiSkill.git /root/ManiSkill",
    )
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d",
        """echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json""",
        """echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json""",
    )
)

volume = modal.Volume.from_name("cs224r-project-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=5 * 60 * 60,
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_sac(
    env_id: str = "PickCube-v1",
    seed: int = 1,
    total_timesteps: int = 500_000,
    num_envs: int = 32,
    buffer_size: int = 500_000,
    eval_freq: int = 50_000,
    control_mode: str = "pd_joint_delta_pos",
    utd: float = 0.5,
    exp_name: str | None = None,
    wandb_project: str = "cs224r-project",
) -> tuple[str, bytes]:
    import glob
    import os
    import subprocess
    import threading

    if exp_name is None:
        exp_name = f"sac_{env_id.replace('-', '_')}_panda_seed{seed}_{total_timesteps}steps"

    cmd = [
        "python",
        "/root/ManiSkill/examples/baselines/sac/sac.py",
        f"--env_id={env_id}",
        f"--seed={seed}",
        f"--total_timesteps={total_timesteps}",
        f"--num_envs={num_envs}",
        f"--buffer_size={buffer_size}",
        f"--eval_freq={eval_freq}",
        f"--control-mode={control_mode}",
        f"--utd={utd}",
        f"--exp_name={exp_name}",
        f"--wandb_project_name={wandb_project}",
        "--track",
    ]
    print("Running:", " ".join(cmd))

    # Background thread: periodically commit the volume so an unexpected
    # crash mid-training doesn't lose every checkpoint written so far.
    stop_event = threading.Event()

    def _commit_periodically(interval_seconds: int = 600):
        while not stop_event.wait(interval_seconds):
            try:
                volume.commit()
                print("[volume.commit] periodic commit done")
            except Exception as exc:
                print(f"[volume.commit] periodic commit failed: {exc}")

    commit_thread = threading.Thread(target=_commit_periodically, daemon=True)
    commit_thread.start()

    try:
        subprocess.run(cmd, cwd="/output", check=True)
    finally:
        stop_event.set()
        commit_thread.join(timeout=10)
        volume.commit()  # final commit — runs even if subprocess errored

    run_dir = f"/output/runs/{exp_name}"
    video_files = sorted(
        glob.glob(f"{run_dir}/**/*.mp4", recursive=True),
        key=os.path.getmtime,
    )
    video_bytes = b""
    if video_files:
        latest = video_files[-1]
        print(f"Returning final video: {latest}")
        with open(latest, "rb") as f:
            video_bytes = f.read()
    else:
        print(f"WARNING: no .mp4 found under {run_dir}")
    return exp_name, video_bytes


@app.local_entrypoint()
def main(
    env_id: str = "PickCube-v1",
    seed: int = 1,
    total_timesteps: int = 500_000,
    eval_freq: int = 50_000,
):
    run_name, video_bytes = train_sac.remote(
        env_id=env_id,
        seed=seed,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
    )
    print(f"Training complete. Run name: {run_name}")
    if video_bytes:
        out_path = f"{run_name}.mp4"
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"saved {len(video_bytes)} bytes to {out_path}")
    else:
        print("No video produced.")


@app.local_entrypoint()
def launch(
    env_id: str = "PickCube-v1",
    seed: int = 1,
    total_timesteps: int = 500_000,
    eval_freq: int = 50_000,
):
    fc = train_sac.spawn(
        env_id=env_id,
        seed=seed,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
    )
    exp_name = f"sac_{env_id.replace('-', '_')}_panda_seed{seed}_{total_timesteps}steps"
    print(f"Spawned function call: {fc.object_id}")
    print(f"Run name: {exp_name}")
    print(f"Monitor logs: uv run modal app logs train-sac")
    print(f"After completion, fetch video with:")
    print(f"  uv run modal volume get cs224r-project-results runs/{exp_name}/videos ./videos")
