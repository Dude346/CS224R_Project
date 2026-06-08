"""
modal_train_hiro.py
-------------------
Modal entry point for HIRO training.  Mirrors modal_train_sac.py but calls our
own trainer (hiro/trainer.py) directly instead of shelling out to upstream
sac.py — we own the training loop.

Reuses the SAC image/volume (CUDA + Vulkan + ManiSkill + torch/wandb/tensorboard,
with the whole project — including hiro/ — mounted at REMOTE_ROOT).

Smoke run (short, just confirms it runs end-to-end — NOT a learning check):
    uv run modal run modal_train_hiro.py \
        --total-timesteps 10000 --eval-freq 5000 --learning-starts 2000 \
        --low-buffer-size 100000 --high-buffer-size 20000

Real run (detached):
    uv run modal run -d modal_train_hiro.py::train_hiro \
        --env-id PickCube-v1 --total-timesteps 1000000 --track
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("train-hiro")


@app.function(
    image=image,
    gpu="T4",
    timeout=15 * 60 * 60,
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_hiro(
    env_id: str = "PickCube-v1",
    subgoal_variant: str = "hybrid",
    robot_uids: str = "panda",
    control_mode: str = "pd_joint_delta_pos",
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    num_envs: int = 32,
    num_eval_envs: int = 16,
    num_eval_steps: int = 50,
    render_size: int = 512,
    eval_freq: int = 50_000,
    learning_starts: int = 4_000,
    c: int = 10,
    subgoal_scale: float = 0.15,
    subgoal_mode: str = "hiro",
    gamma_low: float = 0.95,
    gamma_high: float = 0.8,
    low_buffer_size: int = 1_000_000,
    high_buffer_size: int = 200_000,
    use_off_policy_correction: bool = True,
    use_phased_reward: bool = True,
    pbrs_alpha: float = 0.5,
    worker_extrinsic_weight: float = 0.0,
    place_pbrs_beta: float = 0.0,
    reach_pbrs_weight: float = 0.0,
    anneal_w: bool = False,
    anneal_w_grasp_threshold: float = 0.4,
    manager_utd_mult: int = 1,
    partial_reset: bool = True,
    oracle_manager: bool = False,
    flat_worker: bool = False,
    pretrained_worker_ckpt: str = "",
    resume_from_ckpt: str = "",
    pretrained_manager_ckpt: str = "",
    freeze_manager: bool = False,
    manager_input_mode: str = "full",
    low_her_ratio: float = 0.0,
    latent_subgoal_dim: int = 6,
    track: bool = False,
    capture_video: bool = True,
    save_freq_steps: int = 200_000,
    exp_name: Optional[str] = None,
) -> tuple[str, bytes]:
    import glob
    import os
    import sys
    import threading

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from hiro.trainer import TrainArgs, train

    args = TrainArgs(
        env_id=env_id,
        subgoal_variant=subgoal_variant,
        robot_uids=robot_uids,
        control_mode=control_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        num_eval_steps=num_eval_steps,
        render_size=render_size,
        eval_freq=eval_freq,
        learning_starts=learning_starts,
        c=c,
        subgoal_scale=subgoal_scale,
        subgoal_mode=subgoal_mode,
        gamma_low=gamma_low,
        gamma_high=gamma_high,
        low_buffer_size=low_buffer_size,
        high_buffer_size=high_buffer_size,
        use_off_policy_correction=use_off_policy_correction,
        use_phased_reward=use_phased_reward,
        pbrs_alpha=pbrs_alpha,
        worker_extrinsic_weight=worker_extrinsic_weight,
        place_pbrs_beta=place_pbrs_beta,
        reach_pbrs_weight=reach_pbrs_weight,
        anneal_w=anneal_w,
        anneal_w_grasp_threshold=anneal_w_grasp_threshold,
        manager_utd_mult=manager_utd_mult,
        low_her_ratio=low_her_ratio,
        latent_subgoal_dim=latent_subgoal_dim,
        partial_reset=partial_reset,
        oracle_manager=oracle_manager,
        flat_worker=flat_worker,
        pretrained_worker_ckpt=pretrained_worker_ckpt,
        resume_from_ckpt=resume_from_ckpt,
        pretrained_manager_ckpt=pretrained_manager_ckpt,
        freeze_manager=freeze_manager,
        manager_input_mode=manager_input_mode,
        track=track,
        capture_video=capture_video,
        save_freq_steps=save_freq_steps,
        exp_name=exp_name,
        wandb_project="cs224r-hiro",
        output_root="/output/runs",
        device="cuda",
        buffer_device="cuda",
    )

    # Periodically commit the volume so a mid-training crash doesn't lose progress.
    stop_event = threading.Event()

    def _commit_periodically(interval: int = 600):
        while not stop_event.wait(interval):
            try:
                volume.commit()
                print("[volume.commit] periodic commit done")
            except Exception as exc:
                print(f"[volume.commit] periodic commit failed: {exc}")

    commit_thread = threading.Thread(target=_commit_periodically, daemon=True)
    commit_thread.start()

    try:
        run_name = train(args)
    finally:
        stop_event.set()
        commit_thread.join(timeout=10)
        volume.commit()

    run_dir = f"/output/runs/{run_name}"
    videos = sorted(glob.glob(f"{run_dir}/videos/*.mp4"), key=os.path.getmtime)
    video_bytes = b""
    if videos:
        print(f"Returning final video: {videos[-1]}")
        with open(videos[-1], "rb") as f:
            video_bytes = f.read()
    else:
        print(f"WARNING: no .mp4 found under {run_dir}/videos")
    return run_name, video_bytes


@app.local_entrypoint()
def main(
    env_id: str = "PickCube-v1",
    subgoal_variant: str = "hybrid",
    robot_uids: str = "panda",
    control_mode: str = "pd_joint_delta_pos",
    seed: int = 1,
    total_timesteps: int = 10_000,
    num_envs: int = 32,
    eval_freq: int = 5_000,
    learning_starts: int = 2_000,
    c: int = 10,
    subgoal_scale: float = 0.15,
    low_buffer_size: int = 100_000,
    high_buffer_size: int = 20_000,
    use_off_policy_correction: bool = True,
    low_her_ratio: float = 0.8,
    latent_subgoal_dim: int = 6,
    track: bool = False,
):
    """Foreground run. Defaults are SMOKE-sized; pass real sizes for a full run."""
    run_name, video_bytes = train_hiro.remote(
        env_id=env_id,
        subgoal_variant=subgoal_variant,
        robot_uids=robot_uids,
        control_mode=control_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        eval_freq=eval_freq,
        learning_starts=learning_starts,
        c=c,
        subgoal_scale=subgoal_scale,
        low_buffer_size=low_buffer_size,
        high_buffer_size=high_buffer_size,
        use_off_policy_correction=use_off_policy_correction,
        low_her_ratio=low_her_ratio,
        latent_subgoal_dim=latent_subgoal_dim,
        track=track,
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
    subgoal_variant: str = "hybrid",
    robot_uids: str = "panda",
    control_mode: str = "pd_joint_delta_pos",
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    c: int = 10,
    subgoal_scale: float = 0.15,
    use_off_policy_correction: bool = True,
    low_her_ratio: float = 0.8,
    latent_subgoal_dim: int = 6,
    track: bool = True,
):
    """Detached spawn for a full run."""
    fc = train_hiro.spawn(
        env_id=env_id,
        subgoal_variant=subgoal_variant,
        robot_uids=robot_uids,
        control_mode=control_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        c=c,
        subgoal_scale=subgoal_scale,
        use_off_policy_correction=use_off_policy_correction,
        low_her_ratio=low_her_ratio,
        latent_subgoal_dim=latent_subgoal_dim,
        track=track,
    )
    print(f"Spawned function call: {fc.object_id}")
    print("Fetch artifacts after completion with:")
    print("  .venv/bin/modal volume get cs224r-project-results /runs/<run_name> ./hiro_runs")
