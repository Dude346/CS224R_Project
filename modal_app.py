from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "cs224r-maniskill"
PROJECT_ROOT = Path(__file__).parent
REMOTE_ROOT = "/root/project"
VOLUME_ROOT = "/vol"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libvulkan1",
    )
    # We use frozen=False for now because the lockfile still needs to be refreshed
    # after adding the ManiSkill stack to pyproject.toml.
    .uv_sync(uv_project_dir=str(PROJECT_ROOT), frozen=False)
    .env(
        {
            "MS_ASSET_DIR": f"{VOLUME_ROOT}/mani_skill_assets",
            "MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1",
            "PYTHONPATH": REMOTE_ROOT,
            "WANDB_DIR": f"{VOLUME_ROOT}/wandb",
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

volume = modal.Volume.from_name("cs224r-project-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    gpu="T4",
    timeout=20 * 60,
)
def smoke_test_env(
    env_id: str,
    seed: int = 0,
    num_steps: int = 8,
    obs_mode: str = "state+rgb",
    robot_uids: str = "panda",
    save_first_frame: bool = True,
) -> dict[str, object]:
    from smoke import run_env_smoke

    result = run_env_smoke(
        env_id=env_id,
        seed=seed,
        num_steps=num_steps,
        obs_mode=obs_mode,
        robot_uids=robot_uids,
        save_first_frame=save_first_frame,
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    env_id: str = "PickCube-v1",
    seed: int = 0,
    num_steps: int = 8,
    obs_mode: str = "state+rgb",
    robot_uids: str = "panda",
    save_first_frame: bool = True,
) -> None:
    result = smoke_test_env.remote(
        env_id=env_id,
        seed=seed,
        num_steps=num_steps,
        obs_mode=obs_mode,
        robot_uids=robot_uids,
        save_first_frame=save_first_frame,
    )
    print(result)
