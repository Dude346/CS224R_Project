import pytest

from cs224r_project.smoke import (
    DEFAULT_OBS_MODE,
    SUPPORTED_ENVS,
    run_env_smoke,
    should_skip_local_smoke_for_platform,
)


@pytest.mark.parametrize("env_id", SUPPORTED_ENVS)
def test_state_env_smoke(env_id: str) -> None:
    try:
        result = run_env_smoke(
            env_id=env_id,
            seed=0,
            num_steps=2,
            obs_mode=DEFAULT_OBS_MODE,
            robot_uids="panda",
            save_first_frame=False,
        )
    except RuntimeError as exc:
        if should_skip_local_smoke_for_platform(exc):
            pytest.skip(
                "Local macOS Vulkan/MoltenVK is not configured for SAPIEN. "
                "Run this smoke test on Modal/Linux or install the Vulkan SDK."
            )
        raise

    assert result["env_id"] == env_id
    assert result["obs_mode"] == DEFAULT_OBS_MODE
    assert result["robot_uids"] == "panda"
    assert result["num_steps"] == 2
    assert result["action_shape"]
    assert result["obs_summary"]
