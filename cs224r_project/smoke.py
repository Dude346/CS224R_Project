from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

SUPPORTED_ENVS = ("PickCube-v1", "StackCube-v1")
DEFAULT_OBS_MODE = "state+rgb"
DEFAULT_FRAME_DIR = "/vol/smoke_frames"


def _to_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    return tuple(int(dim) for dim in shape)


def _describe_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _describe_observation(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe_observation(subvalue) for subvalue in value]

    shape = _to_shape(value)
    dtype = getattr(value, "dtype", None)
    if shape:
        return {
            "shape": shape,
            "dtype": str(dtype) if dtype is not None else None,
        }
    return type(value).__name__


def _to_scalar(value: Any) -> float:
    import numpy as np

    return float(np.asarray(value).reshape(-1)[0])


def _extract_first_rgb_frame(obs: Any) -> tuple[str, Any] | tuple[None, None]:
    sensor_data = obs.get("sensor_data") if isinstance(obs, dict) else None
    if not isinstance(sensor_data, dict):
        return None, None

    for camera_name in sorted(sensor_data.keys()):
        camera_obs = sensor_data[camera_name]
        if isinstance(camera_obs, dict) and "rgb" in camera_obs:
            return str(camera_name), camera_obs["rgb"]
    return None, None


def _save_rgb_frame(frame: Any, output_path: str) -> None:
    import numpy as np
    from PIL import Image

    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    frame_array = np.asarray(frame)
    if frame_array.ndim == 4:
        frame_array = frame_array[0]
    image = Image.fromarray(frame_array.astype(np.uint8), mode="RGB")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def is_known_graphics_backend_error(exc: BaseException) -> bool:
    text = str(exc)
    markers = (
        "vk::createInstanceUnique: ErrorIncompatibleDriver",
        "ErrorInitializationFailed",
        "Failed to find system libvulkan",
    )
    return any(marker in text for marker in markers)


def should_skip_local_smoke_for_platform(exc: BaseException) -> bool:
    return platform.system() == "Darwin" and is_known_graphics_backend_error(exc)


def run_env_smoke(
    env_id: str,
    seed: int = 0,
    num_steps: int = 8,
    obs_mode: str = DEFAULT_OBS_MODE,
    robot_uids: str = "panda",
    save_first_frame: bool = True,
    frame_dir: str = DEFAULT_FRAME_DIR,
) -> dict[str, Any]:
    """Create a ManiSkill env, reset it, and step a few random actions."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers ManiSkill envs with gymnasium
    import cs224r_project.mani_skill_ext  # noqa: F401 - registers custom robots

    if env_id not in SUPPORTED_ENVS:
        supported = ", ".join(SUPPORTED_ENVS)
        raise ValueError(f"Unsupported env_id {env_id!r}. Expected one of: {supported}")

    make_kwargs = dict(
        obs_mode=obs_mode,
        robot_uids=robot_uids,
        sim_backend="physx_cpu",
    )
    if platform.system() != "Darwin":
        make_kwargs["render_backend"] = "gpu"

    env = gym.make(env_id, **make_kwargs)
    try:
        obs, info = env.reset(seed=seed)
        action_shape = _to_shape(env.action_space.sample())
        obs_shape = _to_shape(obs)
        obs_summary = _describe_observation(obs)
        frame_camera = None
        frame_path = None
        if save_first_frame:
            frame_camera, frame = _extract_first_rgb_frame(obs)
            if frame is not None:
                safe_obs_mode = obs_mode.replace("+", "_")
                frame_path = str(
                    Path(frame_dir) / f"{env_id}_{safe_obs_mode}_seed{seed}_step0.png"
                )
                _save_rgb_frame(frame, frame_path)
        terminated = False
        truncated = False
        last_reward = 0.0

        for _ in range(num_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            last_reward = _to_scalar(reward)
            obs_shape = _to_shape(obs)
            obs_summary = _describe_observation(obs)
            if terminated or truncated:
                obs, info = env.reset()
                obs_shape = _to_shape(obs)
                obs_summary = _describe_observation(obs)

        return {
            "env_id": env_id,
            "obs_mode": obs_mode,
            "robot_uids": robot_uids,
            "seed": seed,
            "num_steps": num_steps,
            "obs_shape": obs_shape if obs_shape else None,
            "obs_summary": obs_summary,
            "action_shape": action_shape,
            "frame_camera": frame_camera,
            "frame_path": frame_path,
            "last_reward": last_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info_keys": sorted(str(key) for key in info.keys()),
        }
    finally:
        env.close()
