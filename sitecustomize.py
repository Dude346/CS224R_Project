from __future__ import annotations

import os
from typing import Any

DEFAULT_TARGET_ENV_IDS = {"PickCube-v1", "StackCube-v1"}


def _get_target_env_ids() -> set[str]:
    raw = os.environ.get("CS224R_TARGET_ENV_IDS", ",".join(sorted(DEFAULT_TARGET_ENV_IDS)))
    return {item.strip() for item in raw.split(",") if item.strip()}


def _patch_gym_make() -> None:
    import gymnasium as gym

    if getattr(gym.make, "_cs224r_weak_gripper_patch", False):
        return

    original_make = gym.make
    robot_uids = os.environ.get("CS224R_ROBOT_UIDS", "").strip()
    target_env_ids = _get_target_env_ids()

    def patched_make(id_: Any, *args: Any, **kwargs: Any):
        if robot_uids and isinstance(id_, str) and id_ in target_env_ids:
            kwargs.setdefault("robot_uids", robot_uids)
        return original_make(id_, *args, **kwargs)

    patched_make._cs224r_weak_gripper_patch = True  # type: ignore[attr-defined]
    gym.make = patched_make


if os.environ.get("CS224R_ENABLE_WEAK_GRIPPER") == "1":
    import mani_skill.envs  # noqa: F401
    import weak_panda  # noqa: F401

    _patch_gym_make()
