from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill envs with gymnasium
import torch
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots import Panda as PandaBase
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv

DEFAULT_WEAK_PANDA_FORCE_SCALE = 0.3


@register_agent()
class WeakPanda(Panda):
    """Panda robot with reduced gripper drive force."""

    uid = "weakpanda"
    gripper_force_limit = Panda.gripper_force_limit * DEFAULT_WEAK_PANDA_FORCE_SCALE


@register_agent()
class WeakPandaWristCam(PandaWristCam):
    """Wrist-camera Panda robot with reduced gripper drive force."""

    uid = "weakpanda_wristcam"
    gripper_force_limit = (
        PandaWristCam.gripper_force_limit * DEFAULT_WEAK_PANDA_FORCE_SCALE
    )


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _register_task_support() -> None:
    PICK_CUBE_CONFIGS.setdefault("weakpanda", deepcopy(PICK_CUBE_CONFIGS["panda"]))
    PICK_CUBE_CONFIGS.setdefault(
        "weakpanda_wristcam", deepcopy(PICK_CUBE_CONFIGS["panda"])
    )

    for robot_uid in ("weakpanda", "weakpanda_wristcam"):
        _append_unique(PickCubeEnv.SUPPORTED_ROBOTS, robot_uid)
        _append_unique(StackCubeEnv.SUPPORTED_ROBOTS, robot_uid)


def _patch_pick_cube_reward() -> None:
    if getattr(PickCubeEnv.compute_dense_reward, "_cs224r_weak_panda_patch", False):
        return

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        # Treat WeakPanda like Panda here so static reward ignores the two finger joints.
        if isinstance(self.agent, PandaBase) or self.robot_uids == "widowxai":
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    compute_dense_reward._cs224r_weak_panda_patch = True  # type: ignore[attr-defined]
    PickCubeEnv.compute_dense_reward = compute_dense_reward


def ensure_custom_robots_registered() -> None:
    _register_task_support()
    _patch_pick_cube_reward()


def make_env(task_id: str, robot_uids: str = "panda", **kwargs):
    ensure_custom_robots_registered()
    return gym.make(task_id, robot_uids=robot_uids, **kwargs)


def make_weakened_panda_env(
    task_id: str,
    wristcam: bool = False,
    **kwargs,
):
    robot_uid = "weakpanda_wristcam" if wristcam else "weakpanda"
    return make_env(task_id, robot_uids=robot_uid, **kwargs)


ensure_custom_robots_registered()
