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

DEFAULT_WEAK_PANDA_FORCE_SCALE = 0.75

# Tune this single number to experiment with the damped-gripper embodiment.
#  <1.0  → snappier / less viscous gripper (may oscillate or overshoot)
#  >1.0  → sluggish / more viscous gripper (slower to close, more damped)
#  =1.0  → identical to stock Panda (no perturbation)
DEFAULT_DAMPED_PANDA_DAMPING_SCALE = 1.5

# Tune this single number to experiment with the scaled-action embodiment.
# Scales the arm's per-step joint-delta bounds for pd_joint_delta_pos control.
#  <1.0  → arm responds more slowly (smaller per-step motion at same policy logits)
#  >1.0  → arm responds more aggressively (larger per-step motion)
#  =1.0  → identical to stock Panda (no perturbation)
DEFAULT_SCALED_PANDA_ACTION_SCALE = 0.75


@register_agent()
class WeakPanda(Panda):
    """Panda robot with reduced gripper drive force."""

    uid = "weakpanda"
    # CRITICAL: explicitly copy Panda's keyframes into this subclass's __dict__.
    # @register_agent / the env's pose-initialization path appears to use
    # cls.__dict__.get("keyframes", ...) rather than walking the MRO, so
    # subclasses that rely on inheritance get NO keyframes registered and the
    # env falls back to zero qpos — which silently breaks the entire pipeline.
    keyframes = Panda.keyframes
    gripper_force_limit = Panda.gripper_force_limit * DEFAULT_WEAK_PANDA_FORCE_SCALE


@register_agent()
class WeakPandaWristCam(PandaWristCam):
    """Wrist-camera Panda robot with reduced gripper drive force."""

    uid = "weakpanda_wristcam"
    keyframes = PandaWristCam.keyframes
    gripper_force_limit = (
        PandaWristCam.gripper_force_limit * DEFAULT_WEAK_PANDA_FORCE_SCALE
    )


@register_agent()
class DampedPanda(Panda):
    """Panda robot with modified gripper damping (full strength preserved)."""

    uid = "dampedpanda"
    keyframes = Panda.keyframes
    gripper_damping = Panda.gripper_damping * DEFAULT_DAMPED_PANDA_DAMPING_SCALE


@register_agent()
class DampedPandaWristCam(PandaWristCam):
    """Wrist-camera Panda with modified gripper damping (full strength preserved)."""

    uid = "dampedpanda_wristcam"
    keyframes = PandaWristCam.keyframes
    gripper_damping = (
        PandaWristCam.gripper_damping * DEFAULT_DAMPED_PANDA_DAMPING_SCALE
    )


@register_agent()
class WeakDampedPanda(Panda):
    """Panda robot with BOTH reduced gripper force AND modified gripper damping.

    Compound perturbation for the additive-effects ablation: stacks WeakPanda's
    force scale and DampedPanda's damping scale on the same robot. Uses the
    existing DEFAULT_WEAK_PANDA_FORCE_SCALE and DEFAULT_DAMPED_PANDA_DAMPING_SCALE
    constants so single-variable and compound experiments share scale values.
    """

    uid = "weakdampedpanda"
    keyframes = Panda.keyframes
    gripper_force_limit = Panda.gripper_force_limit * DEFAULT_WEAK_PANDA_FORCE_SCALE
    gripper_damping = Panda.gripper_damping * DEFAULT_DAMPED_PANDA_DAMPING_SCALE


@register_agent()
class WeakDampedPandaWristCam(PandaWristCam):
    """Wrist-camera Panda with both gripper-force and gripper-damping perturbations."""

    uid = "weakdampedpanda_wristcam"
    keyframes = PandaWristCam.keyframes
    gripper_force_limit = (
        PandaWristCam.gripper_force_limit * DEFAULT_WEAK_PANDA_FORCE_SCALE
    )
    gripper_damping = (
        PandaWristCam.gripper_damping * DEFAULT_DAMPED_PANDA_DAMPING_SCALE
    )


@register_agent()
class ScaledActionPanda(Panda):
    """Panda robot whose per-step arm joint-delta bounds are scaled by a constant.

    Only `pd_joint_delta_pos` is affected (the controller we use for joint training).
    The gripper bounds are left at stock so grasp dynamics are unchanged — the
    perturbation is purely about how much physical arm motion the same policy
    logits produce per step.
    """

    uid = "scaledpanda"
    keyframes = Panda.keyframes

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        if "pd_joint_delta_pos" in configs:
            arm = configs["pd_joint_delta_pos"]["arm"]
            arm.lower = arm.lower * DEFAULT_SCALED_PANDA_ACTION_SCALE
            arm.upper = arm.upper * DEFAULT_SCALED_PANDA_ACTION_SCALE
        return configs


@register_agent()
class ScaledActionPandaWristCam(PandaWristCam):
    """Wrist-camera Panda with scaled per-step arm joint-delta bounds."""

    uid = "scaledpanda_wristcam"
    keyframes = PandaWristCam.keyframes

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        if "pd_joint_delta_pos" in configs:
            arm = configs["pd_joint_delta_pos"]["arm"]
            arm.lower = arm.lower * DEFAULT_SCALED_PANDA_ACTION_SCALE
            arm.upper = arm.upper * DEFAULT_SCALED_PANDA_ACTION_SCALE
        return configs


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _register_task_support() -> None:
    for robot_uid in (
        "weakpanda",
        "weakpanda_wristcam",
        "dampedpanda",
        "dampedpanda_wristcam",
        "weakdampedpanda",
        "weakdampedpanda_wristcam",
        "scaledpanda",
        "scaledpanda_wristcam",
    ):
        PICK_CUBE_CONFIGS.setdefault(robot_uid, deepcopy(PICK_CUBE_CONFIGS["panda"]))
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


def _patch_table_scene_builder() -> None:
    """Make TableSceneBuilder.initialize recognize our custom robot uids.

    Critical: ManiSkill's TableSceneBuilder.initialize hardcodes an if/elif
    chain on `self.env.robot_uids` to set the initial qpos for the robot.
    Our custom uids (weakpanda / dampedpanda / scaledpanda + WristCam variants)
    aren't in any branch, so the env silently falls through to ZERO qpos —
    which puts the arm in a flat T-pose with the gripper closed, making the
    task essentially unsolvable. That's the root cause of the "horrible
    convergence" we observed in every custom-robot experiment.

    This patch temporarily remaps custom uids to their stock equivalents
    (panda / panda_wristcam) just for the duration of the initialize() call,
    then restores the original uid afterwards.
    """
    from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder

    if getattr(TableSceneBuilder.initialize, "_cs224r_table_scene_patch", False):
        return

    _original_initialize = TableSceneBuilder.initialize

    CUSTOM_TO_STOCK = {
        "weakpanda": "panda",
        "weakpanda_wristcam": "panda_wristcam",
        "dampedpanda": "panda",
        "dampedpanda_wristcam": "panda_wristcam",
        "weakdampedpanda": "panda",
        "weakdampedpanda_wristcam": "panda_wristcam",
        "scaledpanda": "panda",
        "scaledpanda_wristcam": "panda_wristcam",
    }

    def patched_initialize(self, env_idx):
        original_uid = self.env.robot_uids
        if isinstance(original_uid, str) and original_uid in CUSTOM_TO_STOCK:
            self.env.robot_uids = CUSTOM_TO_STOCK[original_uid]
            try:
                return _original_initialize(self, env_idx)
            finally:
                self.env.robot_uids = original_uid
        return _original_initialize(self, env_idx)

    patched_initialize._cs224r_table_scene_patch = True  # type: ignore[attr-defined]
    TableSceneBuilder.initialize = patched_initialize


def ensure_custom_robots_registered() -> None:
    _register_task_support()
    _patch_pick_cube_reward()
    _patch_table_scene_builder()


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
