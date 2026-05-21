from mani_skill.agents.registration import REGISTERED_AGENTS
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv

from weak_panda import (
    DEFAULT_WEAK_PANDA_FORCE_SCALE,
    WeakPanda,
    WeakPandaWristCam,
)


def test_weak_pandas_registered() -> None:
    assert "weakpanda" in REGISTERED_AGENTS
    assert "weakpanda_wristcam" in REGISTERED_AGENTS
    assert WeakPanda.gripper_force_limit == 100 * DEFAULT_WEAK_PANDA_FORCE_SCALE
    assert (
        WeakPandaWristCam.gripper_force_limit
        == 100 * DEFAULT_WEAK_PANDA_FORCE_SCALE
    )


def test_tasks_accept_weak_panda_uids() -> None:
    assert "weakpanda" in PickCubeEnv.SUPPORTED_ROBOTS
    assert "weakpanda" in StackCubeEnv.SUPPORTED_ROBOTS
