"""
hiro/subgoal_space.py
---------------------
Defines the HIRO subgoal space: which observation dimensions carry the
subgoal signal, and the three core operations that depend on that choice.

HIRO mechanics (all operating only on s[indices]):
  - project(s)                    -> s[indices]
  - subgoal_transition(s, g, s')  -> s[indices] + g - s'[indices]   (g_next)
  - intrinsic_reward(s, g, s')    -> -||s[indices] + g - s'[indices]||_2

The absolute target implied by subgoal g at state s is T = s[indices] + g.
After the worker takes a step to s', the remaining displacement is T - s'[indices],
which becomes the next subgoal g_next.  If the worker reaches exactly T,
g_next = 0 and the intrinsic reward = 0 (maximum).

Subgoal space is world-frame Cartesian positions only — no joints, velocities,
or quaternions.  This keeps the subgoal Euclidean so the transition arithmetic
is valid, and keeps the manager's language task-relevant (EE position, object
positions).

Canonical spaces
----------------
  PickCube-v1  : tcp_xyz + cube_xyz          (6 dims)
  StackCube-v1 : tcp_xyz + cubeA_xyz + cubeB_xyz  (9 dims)

The indices assume obs_mode="state" with pd_joint_delta_pos control and
the standard Panda agent.  Run modal_verify_subgoal_space.py to confirm
before training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class SubgoalSpace:
    """
    Configurable subgoal space for HIRO.

    Parameters
    ----------
    indices : list[int]
        Indices into the flat observation vector that form the subgoal.
    obs_dim : int
        Total observation dimension — used only for bounds validation.
    label : str
        Human-readable description, e.g. "tcp_xyz+cube_xyz".
    """

    indices: list[int]
    obs_dim: int
    label: str = ""

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("indices must be non-empty")
        if len(self.indices) != len(set(self.indices)):
            raise ValueError(f"duplicate indices: {self.indices}")
        bad = [i for i in self.indices if not (0 <= i < self.obs_dim)]
        if bad:
            raise ValueError(
                f"indices {bad} out of range for obs_dim={self.obs_dim}"
            )

    @property
    def dim(self) -> int:
        """Dimensionality of the subgoal vector."""
        return len(self.indices)

    # ------------------------------------------------------------------
    # Core HIRO operations.
    # All inputs may be a single sample (..., obs_dim) or batched.
    # g always has shape (..., subgoal_dim).
    # ------------------------------------------------------------------

    def project(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract the subgoal dimensions from a full observation.
        (..., obs_dim) -> (..., subgoal_dim)
        """
        return obs[..., self.indices]

    def subgoal_transition(
        self,
        s: torch.Tensor,       # (..., obs_dim)     current observation
        g: torch.Tensor,       # (..., subgoal_dim) current subgoal
        s_next: torch.Tensor,  # (..., obs_dim)     next observation
    ) -> torch.Tensor:         # (..., subgoal_dim) next subgoal g'
        """
        g' = s[indices] + g - s'[indices]

        Tracks the absolute target T = s[indices] + g as the agent moves.
        The returned g' is the remaining displacement to T from s'.
        """
        return self.project(s) + g - self.project(s_next)

    def intrinsic_reward(
        self,
        s: torch.Tensor,       # (..., obs_dim)
        g: torch.Tensor,       # (..., subgoal_dim)
        s_next: torch.Tensor,  # (..., obs_dim)
    ) -> torch.Tensor:         # (...,)  one scalar per sample, always <= 0
        """
        r = -||s[indices] + g - s'[indices]||_2

        Zero when the worker reaches the target exactly; more negative the
        further it is from the target.
        """
        residual = self.subgoal_transition(s, g, s_next)
        return -torch.norm(residual, dim=-1)

    def __repr__(self) -> str:
        tag = f" ({self.label})" if self.label else ""
        return f"SubgoalSpace(dim={self.dim}, obs_dim={self.obs_dim}{tag})"


# ---------------------------------------------------------------------------
# Canonical per-task subgoal spaces.
# Indices verified against ManiSkill3 source (pick_cube.py / stack_cube.py)
# and confirmed by modal_verify_subgoal_space.py.
#
# PickCube-v1 flat obs layout (42 dims, obs_mode="state"):
#   [0:9]   agent.qpos
#   [9:18]  agent.qvel
#   [18]    extra.is_grasped
#   [19:26] extra.tcp_pose  -> [19:22] xyz  [22:26] quaternion
#   [26:29] extra.goal_pos  (fixed per episode)
#   [29:36] extra.obj_pose  -> [29:32] xyz  [32:36] quaternion
#   [36:39] extra.tcp_to_obj_pos
#   [39:42] extra.obj_to_goal_pos
#
# StackCube-v1 flat obs layout (48 dims, obs_mode="state"):
#   [0:9]   agent.qpos
#   [9:18]  agent.qvel
#   [18:25] extra.tcp_pose  -> [18:21] xyz  [21:25] quaternion
#   [25:32] extra.cubeA_pose -> [25:28] xyz [28:32] quaternion
#   [32:39] extra.cubeB_pose -> [32:35] xyz [35:39] quaternion
#   [39:42] extra.tcp_to_cubeA_pos
#   [42:45] extra.tcp_to_cubeB_pos
#   [45:48] extra.cubeA_to_cubeB_pos
# ---------------------------------------------------------------------------

HIRO_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": SubgoalSpace(
        indices=[19, 20, 21,   # tcp_xyz  : extra.tcp_pose[:3]  starts at obs[19]
                 29, 30, 31],  # cube_xyz : extra.obj_pose[:3]  starts at obs[29]
        obs_dim=42,
        label="tcp_xyz+cube_xyz",
    ),
    "StackCube-v1": SubgoalSpace(
        indices=[18, 19, 20,   # tcp_xyz   : extra.tcp_pose[:3]   starts at obs[18]
                 25, 26, 27,   # cubeA_xyz : extra.cubeA_pose[:3] starts at obs[25]
                 32, 33, 34],  # cubeB_xyz : extra.cubeB_pose[:3] starts at obs[32]
        obs_dim=48,
        label="tcp_xyz+cubeA_xyz+cubeB_xyz",
    ),
}


def get_subgoal_space(env_id: str) -> SubgoalSpace:
    """Return the canonical HIRO subgoal space for the given env."""
    if env_id not in HIRO_SUBGOAL_SPACES:
        raise ValueError(
            f"No canonical subgoal space defined for {env_id!r}. "
            f"Known envs: {list(HIRO_SUBGOAL_SPACES)}"
        )
    return HIRO_SUBGOAL_SPACES[env_id]
