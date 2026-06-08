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
from typing import Protocol, Sequence

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
    # Positions WITHIN the subgoal vector (0..dim-1) that are object/offset dims
    # the worker can only affect AFTER it has grasped the object. Empty => every dim
    # is always active (plain HIRO behaviour used by the original tests).
    object_dims: tuple[int, ...] = ()

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
        dim = len(self.indices)
        bad_obj = [d for d in self.object_dims if not (0 <= d < dim)]
        if bad_obj:
            raise ValueError(f"object_dims {bad_obj} out of range for subgoal dim={dim}")
        obj = set(self.object_dims)
        self.object_positions = list(self.object_dims)            # grasp-gated dims
        self.hand_positions = [i for i in range(dim) if i not in obj]  # always-active dims

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

    def phased_pbrs_intrinsic_reward(
        self,
        s: torch.Tensor,                # (..., obs_dim)
        g: torch.Tensor,                # (..., subgoal_dim)
        s_next: torch.Tensor,           # (..., obs_dim)
        is_grasped_s: torch.Tensor,     # (...,)  is_grasped at the CURRENT state s
        is_grasped_s_next: torch.Tensor,  # (...,) is_grasped at the next state s'
        gamma: float,                   # MUST equal the worker's MDP discount factor (gamma_low)
        alpha: float,                   # strength of the grasp potential
    ) -> torch.Tensor:                  # (...,)
        """Phase-aware worker reward with potential-based grasp shaping (Ng et al. 1999).

            r = -||hand_residual||                                  always active
              + is_grasped(s) * (-||object_residual||)              gated by current grasp
              + γ·α·is_grasped(s')  −  α·is_grasped(s)              PBRS shaping

        The PBRS term is policy-invariant by Ng's theorem: every drop-regrasp
        cycle costs α(1−γ), so the grasp signal cannot be farmed. γ must equal
        the worker's MDP discount factor for the invariance to hold exactly.

        The cube term is gated by is_grasped(s) (the state the action was taken
        in), NOT by is_grasped(s'). This means:
          - grasp step (s=F, s'=T): cube term off (no penalty for "far cube" at
            the moment of acquisition); only the PBRS +γα bonus fires.
          - drop step (s=T, s'=F): cube term ON with the residual-at-drop, plus
            the PBRS -α. Drops at the goal (cube_residual ≈ 0) only pay -α;
            drops in transit get the position-proportional cube penalty too.

        `hand_positions` / `object_positions` are positions within the subgoal
        vector; only `indices` differ between HIRO (absolute positions) and
        object-centric (relative offsets). The env reward is NOT touched -- this
        is only the low-level/worker reward.
        """
        residual = self.subgoal_transition(s, g, s_next)
        hand_term = -torch.norm(residual[..., self.hand_positions], dim=-1)
        if self.object_positions:
            cube_term = -torch.norm(residual[..., self.object_positions], dim=-1)
        else:
            cube_term = torch.zeros_like(hand_term)
        igs = is_grasped_s.to(hand_term.dtype)
        igs_next = is_grasped_s_next.to(hand_term.dtype)
        pbrs = gamma * alpha * igs_next - alpha * igs
        return hand_term + igs * cube_term + pbrs

    def phased_intrinsic_reward(
        self,
        s: torch.Tensor,             # (..., obs_dim)
        g: torch.Tensor,             # (..., subgoal_dim)
        s_next: torch.Tensor,        # (..., obs_dim)
        is_grasped: torch.Tensor,    # (...,) 1 if the object is currently grasped/controllable
        just_grasped: torch.Tensor,  # (...,) 1 on the step is_grasped flips False -> True
        grasp_bonus: float = 0.0,
    ) -> torch.Tensor:               # (...,)
        """Phase-aware worker reward (identical structure for HIRO & object-centric).

        Pre-grasp the object dims are physically unreachable, so they are dropped:
            r = -||residual_hand||
        Once grasped, the object term is added, plus a one-time bonus on the step
        the grasp is acquired:
            r = -||residual_hand|| - ||residual_object|| + grasp_bonus * just_grasped

        `hand_positions` / `object_positions` are positions within the subgoal
        vector; only `indices` differ between HIRO (absolute positions) and
        object-centric (relative offsets). The env reward is NOT touched — this is
        only the low-level/worker reward.
        """
        residual = self.subgoal_transition(s, g, s_next)          # (..., dim)
        hand_term = -torch.norm(residual[..., self.hand_positions], dim=-1)
        if self.object_positions:
            cube_term = -torch.norm(residual[..., self.object_positions], dim=-1)
        else:
            cube_term = torch.zeros_like(hand_term)
        ig = is_grasped.to(hand_term.dtype)
        jg = just_grasped.to(hand_term.dtype)
        return hand_term + ig * cube_term + jg * grasp_bonus

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
        indices=[19, 20, 21,   # tcp_xyz  : extra.tcp_pose[:3]  starts at obs[19]  (HAND)
                 29, 30, 31],  # cube_xyz : extra.obj_pose[:3]  starts at obs[29]  (OBJECT)
        obs_dim=42,
        label="tcp_xyz+cube_xyz",
        object_dims=(3, 4, 5),  # cube dims are grasp-gated (worker can't move cube pre-grasp)
    ),
    "StackCube-v1": SubgoalSpace(
        indices=[18, 19, 20,   # tcp_xyz   : extra.tcp_pose[:3]   starts at obs[18]  (HAND)
                 25, 26, 27],  # cubeA_xyz : extra.cubeA_pose[:3] starts at obs[25]  (OBJECT)
        # cubeB_xyz (obs[32:35]) deliberately EXCLUDED: cubeB is the static target
        # and never moves. Including it would put a flat -||g_cubeB|| penalty into
        # the worker reward that no action can ever reduce (the same pathology that
        # blocked v1-v3). cubeB position is still in obs, so the manager can still
        # condition on "where to place cubeA"; it just isn't a subgoal dim.
        obs_dim=48,
        label="tcp_xyz+cubeA_xyz",
        object_dims=(3, 4, 5),  # cubeA dims are grasp-gated
    ),
}


# ---------------------------------------------------------------------------
# Phase 5: CUBE-CENTRIC (object-only) subgoal spaces.
# The subgoal is ONLY the object position (no TCP/hand dims). Combined with the
# phased grasp-gated reward this makes hand_positions empty => hand_term == 0,
# so the worker can no longer farm reward by hovering the gripper at a TCP
# target without grasping (the Test-G hover hack). The worker earns intrinsic
# reward ONLY by (a) grasping [PBRS bonus] and (b) moving the GRASPED object
# toward the manager's commanded object target [grasp-gated cube term].
# Pre-grasp guidance comes from the hybrid env reward (worker_extrinsic_weight),
# so use this mode WITH the phased reward and w>0.
# ---------------------------------------------------------------------------

CUBE_CENTRIC_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": SubgoalSpace(
        indices=[29, 30, 31],          # cube_xyz only (extra.obj_pose[:3])
        obs_dim=42,
        label="cube_xyz (object-centric)",
        object_dims=(0, 1, 2),         # ALL dims grasp-gated => hand_term empty
    ),
    "StackCube-v1": SubgoalSpace(
        indices=[25, 26, 27],          # cubeA_xyz only (extra.cubeA_pose[:3])
        obs_dim=48,
        label="cubeA_xyz (object-centric)",
        object_dims=(0, 1, 2),
    ),
}


# ---------------------------------------------------------------------------
# Grasp-state readers for the phased/PBRS worker reward.
# PickCube already exposes a true binary is_grasped flag in the observation, so
# use that directly. StackCube does not, so it falls back to the old position-
# only heuristic. Keeping the interface callable lets the agent/trainer stay
# unchanged while giving PickCube the exact signal the env already computes.
# ---------------------------------------------------------------------------

class GraspReader(Protocol):
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class ObsBitGraspReader:
    obs_index: int
    threshold: float = 0.5

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.obs_index] > self.threshold


@dataclass
class PositionalGraspDetector:
    tcp_to_obj_indices: tuple[int, int, int]  # obs indices of the gripper->object vector
    obj_z_index: int                          # obs index of the object's height (z)
    near_threshold: float = 0.05              # gripper within 5 cm of the object
    lift_threshold: float = 0.035             # object center lifted above the table rest height

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(obs[..., list(self.tcp_to_obj_indices)], dim=-1)
        z = obs[..., self.obj_z_index]
        return (dist < self.near_threshold) & (z > self.lift_threshold)


GRASP_DETECTORS: dict[str, GraspReader] = {
    # PickCube exposes the true task grasp bit at obs[18]; use it directly.
    "PickCube-v1": ObsBitGraspReader(obs_index=18),
    # StackCube has no explicit grasp bit in obs_mode="state", so use the
    # embodiment-independent positional heuristic for cubeA.
    # lift_threshold lowered 0.035->0.025 (cubeA rest center z=0.02): detect a grasp when
    # cubeA is lifted even slightly, so the cube-only oracle's grasp-gated stack target fires.
    "StackCube-v1": PositionalGraspDetector(tcp_to_obj_indices=(39, 40, 41), obj_z_index=27, lift_threshold=0.025),
}


# ---------------------------------------------------------------------------
# FETCH robot support (PickCube-v1). ADDITIVE: the Panda / weak-panda obs layout
# is unchanged; only robot_uids starting with "fetch" routes to these tables.
#
# Fetch PickCube-v1 flat obs (54 dims, obs_mode="state", pd_joint_delta_pos):
#   [0:30]  agent proprioception (Fetch has a larger joint set than Panda)
#   [30]    extra.is_grasped
#   [31:38] extra.tcp_pose      -> [31:34] xyz  [34:38] quat
#   [38:41] extra.goal_pos
#   [41:48] extra.obj_pose      -> [41:44] xyz  [44:48] quat
#   [48:51] extra.tcp_to_obj_pos
#   [51:54] extra.obj_to_goal_pos
# tcp_xyz@31 / goal@38 / cube@41 value-matched on a live Fetch env
# (modal_inspect_fetch_obs.py); is_grasped@30 + the two delta vectors follow from
# ManiSkill's fixed _get_obs_extra ordering.
# ---------------------------------------------------------------------------

FETCH_HIRO_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": SubgoalSpace(
        indices=[31, 32, 33,   # tcp_xyz  (HAND)
                 41, 42, 43],  # cube_xyz (OBJECT)
        obs_dim=54,
        label="fetch tcp_xyz+cube_xyz",
        object_dims=(3, 4, 5),
    ),
}

FETCH_CUBE_CENTRIC_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": SubgoalSpace(
        indices=[41, 42, 43],          # cube_xyz only
        obs_dim=54,
        label="fetch cube_xyz (object-centric)",
        object_dims=(0, 1, 2),
    ),
}

FETCH_GRASP_DETECTORS: dict[str, GraspReader] = {
    "PickCube-v1": ObsBitGraspReader(obs_index=30),
}

FETCH_PLACE_RESIDUAL_DIMS: dict[str, tuple[int, ...]] = {
    "PickCube-v1": (51, 52, 53),
}

FETCH_REACH_DIMS: dict[str, tuple[int, ...]] = {
    "PickCube-v1": (48, 49, 50),
}


def _is_fetch(robot_uids: "str | None") -> bool:
    """True for the Fetch robot (and its wristcam variant). Everything else
    (panda + weak/damped/scaled panda variants) shares the Panda obs layout."""
    return isinstance(robot_uids, str) and robot_uids.startswith("fetch")


def get_grasp_detector(env_id: str, robot_uids: str = "panda") -> "GraspReader | None":
    if _is_fetch(robot_uids):
        return FETCH_GRASP_DETECTORS.get(env_id)
    return GRASP_DETECTORS.get(env_id)


def get_subgoal_space(env_id: str, mode: str = "hiro", robot_uids: str = "panda") -> SubgoalSpace:
    """Return the subgoal space for the given env.

    mode="hiro"         : canonical tcp+object positions (default; original).
    mode="cube_centric" : object-only positions (Phase 5; kills the hover hack).
    robot_uids          : "fetch" routes to the Fetch obs layout; anything else
                          (panda + weak-panda variants) uses the Panda layout.
    """
    if _is_fetch(robot_uids):
        table = FETCH_CUBE_CENTRIC_SUBGOAL_SPACES if mode == "cube_centric" else FETCH_HIRO_SUBGOAL_SPACES
    else:
        table = CUBE_CENTRIC_SUBGOAL_SPACES if mode == "cube_centric" else HIRO_SUBGOAL_SPACES
    if env_id not in table:
        raise ValueError(
            f"No {mode} subgoal space defined for {env_id!r} (robot={robot_uids!r}). Known envs: {list(table)}"
        )
    return table[env_id]


# ---------------------------------------------------------------------------
# Cube->goal residual dims: the env-provided vector whose L2 norm is the
# object-to-goal distance. Used by (a) the manager place-PBRS potential
# Phi = -beta*||cube-goal|| and (b) the eval cube-at-goal / subgoal-alignment
# diagnostics. PickCube exposes extra.obj_to_goal_pos at obs[39:42];
# StackCube exposes extra.cubeA_to_cubeB_pos at obs[45:48].
# ---------------------------------------------------------------------------

PLACE_RESIDUAL_DIMS: dict[str, tuple[int, ...]] = {
    "PickCube-v1": (39, 40, 41),
    "StackCube-v1": (45, 46, 47),
}


def get_place_residual_dims(env_id: str, robot_uids: str = "panda") -> "tuple[int, ...] | None":
    """Obs indices of the cube->goal vector (||.|| = cube-to-goal distance)."""
    if _is_fetch(robot_uids):
        return FETCH_PLACE_RESIDUAL_DIMS.get(env_id)
    return PLACE_RESIDUAL_DIMS.get(env_id)


# ---------------------------------------------------------------------------
# tcp->object vector dims, for the Phase-D pre-grasp reach bootstrap. ||.|| is
# the gripper-to-object distance. PickCube exposes extra.tcp_to_obj_pos at
# obs[36:39]; StackCube exposes extra.tcp_to_cubeA_pos at obs[39:42].
# ---------------------------------------------------------------------------

REACH_DIMS: dict[str, tuple[int, ...]] = {
    "PickCube-v1": (36, 37, 38),
    "StackCube-v1": (39, 40, 41),
}


def get_reach_dims(env_id: str, robot_uids: str = "panda") -> "tuple[int, ...] | None":
    """Obs indices of the tcp->object vector (||.|| = gripper-to-object distance)."""
    if _is_fetch(robot_uids):
        return FETCH_REACH_DIMS.get(env_id)
    return REACH_DIMS.get(env_id)


# ---------------------------------------------------------------------------
# Body-independent MANAGER INPUT: the object-centric features the manager reads
# instead of the full robot-specific obs -> [tcp_xyz, cube_xyz, goal_xyz] (9-D).
# These are the SAME semantic features on every robot (just at different obs
# indices), so a manager trained on them transfers across embodiments with
# different obs sizes (panda 42-D <-> fetch 54-D). PickCube only for now.
# ---------------------------------------------------------------------------

MANAGER_INPUT_DIMS: dict[str, tuple[int, ...]] = {
    "PickCube-v1": (19, 20, 21, 29, 30, 31, 26, 27, 28),  # tcp_xyz, cube_xyz, goal_xyz
}

FETCH_MANAGER_INPUT_DIMS: dict[str, tuple[int, ...]] = {
    "PickCube-v1": (31, 32, 33, 41, 42, 43, 38, 39, 40),  # tcp_xyz, cube_xyz, goal_xyz
}


def get_manager_input_dims(env_id: str, robot_uids: str = "panda") -> "tuple[int, ...] | None":
    """Obs indices of the body-independent manager input [tcp,cube,goal] (9-D)."""
    if _is_fetch(robot_uids):
        return FETCH_MANAGER_INPUT_DIMS.get(env_id)
    return MANAGER_INPUT_DIMS.get(env_id)
