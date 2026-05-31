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
Absolute (legacy):
  PickCube-v1  : tcp_xyz + cube_xyz
  StackCube-v1 : tcp_xyz + cubeA_xyz

Hybrid (revised HIRO default):
  PickCube-v1  : tcp_xyz + tcp_to_obj + obj_to_goal
  StackCube-v1 : tcp_xyz + tcp_to_cubeA + cubeA_to_cubeB

Object-centric:
  PickCube-v1  : tcp_to_obj + obj_to_goal
  StackCube-v1 : tcp_to_cubeA + cubeA_to_cubeB

Latent object-centric:
  Uses the same underlying geometric object-centric space as above, but the
  manager/worker communicate through a learned low-dimensional latent code.
  Reward and goal-transition math still happen in the geometric base space.

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

    def phased_progress_pbrs_intrinsic_reward(
        self,
        s: torch.Tensor,                # (..., obs_dim)
        g: torch.Tensor,                # (..., subgoal_dim) current residual-to-target at s
        s_next: torch.Tensor,           # (..., obs_dim)
        is_grasped_s: torch.Tensor,     # (...,)
        is_grasped_s_next: torch.Tensor,  # (...,)
        gamma: float,
        alpha: float,
        near_positions: Sequence[int],
        near_threshold: float = 0.04,
        stall_penalty: float = 0.05,
        reach_coef: float = 1.0,
        reach_temp: float = 10.0,
        gripper_qpos_indices: Sequence[int] | None = None,
        open_penalty_scale: float = 0.08,
        max_open_aperture: float = 0.04,
    ) -> torch.Tensor:
        """Progress-shaped worker reward for object-centric PickCube.

        Uses *progress* in the manager-conditioned residual rather than the
        absolute next residual. This makes "hover near the cube" unattractive:
        once progress stalls, the reward goes to ~0, and near-without-grasp is
        further penalized by `stall_penalty`.

        Reward:
            hand_progress   = ||g_hand|| - ||g'_hand||
            object_progress = ||g_obj || - ||g'_obj ||
            pbrs            = γ·α·is_grasped(s') − α·is_grasped(s)
            dense_reach     = c · (1 - tanh(t · ||tcp_to_obj(s')||))   [ungated]
            open_penalty    = k · normalized_aperture(s')
                              only when near the cube and not yet grasped

            r = hand_progress
              + dense_reach
              + is_grasped(s) * object_progress
              + pbrs
              - stall_penalty * 1[ near(s') and not grasped(s') ]
              - open_penalty

        `near_positions` are positions within the projected subgoal state used
        to define "near the cube". For PickCube object-centric this is simply
        the tcp_to_obj block.
        """
        next_residual = self.subgoal_transition(s, g, s_next)
        hand_now = torch.norm(g[..., self.hand_positions], dim=-1)
        hand_next = torch.norm(next_residual[..., self.hand_positions], dim=-1)
        hand_progress = hand_now - hand_next

        if self.object_positions:
            obj_now = torch.norm(g[..., self.object_positions], dim=-1)
            obj_next = torch.norm(next_residual[..., self.object_positions], dim=-1)
            object_progress = obj_now - obj_next
        else:
            object_progress = torch.zeros_like(hand_progress)

        igs = is_grasped_s.to(hand_progress.dtype)
        igs_next = is_grasped_s_next.to(hand_progress.dtype)
        pbrs = gamma * alpha * igs_next - alpha * igs

        projected_next = self.project(s_next)
        near_dist = torch.norm(projected_next[..., list(near_positions)], dim=-1)
        dense_reach = reach_coef * (1.0 - torch.tanh(reach_temp * near_dist))
        hovering = (near_dist < near_threshold) & (~is_grasped_s_next.bool())
        stall = hovering.to(hand_progress.dtype) * stall_penalty
        open_penalty = torch.zeros_like(hand_progress)
        if gripper_qpos_indices is not None:
            aperture_next = s_next[..., list(gripper_qpos_indices)].mean(dim=-1)
            open_fraction = torch.clamp(aperture_next / max_open_aperture, min=0.0, max=1.0)
            open_when_near = hovering
            open_penalty = (
                open_when_near.to(hand_progress.dtype)
                * open_penalty_scale
                * open_fraction
            )

        return hand_progress + dense_reach + igs * object_progress + pbrs - stall - open_penalty

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

OBJECT_CENTRIC_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": SubgoalSpace(
        indices=[36, 37, 38,   # tcp_to_obj_pos : extra.tcp_to_obj_pos
                 39, 40, 41],  # obj_to_goal_pos: extra.obj_to_goal_pos
        obs_dim=42,
        label="tcp_to_obj+obj_to_goal",
        object_dims=(3, 4, 5),  # obj_to_goal dims matter only once the cube is controlled
    ),
    "StackCube-v1": SubgoalSpace(
        indices=[39, 40, 41,   # tcp_to_cubeA_pos : extra.tcp_to_cubeA_pos
                 45, 46, 47],  # cubeA_to_cubeB_pos: extra.cubeA_to_cubeB_pos
        obs_dim=48,
        label="tcp_to_cubeA+cubeA_to_cubeB",
        object_dims=(3, 4, 5),  # cubeA->cubeB placement dims are grasp-gated
    ),
}

HYBRID_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": SubgoalSpace(
        indices=[19, 20, 21,   # tcp_xyz        : extra.tcp_pose[:3]
                 36, 37, 38,   # tcp_to_obj_pos : extra.tcp_to_obj_pos
                 39, 40, 41],  # obj_to_goal_pos: extra.obj_to_goal_pos
        obs_dim=42,
        label="tcp_xyz+tcp_to_obj+obj_to_goal",
        object_dims=(6, 7, 8),  # only the object->goal transport dims are grasp-gated
    ),
    "StackCube-v1": SubgoalSpace(
        indices=[18, 19, 20,   # tcp_xyz          : extra.tcp_pose[:3]
                 39, 40, 41,   # tcp_to_cubeA_pos : extra.tcp_to_cubeA_pos
                 45, 46, 47],  # cubeA_to_cubeB   : extra.cubeA_to_cubeB_pos
        obs_dim=48,
        label="tcp_xyz+tcp_to_cubeA+cubeA_to_cubeB",
        object_dims=(6, 7, 8),  # placement/transport dims are grasp-gated
    ),
}

SUBGOAL_SPACE_VARIANTS: dict[str, dict[str, SubgoalSpace]] = {
    "absolute": HIRO_SUBGOAL_SPACES,
    "hybrid": HYBRID_SUBGOAL_SPACES,
    "object_centric": OBJECT_CENTRIC_SUBGOAL_SPACES,
    "latent_object_centric": OBJECT_CENTRIC_SUBGOAL_SPACES,
}


def normalize_subgoal_variant(variant: str) -> str:
    key = variant.strip().lower().replace("-", "_")
    aliases = {
        "default": "hybrid",
        "abs": "absolute",
        "absolute": "absolute",
        "hybrid": "hybrid",
        "hiro": "hybrid",
        "revised_hiro": "hybrid",
        "task_aligned": "hybrid",
        "object": "object_centric",
        "object_centric": "object_centric",
        "objectcentric": "object_centric",
        "obj": "object_centric",
        "latent_object_centric": "latent_object_centric",
        "latent_objectcentric": "latent_object_centric",
        "latent-object-centric": "latent_object_centric",
        "latent_obj": "latent_object_centric",
        "stitch": "latent_object_centric",
        "policy_stitching": "latent_object_centric",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown subgoal variant {variant!r}. "
            f"Known variants: {sorted(set(aliases.values()))}"
        )
    return aliases[key]


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
    "StackCube-v1": PositionalGraspDetector(tcp_to_obj_indices=(39, 40, 41), obj_z_index=27),
}


def get_grasp_detector(env_id: str) -> "GraspReader | None":
    return GRASP_DETECTORS.get(env_id)


def get_subgoal_space(env_id: str, variant: str = "absolute") -> SubgoalSpace:
    """Return the requested HIRO subgoal space for the given env."""
    variant_key = normalize_subgoal_variant(variant)
    spaces = SUBGOAL_SPACE_VARIANTS[variant_key]
    if env_id not in spaces:
        raise ValueError(
            f"No {variant_key} subgoal space defined for {env_id!r}. "
            f"Known envs: {list(spaces)}"
        )
    return spaces[env_id]
