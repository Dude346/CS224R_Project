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

For ManiSkill `obs_mode="state"`, the task-specific "extra" block lives at the
tail of the observation and has a fixed layout per task. We derive indices from
`obs_dim`, so the same object-centric path works across robots/grippers whose
agent-state prefix has a different size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    reward_mode: str = "intrinsic"
    qvel_slice: tuple[int, int] | None = None
    task_obs_indices: tuple[int, ...] = ()

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
        if self.qvel_slice is not None:
            qvel_start, qvel_end = self.qvel_slice
            if not (0 <= qvel_start <= qvel_end <= self.obs_dim):
                raise ValueError(
                    f"qvel_slice {self.qvel_slice} out of range for obs_dim={self.obs_dim}"
                )
        bad_task = [i for i in self.task_obs_indices if not (0 <= i < self.obs_dim)]
        if bad_task:
            raise ValueError(
                f"task_obs_indices {bad_task} out of range for obs_dim={self.obs_dim}"
            )
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

    def task_potential_intrinsic_reward(
        self,
        s: torch.Tensor,                  # (..., obs_dim)
        g: torch.Tensor,                  # (..., subgoal_dim) manager residual-to-target at s
        s_next: torch.Tensor,             # (..., obs_dim)
        is_grasped_s: torch.Tensor,       # (...,)
        is_grasped_s_next: torch.Tensor,  # (...,)
        reach_coef: float = 1.0,
        grasp_coef: float = 1.0,
        place_coef: float = 1.0,
        object_progress_coef: float = 2.0,
        tanh_temp: float = 5.0,
        settle_coef: float = 0.5,
        static_temp: float = 0.25,
    ) -> torch.Tensor:
        """Farming-proof worker reward for object-centric PickCube.

        REPLACES the old `phased_progress_pbrs_intrinsic_reward`, whose ungated
        `dense_reach` term paid ~+1/step just for keeping the gripper on the cube
        (no lifting required). That term (a) created a "grasp-and-hold" local
        optimum and (b) drowned the only transport signal (`object_progress`) at
        ~1% of the per-step reward, so the worker never lifted the cube to the
        (aerial) goal — exactly the observed plateau.

        Every term here is a *telescoping difference* of a potential, so by Ng et
        al. (1999) it is policy-invariant and cannot be farmed by hovering or by
        drop/regrasp cycling, and — crucially — there is NO per-step "hold tax",
        so reaching and holding the success state is never penalised.

        Task potential (object-centric layout: the projected subgoal vector IS
        [tcp_to_obj (hand dims), obj_to_goal (object dims)], so these distances
        are read straight off `project`). It mirrors PickCube's own dense reward:

            Φ(x) = reach_coef · (1 − tanh(k·‖tcp_to_obj(x)‖))
                 + grasp_coef · is_grasped(x)
                 + place_coef · is_grasped(x) · place_score(x)
                 + settle_coef · is_grasped(x) · place_score(x) · static(x)

        where:
            place_score(x) = 1 − tanh(k·‖obj_to_goal(x)‖)
            static(x)      = 1 − tanh(k_v·‖qvel(x)‖)

        Reward:
            r = Φ(s') − Φ(s)                               # task shaping (γ=1 telescope)
              + (‖g_hand‖ − ‖g'_hand‖)                     # follow the manager hand target
              + is_grasped(s) · object_progress_coef ·
                (‖g_obj‖ − ‖g'_obj‖)                       # follow the manager place target

        Φ is maximised at success (gripper on cube, grasped, cube at goal), so
        climbing Φ solves the task; holding static costs 0; *leaving* a high-Φ
        state — including dropping the cube — costs back the Φ it gives up, which
        is what makes lifting strictly better than holding.

        γ=1 (a pure potential difference) rather than γ·Φ(s')−Φ(s) is deliberate:
        with γ<1 the shaping leaves a residual −(1−γ)·Φ hold-tax that penalises
        the success state every step it is held (the old PBRS pathology). The
        telescoping form sums to Φ(s_T)−Φ(s_0) over an episode, a path-independent
        constant that does not change the optimal policy.
        """
        proj_s = self.project(s)
        proj_sn = self.project(s_next)
        next_residual = self.subgoal_transition(s, g, s_next)

        igs = is_grasped_s.to(s.dtype)
        igs_next = is_grasped_s_next.to(s.dtype)

        def _potential(obs: torch.Tensor, proj: torch.Tensor, grasp: torch.Tensor) -> torch.Tensor:
            reach_dist = torch.norm(proj[..., self.hand_positions], dim=-1)
            phi = reach_coef * (1.0 - torch.tanh(tanh_temp * reach_dist))
            phi = phi + grasp_coef * grasp
            if self.object_positions:
                place_dist = torch.norm(proj[..., self.object_positions], dim=-1)
                place_score = 1.0 - torch.tanh(tanh_temp * place_dist)
                phi = phi + place_coef * grasp * place_score
                qvel = proj.new_zeros(place_dist.shape)
                if self.qvel_slice is not None:
                    qvel_start, qvel_end = self.qvel_slice
                    qvel = torch.norm(obs[..., qvel_start:qvel_end], dim=-1)
                static_score = 1.0 - torch.tanh(static_temp * qvel)
                phi = phi + settle_coef * grasp * place_score * static_score
            return phi

        task_shaping = _potential(s_next, proj_sn, igs_next) - _potential(s, proj_s, igs)

        hand_now = torch.norm(g[..., self.hand_positions], dim=-1)
        hand_next = torch.norm(next_residual[..., self.hand_positions], dim=-1)
        hand_progress = hand_now - hand_next

        if self.object_positions:
            obj_now = torch.norm(g[..., self.object_positions], dim=-1)
            obj_next = torch.norm(next_residual[..., self.object_positions], dim=-1)
            object_progress = obj_now - obj_next
        else:
            object_progress = torch.zeros_like(hand_progress)

        return (
            task_shaping
            + hand_progress
            + igs * object_progress_coef * object_progress
        )

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
# PickCube extra tail (24 dims, obs_mode="state"):
#   [0]     extra.is_grasped
#   [1:8]   extra.tcp_pose  -> [1:4] xyz
#   [8:11]  extra.goal_pos
#   [11:18] extra.obj_pose  -> [11:14] xyz
#   [18:21] extra.tcp_to_obj_pos
#   [21:24] extra.obj_to_goal_pos
#
# StackCube extra tail (30 dims, obs_mode="state"):
#   [0:7]   extra.tcp_pose  -> [0:3] xyz
#   [7:14]  extra.cubeA_pose -> [7:10] xyz
#   [14:21] extra.cubeB_pose -> [14:17] xyz
#   [21:24] extra.tcp_to_cubeA_pos
#   [24:27] extra.tcp_to_cubeB_pos
#   [27:30] extra.cubeA_to_cubeB_pos
# ---------------------------------------------------------------------------

PICKCUBE_EXTRA_DIM = 24
STACKCUBE_EXTRA_DIM = 30


def _infer_agent_layout(obs_dim: int, task_tail_dim: int) -> tuple[int, tuple[int, int]]:
    """Infer the agent-state prefix size and its qvel slice.

    ManiSkill `obs_mode="state"` for these tasks is `[agent.qpos, agent.qvel, extra...]`.
    Across the grippers/robots we care about, qpos and qvel contribute equally-sized
    prefixes, so the fixed-size task tail lets us recover the qvel slice from `obs_dim`.
    """
    agent_dim = obs_dim - task_tail_dim
    if agent_dim <= 0 or agent_dim % 2 != 0:
        raise ValueError(
            f"obs_dim={obs_dim} is incompatible with task tail {task_tail_dim}; "
            "expected an even-sized [qpos, qvel] prefix."
        )
    qvel_start = agent_dim // 2
    return agent_dim, (qvel_start, agent_dim)


def _pickcube_layout(obs_dim: int) -> dict[str, object]:
    agent_dim, qvel_slice = _infer_agent_layout(obs_dim, PICKCUBE_EXTRA_DIM)
    base = agent_dim
    return dict(
        obs_dim=obs_dim,
        qvel_slice=qvel_slice,
        task_obs_indices=tuple(range(base, obs_dim)),
        is_grasped=base,
        tcp_xyz=[base + 1, base + 2, base + 3],
        obj_xyz=[base + 11, base + 12, base + 13],
        tcp_to_obj=[base + 18, base + 19, base + 20],
        obj_to_goal=[base + 21, base + 22, base + 23],
    )


def _stackcube_layout(obs_dim: int) -> dict[str, object]:
    agent_dim, qvel_slice = _infer_agent_layout(obs_dim, STACKCUBE_EXTRA_DIM)
    base = agent_dim
    return dict(
        obs_dim=obs_dim,
        qvel_slice=qvel_slice,
        task_obs_indices=tuple(range(base, obs_dim)),
        tcp_xyz=[base + 0, base + 1, base + 2],
        cubeA_xyz=[base + 7, base + 8, base + 9],
        tcp_to_cubeA=[base + 21, base + 22, base + 23],
        tcp_to_cubeB=[base + 24, base + 25, base + 26],
        cubeA_to_cubeB=[base + 27, base + 28, base + 29],
        cubeA_z=base + 9,
    )


def _build_subgoal_space(env_id: str, variant: str, obs_dim: int) -> SubgoalSpace:
    if env_id == "PickCube-v1":
        layout = _pickcube_layout(obs_dim)
        if variant == "absolute":
            return SubgoalSpace(
                indices=[*layout["tcp_xyz"], *layout["obj_xyz"]],
                obs_dim=obs_dim,
                label="tcp_xyz+cube_xyz",
                object_dims=(3, 4, 5),
                qvel_slice=layout["qvel_slice"],
                task_obs_indices=layout["task_obs_indices"],
            )
        if variant == "hybrid":
            return SubgoalSpace(
                indices=[*layout["tcp_xyz"], *layout["tcp_to_obj"], *layout["obj_to_goal"]],
                obs_dim=obs_dim,
                label="tcp_xyz+tcp_to_obj+obj_to_goal",
                object_dims=(6, 7, 8),
                qvel_slice=layout["qvel_slice"],
                task_obs_indices=layout["task_obs_indices"],
            )
        if variant in {"object_centric", "latent_object_centric"}:
            return SubgoalSpace(
                indices=[*layout["tcp_to_obj"], *layout["obj_to_goal"]],
                obs_dim=obs_dim,
                label="tcp_to_obj+obj_to_goal",
                object_dims=(3, 4, 5),
                reward_mode="pickcube_task_potential",
                qvel_slice=layout["qvel_slice"],
                task_obs_indices=layout["task_obs_indices"],
            )
    elif env_id == "StackCube-v1":
        layout = _stackcube_layout(obs_dim)
        if variant == "absolute":
            return SubgoalSpace(
                indices=[*layout["tcp_xyz"], *layout["cubeA_xyz"]],
                obs_dim=obs_dim,
                label="tcp_xyz+cubeA_xyz",
                object_dims=(3, 4, 5),
                qvel_slice=layout["qvel_slice"],
                task_obs_indices=layout["task_obs_indices"],
            )
        if variant == "hybrid":
            return SubgoalSpace(
                indices=[*layout["tcp_xyz"], *layout["tcp_to_cubeA"], *layout["cubeA_to_cubeB"]],
                obs_dim=obs_dim,
                label="tcp_xyz+tcp_to_cubeA+cubeA_to_cubeB",
                object_dims=(6, 7, 8),
                qvel_slice=layout["qvel_slice"],
                task_obs_indices=layout["task_obs_indices"],
            )
        if variant in {"object_centric", "latent_object_centric"}:
            return SubgoalSpace(
                indices=[*layout["tcp_to_cubeA"], *layout["cubeA_to_cubeB"]],
                obs_dim=obs_dim,
                label="tcp_to_cubeA+cubeA_to_cubeB",
                object_dims=(3, 4, 5),
                qvel_slice=layout["qvel_slice"],
                task_obs_indices=layout["task_obs_indices"],
            )
    raise ValueError(
        f"No {variant} subgoal space defined for {env_id!r}. "
        f"Known envs: ['PickCube-v1', 'StackCube-v1']"
    )

HIRO_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": _build_subgoal_space("PickCube-v1", "absolute", obs_dim=42),
    "StackCube-v1": _build_subgoal_space("StackCube-v1", "absolute", obs_dim=48),
}

OBJECT_CENTRIC_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": _build_subgoal_space("PickCube-v1", "object_centric", obs_dim=42),
    "StackCube-v1": _build_subgoal_space("StackCube-v1", "object_centric", obs_dim=48),
}

HYBRID_SUBGOAL_SPACES: dict[str, SubgoalSpace] = {
    "PickCube-v1": _build_subgoal_space("PickCube-v1", "hybrid", obs_dim=42),
    "StackCube-v1": _build_subgoal_space("StackCube-v1", "hybrid", obs_dim=48),
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


def get_grasp_detector(
    env_id: str, robot_uids: str = "panda", obs_dim: "int | None" = None
) -> "GraspReader | None":
    """Grasp-state reader for the worker's phased reward.

    Fetch robots route to the Fetch obs layout. Otherwise, when obs_dim is given
    the index is derived from the (panda) obs layout for that size; else the
    canonical table is used.
    """
    if _is_fetch(robot_uids):
        return FETCH_GRASP_DETECTORS.get(env_id)
    if obs_dim is not None:
        if env_id == "PickCube-v1":
            return ObsBitGraspReader(obs_index=_pickcube_layout(obs_dim)["is_grasped"])
        if env_id == "StackCube-v1":
            layout = _stackcube_layout(obs_dim)
            return PositionalGraspDetector(
                tcp_to_obj_indices=tuple(layout["tcp_to_cubeA"]),
                obj_z_index=layout["cubeA_z"],
            )
        return None
    return GRASP_DETECTORS.get(env_id)


def get_subgoal_space(
    env_id: str, mode: str = "hiro", robot_uids: str = "panda", obs_dim: "int | None" = None
) -> SubgoalSpace:
    """Return the subgoal space for the given env.

    Two selection schemes are supported (the project merged two lines of work):
      - geometric modes: mode="hiro" (tcp+object) | "cube_centric" (object-only),
        routed by robot_uids ("fetch" -> Fetch obs layout, else Panda layout);
      - named variants: mode="object_centric" | "latent_object_centric" | "hybrid"
        | "absolute", built from the obs layout (pass obs_dim for non-standard sizes).
    """
    if _is_fetch(robot_uids):
        table = FETCH_CUBE_CENTRIC_SUBGOAL_SPACES if mode == "cube_centric" else FETCH_HIRO_SUBGOAL_SPACES
        if env_id not in table:
            raise ValueError(
                f"No {mode} Fetch subgoal space defined for {env_id!r}. Known envs: {list(table)}"
            )
        return table[env_id]
    if mode in ("hiro", "cube_centric"):
        table = CUBE_CENTRIC_SUBGOAL_SPACES if mode == "cube_centric" else HIRO_SUBGOAL_SPACES
        if env_id not in table:
            raise ValueError(
                f"No {mode} subgoal space defined for {env_id!r} (robot={robot_uids!r}). Known envs: {list(table)}"
            )
        return table[env_id]
    variant_key = normalize_subgoal_variant(mode)
    if obs_dim is not None:
        return _build_subgoal_space(env_id, variant_key, obs_dim=obs_dim)
    spaces = SUBGOAL_SPACE_VARIANTS[variant_key]
    if env_id not in spaces:
        raise ValueError(
            f"No {variant_key} subgoal space defined for {env_id!r}. Known envs: {list(spaces)}"
        )
    return spaces[env_id]


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
