"""
hiro/tests/test_subgoal_space.py
---------------------------------
Unit tests for SubgoalSpace.  No environment needed — pure tensor arithmetic.

All expected values are hand-computed from the HIRO formula:
  target  = s[indices] + g
  g_next  = target - s'[indices]
  reward  = -||g_next||_2

Run locally:
    uv run pytest hiro/tests/test_subgoal_space.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from hiro.subgoal_space import (
    GRASP_DETECTORS,
    HIRO_SUBGOAL_SPACES,
    ObsBitGraspReader,
    PositionalGraspDetector,
    SubgoalSpace,
    get_subgoal_space,
)

# ---------------------------------------------------------------------------
# Shared fixture: 6-dim obs, subgoal on dims [1, 3, 5]
# Full obs: [*, s0, *, s1, *, s2]  where * dims are irrelevant noise.
# ---------------------------------------------------------------------------

@pytest.fixture
def sp() -> SubgoalSpace:
    return SubgoalSpace(indices=[1, 3, 5], obs_dim=6, label="test")


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_dim(self, sp):
        assert sp.dim == 3

    def test_repr_includes_dim(self, sp):
        assert "dim=3" in repr(sp)

    def test_empty_indices_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            SubgoalSpace(indices=[], obs_dim=6)

    def test_duplicate_indices_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            SubgoalSpace(indices=[0, 0, 1], obs_dim=6)

    def test_out_of_range_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            SubgoalSpace(indices=[0, 6], obs_dim=6)  # 6 is out of range

    def test_canonical_pickcube_dim(self):
        assert HIRO_SUBGOAL_SPACES["PickCube-v1"].dim == 6

    def test_canonical_stackcube_dim(self):
        # StackCube subgoal = tcp_xyz + cubeA_xyz (6 dims).
        # cubeB is deliberately excluded -- it's the static target, never moves,
        # so including it would put an action-independent penalty in the worker
        # reward (the same pathology that blocked the early HIRO runs).
        assert HIRO_SUBGOAL_SPACES["StackCube-v1"].dim == 6

    def test_get_subgoal_space_unknown_raises(self):
        with pytest.raises(ValueError):
            get_subgoal_space("UnknownEnv-v1")


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

class TestProject:
    def test_extracts_correct_dims(self, sp):
        # obs = [10, 1, 20, 2, 30, 3] — indices [1,3,5] → [1, 2, 3]
        s = torch.tensor([10.0, 1.0, 20.0, 2.0, 30.0, 3.0])
        assert torch.allclose(sp.project(s), torch.tensor([1.0, 2.0, 3.0]))

    def test_batched_shape(self, sp):
        s = torch.zeros(8, 6)
        s[:, [1, 3, 5]] = torch.tensor([1.0, 2.0, 3.0])
        out = sp.project(s)
        assert out.shape == (8, 3)
        assert torch.allclose(out, torch.ones(8, 3) * torch.tensor([1.0, 2.0, 3.0]))

    def test_irrelevant_dims_ignored(self, sp):
        s1 = torch.tensor([99.0, 1.0, 99.0, 2.0, 99.0, 3.0])
        s2 = torch.tensor([-99.0, 1.0, -99.0, 2.0, -99.0, 3.0])
        assert torch.allclose(sp.project(s1), sp.project(s2))


# ---------------------------------------------------------------------------
# subgoal_transition
# Hand-computed cases — each case includes the derivation.
# ---------------------------------------------------------------------------

class TestSubgoalTransition:
    def test_exact_achievement_gives_zero(self, sp):
        """
        s[idx]  = [0, 0, 0]
        g       = [1, 2, 3]
        target  = [1, 2, 3]
        s'[idx] = target = [1, 2, 3]
        g_next  = [0, 0, 0]
        """
        s      = torch.tensor([0., 0., 0., 0., 0., 0.])
        g      = torch.tensor([1., 2., 3.])
        s_next = torch.tensor([0., 1., 0., 2., 0., 3.])
        g_next = sp.subgoal_transition(s, g, s_next)
        assert torch.allclose(g_next, torch.zeros(3))

    def test_no_movement_returns_g_unchanged(self, sp):
        """
        s = s_next  ->  g_next = s[idx] + g - s[idx] = g
        """
        s      = torch.tensor([0., 1., 0., 2., 0., 3.])
        g      = torch.tensor([0.5, -1.0, 0.25])
        s_next = s.clone()
        g_next = sp.subgoal_transition(s, g, s_next)
        assert torch.allclose(g_next, g)

    def test_half_progress_halves_remaining(self, sp):
        """
        s[idx]  = [0, 0, 0];  g = [1, 0, 0]  ->  target = [1, 0, 0]
        s'[idx] = [0.5, 0, 0]  (moved halfway on dim 0)
        g_next  = [1-0.5, 0, 0] = [0.5, 0, 0]
        """
        s      = torch.tensor([0., 0., 0., 0., 0., 0.])
        g      = torch.tensor([1., 0., 0.])
        s_next = torch.tensor([0., 0.5, 0., 0., 0., 0.])
        g_next = sp.subgoal_transition(s, g, s_next)
        assert torch.allclose(g_next, torch.tensor([0.5, 0., 0.]))

    def test_multi_dim_partial_progress(self, sp):
        """
        s[idx]  = [1.0, 2.0, 3.0]
        g       = [-0.5, 0.5, -1.0]
        target  = [0.5, 2.5, 2.0]
        s'[idx] = [0.8, 2.3, 2.5]
        g_next  = [0.5-0.8, 2.5-2.3, 2.0-2.5] = [-0.3, 0.2, -0.5]
        """
        s      = torch.tensor([0., 1.0, 0., 2.0, 0., 3.0])
        g      = torch.tensor([-0.5, 0.5, -1.0])
        s_next = torch.tensor([0., 0.8, 0., 2.3, 0., 2.5])
        g_next = sp.subgoal_transition(s, g, s_next)
        assert torch.allclose(g_next, torch.tensor([-0.3, 0.2, -0.5]), atol=1e-5)

    def test_overshoot_gives_negative_remaining(self, sp):
        """
        target = [1, 0, 0];  s' overshoots to [1.5, 0, 0]
        g_next = [1-1.5, 0, 0] = [-0.5, 0, 0]  (pointing back toward target)
        """
        s      = torch.tensor([0., 0., 0., 0., 0., 0.])
        g      = torch.tensor([1., 0., 0.])
        s_next = torch.tensor([0., 1.5, 0., 0., 0., 0.])
        g_next = sp.subgoal_transition(s, g, s_next)
        assert torch.allclose(g_next, torch.tensor([-0.5, 0., 0.]))


# ---------------------------------------------------------------------------
# intrinsic_reward
# ---------------------------------------------------------------------------

class TestIntrinsicReward:
    def test_exact_achievement_gives_zero_reward(self, sp):
        """Perfect tracking -> reward = 0."""
        s      = torch.tensor([0., 0., 0., 0., 0., 0.])
        g      = torch.tensor([1., 2., 3.])
        s_next = torch.tensor([0., 1., 0., 2., 0., 3.])
        r = sp.intrinsic_reward(s, g, s_next)
        assert torch.allclose(r, torch.tensor(0.0))

    def test_no_movement_reward_equals_neg_norm_g(self, sp):
        """
        No movement -> residual = g -> reward = -||g||_2.
        g = [3, 4, 0]  ->  ||g|| = 5  ->  reward = -5
        """
        s      = torch.tensor([0., 0., 0., 0., 0., 0.])
        g      = torch.tensor([3., 4., 0.])
        s_next = s.clone()
        r = sp.intrinsic_reward(s, g, s_next)
        assert torch.allclose(r, torch.tensor(-5.0))

    def test_half_progress(self, sp):
        """
        g = [1, 0, 0];  moved halfway  ->  residual = [0.5, 0, 0]
        reward = -0.5
        """
        s      = torch.tensor([0., 0., 0., 0., 0., 0.])
        g      = torch.tensor([1., 0., 0.])
        s_next = torch.tensor([0., 0.5, 0., 0., 0., 0.])
        r = sp.intrinsic_reward(s, g, s_next)
        assert torch.allclose(r, torch.tensor(-0.5))

    def test_multi_dim_partial(self, sp):
        """
        residual = [-0.3, 0.2, -0.5]
        reward = -sqrt(0.09 + 0.04 + 0.25) = -sqrt(0.38) ≈ -0.61644
        """
        s      = torch.tensor([0., 1.0, 0., 2.0, 0., 3.0])
        g      = torch.tensor([-0.5, 0.5, -1.0])
        s_next = torch.tensor([0., 0.8, 0., 2.3, 0., 2.5])
        r = sp.intrinsic_reward(s, g, s_next)
        expected = -math.sqrt(0.09 + 0.04 + 0.25)
        assert abs(r.item() - expected) < 1e-5

    def test_reward_is_always_non_positive(self, sp):
        """By definition ||...||_2 >= 0, so reward <= 0 always."""
        torch.manual_seed(0)
        for _ in range(20):
            s      = torch.randn(6)
            g      = torch.randn(3)
            s_next = torch.randn(6)
            assert sp.intrinsic_reward(s, g, s_next).item() <= 0.0

    def test_batched_reward_shape(self, sp):
        """Batched input should return one reward per sample."""
        s      = torch.randn(16, 6)
        g      = torch.randn(16, 3)
        s_next = torch.randn(16, 6)
        r = sp.intrinsic_reward(s, g, s_next)
        assert r.shape == (16,)
        assert (r <= 0).all()


# ---------------------------------------------------------------------------
# Phase-aware worker reward
# Subgoal dim 4: hand = dims [0,1], object = dims [2,3] (grasp-gated).
# s=0, g=[1,0,0,1], proj(s')=[0.5,0,0,0]  ->  residual=[0.5,0,0,1]
#   hand_term = -||[0.5,0]|| = -0.5
#   cube_term = -||[0,1]||   = -1.0
# ---------------------------------------------------------------------------

class TestPhasedReward:
    @pytest.fixture
    def psp(self) -> SubgoalSpace:
        return SubgoalSpace(indices=[0, 1, 2, 3], obs_dim=4, object_dims=(2, 3))

    @staticmethod
    def _case():
        s      = torch.zeros(4)
        g      = torch.tensor([1.0, 0.0, 0.0, 1.0])
        s_next = torch.tensor([0.5, 0.0, 0.0, 0.0])
        return s, g, s_next

    def test_partition(self, psp):
        assert psp.hand_positions == [0, 1]
        assert psp.object_positions == [2, 3]

    def test_pregrasp_ignores_object_term(self, psp):
        s, g, sn = self._case()
        r = psp.phased_intrinsic_reward(s, g, sn, torch.tensor(0.0), torch.tensor(0.0), grasp_bonus=0.3)
        assert torch.allclose(r, torch.tensor(-0.5))           # hand only; cube term dropped

    def test_sustained_grasp_adds_object_term(self, psp):
        s, g, sn = self._case()
        r = psp.phased_intrinsic_reward(s, g, sn, torch.tensor(1.0), torch.tensor(0.0), grasp_bonus=0.3)
        assert torch.allclose(r, torch.tensor(-1.5))           # hand + cube, no bonus

    def test_grasp_step_adds_object_and_bonus(self, psp):
        s, g, sn = self._case()
        r = psp.phased_intrinsic_reward(s, g, sn, torch.tensor(1.0), torch.tensor(1.0), grasp_bonus=0.3)
        assert torch.allclose(r, torch.tensor(-1.2))           # hand + cube + bonus (-1.5 + 0.3)

    def test_batched_gating(self, psp):
        s = torch.zeros(4, 4)
        g = torch.zeros(4, 4); g[:, 0] = 1.0; g[:, 3] = 1.0
        sn = torch.zeros(4, 4); sn[:, 0] = 0.5
        is_grasped = torch.tensor([0.0, 1.0, 0.0, 1.0])
        just = torch.tensor([0.0, 0.0, 0.0, 1.0])
        r = psp.phased_intrinsic_reward(s, g, sn, is_grasped, just, grasp_bonus=0.3)
        assert r.shape == (4,)
        assert torch.allclose(r, torch.tensor([-0.5, -1.5, -0.5, -1.2]))

    def test_canonical_pickcube_partition(self):
        sp = HIRO_SUBGOAL_SPACES["PickCube-v1"]
        assert sp.hand_positions == [0, 1, 2]      # tcp_xyz always active
        assert sp.object_positions == [3, 4, 5]    # cube_xyz grasp-gated

    def test_empty_object_dims_equals_plain(self):
        # With no object dims, phased reward (grasped or not) == plain -||residual||.
        sp = SubgoalSpace(indices=[1, 3, 5], obs_dim=6)  # object_dims default ()
        s = torch.randn(6); g = torch.randn(3); sn = torch.randn(6)
        plain = sp.intrinsic_reward(s, g, sn)
        phased = sp.phased_intrinsic_reward(s, g, sn, torch.tensor(1.0), torch.tensor(0.0))
        assert torch.allclose(plain, phased)


# ---------------------------------------------------------------------------
# Potential-Based Reward Shaping (Ng et al. 1999) — the v5 worker reward.
# Φ(s) = α · is_grasped(s);  shaping = γ·Φ(s') − Φ(s).
#
# Per-transition shaping payouts (γ=0.95, α=0.5):
#   F → F:           γ·0 − 0   = 0
#   F → T (grasp):   γα − 0    = +0.475
#   T → T (hold):    γα − α    = -0.025  (small per-step "tax")
#   T → F (drop):    γ·0 − α   = -0.5
#
# Non-hackability: one drop-regrasp cycle nets α(γ−1) = -0.025 < 0.
# ---------------------------------------------------------------------------

class TestPBRSReward:
    GAMMA = 0.95
    ALPHA = 0.5

    @pytest.fixture
    def psp(self) -> SubgoalSpace:
        # Same fixture as TestPhasedReward: hand=dims[0,1], object=dims[2,3].
        return SubgoalSpace(indices=[0, 1, 2, 3], obs_dim=4, object_dims=(2, 3))

    @staticmethod
    def _setup():
        # Zero residual so we can read off the PBRS payout cleanly.
        s = torch.zeros(4)
        g = torch.zeros(4)
        s_next = torch.zeros(4)
        return s, g, s_next

    def _r(self, psp, igs, igs_next):
        s, g, sn = self._setup()
        return psp.phased_pbrs_intrinsic_reward(
            s, g, sn, torch.tensor(float(igs)), torch.tensor(float(igs_next)),
            gamma=self.GAMMA, alpha=self.ALPHA,
        ).item()

    def test_F_to_F_no_shaping(self, psp):
        assert self._r(psp, igs=0, igs_next=0) == pytest.approx(0.0)

    def test_F_to_T_grasp_bonus(self, psp):
        # +γα at grasp acquisition; cube_term off because is_grasped(s)=F.
        assert self._r(psp, igs=0, igs_next=1) == pytest.approx(self.GAMMA * self.ALPHA)

    def test_T_to_T_sustained_hold_tax(self, psp):
        # α(γ−1) per step; cube_term active (=0 since residual=0).
        assert self._r(psp, igs=1, igs_next=1) == pytest.approx(self.ALPHA * (self.GAMMA - 1.0))

    def test_T_to_F_drop_penalty(self, psp):
        # -α at drop; cube_term active (=0 since residual=0).
        assert self._r(psp, igs=1, igs_next=0) == pytest.approx(-self.ALPHA)

    def test_drop_regrasp_cycle_is_negative(self, psp):
        """The cornerstone non-hackability property: a drop-then-regrasp cycle
        must net strictly negative, so the agent cannot farm the grasp bonus."""
        drop = self._r(psp, igs=1, igs_next=0)         # -α
        regrasp = self._r(psp, igs=0, igs_next=1)      # +γα
        cycle = drop + regrasp
        assert cycle < 0
        assert cycle == pytest.approx(self.ALPHA * (self.GAMMA - 1.0))

    def test_drop_step_also_pays_cube_term(self, psp):
        # On a drop step with nonzero residual the worker pays BOTH the cube_term
        # penalty (proportional to residual) AND the PBRS -α.
        s = torch.zeros(4)
        g = torch.tensor([0.0, 0.0, 1.0, 0.0])   # object subgoal = move dim2 by 1
        s_next = torch.zeros(4)                  # object didn't move => residual = 1.0
        r = psp.phased_pbrs_intrinsic_reward(
            s, g, s_next, torch.tensor(1.0), torch.tensor(0.0),
            gamma=self.GAMMA, alpha=self.ALPHA,
        ).item()
        # cube_term = -1.0 (gated on by is_grasped(s)=1); PBRS = -α
        assert r == pytest.approx(-1.0 - self.ALPHA)

    def test_grasp_step_no_cube_penalty(self, psp):
        # At the grasp step is_grasped(s)=F, so cube_term is OFF -- the worker
        # isn't punished for "the cube is far" at the moment it just grasped.
        s = torch.zeros(4)
        g = torch.tensor([0.0, 0.0, 1.0, 0.0])
        s_next = torch.zeros(4)
        r = psp.phased_pbrs_intrinsic_reward(
            s, g, s_next, torch.tensor(0.0), torch.tensor(1.0),
            gamma=self.GAMMA, alpha=self.ALPHA,
        ).item()
        # cube_term=0 (gate off); PBRS = +γα; hand_term = 0
        assert r == pytest.approx(self.GAMMA * self.ALPHA)

    def test_batched(self, psp):
        s = torch.zeros(4, 4)
        g = torch.zeros(4, 4)
        sn = torch.zeros(4, 4)
        igs = torch.tensor([0.0, 0.0, 1.0, 1.0])      # F, F, T, T
        igs_next = torch.tensor([0.0, 1.0, 1.0, 0.0]) # F, T, T, F
        r = psp.phased_pbrs_intrinsic_reward(
            s, g, sn, igs, igs_next, gamma=self.GAMMA, alpha=self.ALPHA,
        )
        expected = torch.tensor([
            0.0,
            self.GAMMA * self.ALPHA,
            self.ALPHA * (self.GAMMA - 1.0),
            -self.ALPHA,
        ])
        assert torch.allclose(r, expected)


# ---------------------------------------------------------------------------
# Grasp-state readers
# ---------------------------------------------------------------------------

class TestGraspReaders:
    def test_pickcube_uses_true_obs_bit(self):
        det = GRASP_DETECTORS["PickCube-v1"]
        obs = torch.zeros(3, 42)
        # env0: true is_grasped bit on -> grasped, regardless of geometry fields
        obs[0, 18] = 1.0
        obs[0, 36:39] = torch.tensor([0.30, 0.0, 0.0]); obs[0, 31] = 0.02
        # env1: bit off -> not grasped, even if heuristic geometry would say yes
        obs[1, 18] = 0.0
        obs[1, 36:39] = torch.tensor([0.01, 0.0, 0.0]); obs[1, 31] = 0.10
        # env2: thresholded bit still off
        obs[2, 18] = 0.49
        g = det(obs)
        assert bool(g[0]) is True
        assert bool(g[1]) is False
        assert bool(g[2]) is False

    def test_stackcube_detector_is_force_free(self):
        # StackCube still uses the positional heuristic, so identical obs ->
        # identical result regardless of any (absent) force fields.
        det = PositionalGraspDetector(
            tcp_to_obj_indices=(0, 1, 2), obj_z_index=3,
            near_threshold=0.05, lift_threshold=0.035,
        )
        obs = torch.tensor([[0.0, 0.0, 0.0, 0.10]])   # at object, lifted
        assert bool(det(obs)[0]) is True

    def test_obs_bit_reader_thresholds(self):
        det = ObsBitGraspReader(obs_index=2, threshold=0.5)
        obs = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
        ])
        g = det(obs)
        assert torch.equal(g, torch.tensor([False, False, True]))
