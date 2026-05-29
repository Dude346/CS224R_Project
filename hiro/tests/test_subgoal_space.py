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
    HIRO_SUBGOAL_SPACES,
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
        assert HIRO_SUBGOAL_SPACES["StackCube-v1"].dim == 9

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
