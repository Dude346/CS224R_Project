"""
hiro/tests/test_replay_buffer.py
---------------------------------
Unit tests for hiro/replay_buffer.py.  Pure PyTorch, no env / GPU needed.

Run locally:
    uv run pytest hiro/tests/test_replay_buffer.py -v
"""
from __future__ import annotations

import pytest
import torch

from hiro.replay_buffer import (
    HighLevelBuffer,
    HighLevelSample,
    LowLevelBuffer,
    LowLevelSample,
)

OBS, SG, ACT = 42, 6, 8


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _ll_step(num_envs, tag: float):
    """Build one low-level transition for all envs, tagged so fields stay tied.

    obs is all `tag`; subgoal = obs + 100 elementwise.  After sampling we can
    assert subgoal == obs + 100, i.e. the per-field rows were never torn apart.
    """
    obs = torch.full((num_envs, OBS), tag)
    subgoal = obs[:, :SG] + 100.0
    action = torch.full((num_envs, ACT), tag)
    reward = torch.full((num_envs,), tag)
    next_obs = obs + 1.0
    next_subgoal = subgoal + 1.0
    done = torch.zeros(num_envs)
    return obs, subgoal, action, reward, next_obs, next_subgoal, done


# ---------------------------------------------------------------------------
# LowLevelBuffer
# ---------------------------------------------------------------------------

class TestLowLevelBuffer:
    def test_capacity_rounds_to_per_env(self):
        b = LowLevelBuffer(100, num_envs=8, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        assert b.per_env_capacity == 12          # 100 // 8
        assert b.capacity == 96                  # 12 * 8

    def test_too_small_capacity_raises(self):
        with pytest.raises(ValueError):
            LowLevelBuffer(4, num_envs=8, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)

    def test_len_grows_with_adds(self):
        E = 4
        b = LowLevelBuffer(40, num_envs=E, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        assert len(b) == 0
        for k in range(3):
            b.add(*_ll_step(E, float(k)))
        assert len(b) == 3 * E

    def test_sample_shapes(self):
        E = 4
        b = LowLevelBuffer(400, num_envs=E, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        for k in range(5):
            b.add(*_ll_step(E, float(k)))
        s = b.sample(32)
        assert isinstance(s, LowLevelSample)
        assert s.obs.shape == (32, OBS)
        assert s.subgoal.shape == (32, SG)
        assert s.action.shape == (32, ACT)
        assert s.reward.shape == (32,)
        assert s.next_obs.shape == (32, OBS)
        assert s.next_subgoal.shape == (32, SG)
        assert s.done.shape == (32,)

    def test_sample_preserves_field_alignment(self):
        # subgoal == obs[:SG] + 100 and next_obs == obs + 1 must hold for every sample.
        E = 4
        b = LowLevelBuffer(400, num_envs=E, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        for k in range(5):
            b.add(*_ll_step(E, float(k)))
        s = b.sample(64)
        assert torch.allclose(s.subgoal, s.obs[:, :SG] + 100.0)
        assert torch.allclose(s.next_obs, s.obs + 1.0)
        assert torch.allclose(s.next_subgoal, s.subgoal + 1.0)

    def test_ring_wraps_and_caps_len(self):
        E = 2
        b = LowLevelBuffer(20, num_envs=E, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        assert b.per_env_capacity == 10
        for k in range(13):              # overflow by 3 per env
            b.add(*_ll_step(E, float(k)))
        assert b.full is True
        assert len(b) == 20              # capped at capacity
        assert b.pos == 3                # 13 % 10

    def test_empty_sample_raises(self):
        b = LowLevelBuffer(40, num_envs=4, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        with pytest.raises(RuntimeError):
            b.sample(8)

    def test_done_cast_to_float(self):
        E = 4
        b = LowLevelBuffer(40, num_envs=E, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT)
        obs, sg, act, rew, nobs, nsg, _ = _ll_step(E, 1.0)
        done_bool = torch.ones(E, dtype=torch.bool)
        b.add(obs, sg, act, rew, nobs, nsg, done_bool)
        s = b.sample(8)
        assert s.done.dtype == torch.float32
        assert torch.all(s.done == 1.0)


# ---------------------------------------------------------------------------
# HighLevelBuffer
# ---------------------------------------------------------------------------

def _hl_batch(M, c, tag_start: float):
    """M high-level transitions, tagged so s0 and g0 stay tied: g0 == s0[:SG] + 500."""
    tags = torch.arange(M, dtype=torch.float32) + tag_start
    s0 = tags.view(M, 1).expand(M, OBS).clone()
    g0 = s0[:, :SG] + 500.0
    reward_sum = tags.clone()
    s_c = s0 + 1.0
    done = torch.zeros(M)
    inter_obs = torch.zeros(M, c, OBS)
    inter_act = torch.zeros(M, c, ACT)
    inter_len = torch.full((M,), c, dtype=torch.long)
    return s0, g0, reward_sum, s_c, done, inter_obs, inter_act, inter_len


class TestHighLevelBuffer:
    def test_len_grows_with_variable_batches(self):
        c = 10
        b = HighLevelBuffer(100, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=c)
        assert len(b) == 0
        b.add_batch(*_hl_batch(3, c, 0.0))   # 3 envs flushed
        assert len(b) == 3
        b.add_batch(*_hl_batch(1, c, 10.0))  # 1 env flushed
        assert len(b) == 4

    def test_empty_batch_is_noop(self):
        c = 10
        b = HighLevelBuffer(100, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=c)
        empty = _hl_batch(0, c, 0.0)
        b.add_batch(*empty)
        assert len(b) == 0

    def test_sample_shapes(self):
        c = 10
        b = HighLevelBuffer(100, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=c)
        b.add_batch(*_hl_batch(8, c, 0.0))
        s = b.sample(16)
        assert isinstance(s, HighLevelSample)
        assert s.s0.shape == (16, OBS)
        assert s.g0.shape == (16, SG)
        assert s.reward_sum.shape == (16,)
        assert s.s_c.shape == (16, OBS)
        assert s.done.shape == (16,)
        assert s.inter_obs.shape == (16, c, OBS)
        assert s.inter_act.shape == (16, c, ACT)
        assert s.inter_len.shape == (16,)
        assert s.inter_len.dtype == torch.long

    def test_wrap_around_preserves_alignment(self):
        # Small ring; force a wrap that splits a batch across the seam.
        c = 4
        b = HighLevelBuffer(5, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=c)
        b.add_batch(*_hl_batch(3, c, 0.0))   # pos -> 3
        b.add_batch(*_hl_batch(4, c, 10.0))  # 3..5 then wrap 0..2; pos -> 2, full
        assert b.full is True
        assert len(b) == 5
        assert b.pos == 2
        # Every stored entry must still satisfy g0 == s0[:SG] + 500 (no torn writes).
        s = b.sample(128)
        assert torch.allclose(s.g0, s.s0[:, :SG] + 500.0)
        # reward_sum was tagged equal to s0's value too.
        assert torch.allclose(s.reward_sum, s.s0[:, 0])

    def test_batch_exceeding_capacity_raises(self):
        c = 4
        b = HighLevelBuffer(5, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=c)
        with pytest.raises(ValueError):
            b.add_batch(*_hl_batch(6, c, 0.0))

    def test_variable_intermediate_length_stored(self):
        c = 10
        b = HighLevelBuffer(100, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=c)
        s0, g0, rs, s_c, done, io, ia, _ = _hl_batch(4, c, 0.0)
        inter_len = torch.tensor([10, 3, 7, 1], dtype=torch.long)  # early-terminated segments
        b.add_batch(s0, g0, rs, s_c, done, io, ia, inter_len)
        # Pull enough samples to see all four; the set of lengths must be a subset.
        s = b.sample(256)
        seen = set(s.inter_len.tolist())
        assert seen.issubset({10, 3, 7, 1})

    def test_empty_sample_raises(self):
        b = HighLevelBuffer(10, obs_dim=OBS, subgoal_dim=SG, action_dim=ACT, c=4)
        with pytest.raises(RuntimeError):
            b.sample(4)
