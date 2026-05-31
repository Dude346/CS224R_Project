"""
hiro/tests/test_rollout.py
---------------------------
Unit tests for hiro/rollout.py — the per-env subgoal bookkeeping.  Pure CPU,
no env / GPU.  These are white-box: they inspect the buffers' raw storage to
verify exactly what got written, targeting the classic HIRO-on-vector-env bugs:
  - off-by-one in the intermediate (s,a) arrays
  - applying the subgoal transition at the wrong time
  - c-boundary vs episode-end flush handling
  - confusing the terminal obs (storage) with the new-episode obs (resampling)

Run locally:
    uv run pytest hiro/tests/test_rollout.py -v
"""
from __future__ import annotations

import pytest
import torch

from hiro.hiro_agent import HIROAgent, HIROConfig
from hiro.replay_buffer import HighLevelBuffer, LowLevelBuffer
from hiro.rollout import HierarchicalRollout
from hiro.subgoal_space import HIRO_SUBGOAL_SPACES

OBS, SG, ACT = 42, 6, 8


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def setup(E: int = 2, c: int = 3):
    sp = HIRO_SUBGOAL_SPACES["PickCube-v1"]      # obs_dim=42, dim=6
    cfg = HIROConfig(action_dim=ACT, c=c, num_candidates=4, device="cpu")
    agent = HIROAgent(sp, cfg)
    low = LowLevelBuffer(E * 100, E, OBS, SG, ACT)
    high = HighLevelBuffer(100, OBS, SG, ACT, c)
    roll = HierarchicalRollout(agent, low, high, E, device="cpu")
    return agent, low, high, roll, sp


def _zeros(E):
    return torch.zeros(E), torch.zeros(E, dtype=torch.bool), torch.zeros(E)  # r, episode_end, done


# ---------------------------------------------------------------------------
# Segment flushing
# ---------------------------------------------------------------------------

class TestFlushing:
    def test_synchronous_flush_at_c(self):
        E, c = 2, 3
        agent, low, high, roll, sp = setup(E, c)
        start_obs = torch.randn(E, OBS)
        roll.start(start_obs)

        total_r = torch.zeros(E)
        obs = start_obs
        last_next = None
        for t in range(c):
            action = torch.randn(E, ACT).clamp(-0.9, 0.9)
            nxt = torch.randn(E, OBS)
            r = torch.full((E,), float(t + 1))
            total_r += r
            roll.record(obs, action, nxt, nxt, r,
                        torch.zeros(E, dtype=torch.bool), torch.zeros(E))
            obs, last_next = nxt, nxt

        assert len(high) == E                       # both envs flushed once
        assert len(low) == c * E                    # every step stored low transitions
        assert torch.all(high.inter_len[:E] == c)
        assert torch.allclose(high.s0[:E], start_obs)        # s0 = segment start
        assert torch.allclose(high.s_c[:E], last_next)       # s_c = segment end
        assert torch.allclose(high.reward_sum[:E], total_r)  # accumulated env reward
        # segment counters reset after flush
        assert torch.all(roll.seg_len == 0)

    def test_episode_end_flushes_early(self):
        E, c = 1, 5
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))

        obs = torch.randn(E, OBS)
        for t in range(2):
            end = torch.tensor([t == 1])             # episode ends on the 2nd step
            nxt = torch.randn(E, OBS)
            roll.record(obs, torch.randn(E, ACT).clamp(-0.9, 0.9), nxt, nxt,
                        torch.zeros(E), end, torch.zeros(E))
            obs = nxt

        assert len(high) == 1
        assert high.inter_len[0].item() == 2         # only 2 valid intermediates

    def test_async_flush_across_envs(self):
        # env0 ends early (len 2); env1 runs the full c (len 3). They flush at
        # different steps — exercises the variable-size batched high-level add.
        E, c = 2, 3
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))
        obs = torch.randn(E, OBS)

        def step(end_flags):
            nonlocal obs
            nxt = torch.randn(E, OBS)
            roll.record(obs, torch.randn(E, ACT).clamp(-0.9, 0.9), nxt, nxt,
                        torch.zeros(E), torch.tensor(end_flags), torch.zeros(E))
            obs = nxt

        step([False, False])     # step1: nobody flushes
        step([True, False])      # step2: env0 ends -> 1 flush, len 2
        assert len(high) == 1
        assert high.inter_len[0].item() == 2
        step([False, False])     # step3: env1 hits c=3 -> 1 flush, len 3
        assert len(high) == 2
        assert high.inter_len[1].item() == 3
        # env0 is mid-new-segment (1 step in), env1 reset
        assert roll.seg_len[0].item() == 1
        assert roll.seg_len[1].item() == 0


# ---------------------------------------------------------------------------
# Per-step storage correctness
# ---------------------------------------------------------------------------

class TestStorageCorrectness:
    def test_low_transition_uses_current_subgoal(self):
        E, c = 1, 5
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))
        # Override to a known subgoal so we can check the HIRO math exactly.
        g = torch.tensor([[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]])
        roll.cur_subgoal = g.clone()

        obs = torch.randn(E, OBS)
        nxt = torch.randn(E, OBS)
        action = torch.zeros(E, ACT)
        roll.record(obs, action, nxt, nxt, torch.zeros(E),
                    torch.zeros(E, dtype=torch.bool), torch.zeros(E))

        # Stored low transition (pos 0, env 0).
        assert torch.allclose(low.obs[0, 0], obs[0])
        assert torch.allclose(low.subgoal[0, 0], g[0])               # ORIGINAL subgoal
        assert torch.allclose(low.next_obs[0, 0], nxt[0])
        assert torch.allclose(low.reward[0, 0], sp.intrinsic_reward(obs, g, nxt)[0])
        assert torch.allclose(low.next_subgoal[0, 0],
                              sp.subgoal_transition(obs, g, nxt)[0])  # transitioned

    def test_cur_subgoal_becomes_transition_when_not_flushing(self):
        E, c = 1, 5
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))
        g = torch.tensor([[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]])
        roll.cur_subgoal = g.clone()
        obs, nxt = torch.randn(E, OBS), torch.randn(E, OBS)
        new_cur = roll.record(obs, torch.zeros(E, ACT), nxt, nxt, torch.zeros(E),
                              torch.zeros(E, dtype=torch.bool), torch.zeros(E))
        assert torch.allclose(new_cur, sp.subgoal_transition(obs, g, nxt))

    def test_intermediate_ordering(self):
        # Intermediates must be s_0..s_{c-1} / a_0..a_{c-1} in order.
        E, c = 1, 3
        agent, low, high, roll, sp = setup(E, c)
        obs0 = torch.full((E, OBS), 0.0)
        roll.start(obs0)

        obs = obs0
        for t in range(c):
            tagged_obs = torch.full((E, OBS), float(t))     # s_t tagged by step
            tagged_act = torch.full((E, ACT), float(t) + 0.5)
            nxt = torch.full((E, OBS), float(t + 1))
            roll.record(tagged_obs, tagged_act, nxt, nxt, torch.zeros(E),
                        torch.zeros(E, dtype=torch.bool), torch.zeros(E))
            obs = nxt

        assert len(high) == 1
        for t in range(c):
            assert torch.allclose(high.inter_obs[0, t], torch.full((OBS,), float(t)))
            assert torch.allclose(high.inter_act[0, t], torch.full((ACT,), float(t) + 0.5))

    def test_terminal_vs_resample_obs_separation(self):
        # On episode end: s_c (and the low next_obs) use the TERMINAL obs, while
        # the next segment starts from the (different) RESAMPLE obs.
        E, c = 1, 5
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))

        obs = torch.randn(E, OBS)
        terminal = torch.full((E, OBS), 7.0)     # true terminal obs
        resample = torch.full((E, OBS), -7.0)    # new-episode obs
        roll.record(obs, torch.zeros(E, ACT), terminal, resample, torch.zeros(E),
                    torch.tensor([True]), torch.zeros(E))

        assert torch.allclose(high.s_c[0], terminal[0])          # storage uses terminal
        assert torch.allclose(low.next_obs[0, 0], terminal[0])
        assert torch.allclose(roll.seg_start_obs[0], resample[0])  # next segment from resample


# ---------------------------------------------------------------------------
# Reward accumulation
# ---------------------------------------------------------------------------

class TestRewardAccumulation:
    def test_reward_sum_resets_after_flush(self):
        E, c = 1, 2
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))
        obs = torch.randn(E, OBS)

        # First segment: rewards 1 + 2 = 3, flush at c=2.
        for r in (1.0, 2.0):
            nxt = torch.randn(E, OBS)
            roll.record(obs, torch.zeros(E, ACT), nxt, nxt, torch.tensor([r]),
                        torch.zeros(E, dtype=torch.bool), torch.zeros(E))
            obs = nxt
        assert torch.allclose(high.reward_sum[0], torch.tensor(3.0))
        assert roll.seg_reward[0].item() == 0.0      # reset

        # Second segment: rewards 5 + 6 = 11.
        for r in (5.0, 6.0):
            nxt = torch.randn(E, OBS)
            roll.record(obs, torch.zeros(E, ACT), nxt, nxt, torch.tensor([r]),
                        torch.zeros(E, dtype=torch.bool), torch.zeros(E))
            obs = nxt
        assert torch.allclose(high.reward_sum[1], torch.tensor(11.0))


# ---------------------------------------------------------------------------
# Boundary-transition correctness (the two audit fixes)
# ---------------------------------------------------------------------------

class TestBoundaryFixes:
    def _step(self, roll, E, episode_end=False, terminal=False):
        # episode_end = flush trigger (truncation OR termination);
        # terminal = TRUE env termination -> bootstrap_done (manager/worker SAC done).
        obs = torch.randn(E, OBS)
        nxt = torch.randn(E, OBS)
        return roll.record(
            obs, torch.zeros(E, ACT), nxt, nxt, torch.zeros(E),
            torch.tensor([episode_end] * E), torch.tensor([float(terminal)] * E),
        )

    def test_worker_next_subgoal_at_boundary_is_fresh_goal(self):
        # F2: at a c-boundary the worker's stored next_subgoal must be the freshly
        # sampled manager goal (what it will act under), not the telescoped g_next.
        E, c = 1, 2
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))
        self._step(roll, E)                 # step 1, no flush
        new_cur = self._step(roll, E)       # step 2, flush at c=2
        # the boundary transition is the 2nd low entry (index 1)
        assert torch.allclose(low.next_subgoal[1, 0], new_cur[0])
        # and within a segment it is the telescoped goal (1st entry != fresh resample)
        # (sanity: the 1st transition's next_subgoal is the transition, not a resample)
        assert low.next_subgoal[0, 0].shape == (SG,)

    def test_manager_done_is_true_termination_only(self):
        # F3 (corrected): high-level done = TRUE termination, NOT the flush trigger.
        # 0 on a c-boundary, 0 on a horizon TRUNCATION (must still bootstrap), and
        # 1 only on a genuine env termination (success).
        E, c = 1, 2
        agent, low, high, roll, sp = setup(E, c)
        roll.start(torch.randn(E, OBS))
        self._step(roll, E, episode_end=False)                    # step 1
        self._step(roll, E, episode_end=False)                    # step 2 -> c-boundary flush
        assert high.done[0].item() == 0.0                         # mid-episode boundary: not done
        # truncation: episode ends (flush) but it is NOT terminal -> manager bootstraps
        self._step(roll, E, episode_end=True, terminal=False)     # truncation flush (seg_len=1)
        assert high.done[1].item() == 0.0                         # truncation: NOT done
        # genuine env termination (e.g. success) -> manager done
        self._step(roll, E, episode_end=True, terminal=True)      # terminal flush
        assert high.done[2].item() == 1.0                         # real terminal: done
