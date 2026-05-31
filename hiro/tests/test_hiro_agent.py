"""
hiro/tests/test_hiro_agent.py
------------------------------
Unit tests for hiro/hiro_agent.py.  Pure PyTorch on CPU, no env / GPU needed.

These do NOT test that HIRO *learns* (that needs the env + trainer); they test
that the wiring is correct: shapes, bounds, the off-policy-correction logic,
that updates run without NaNs, and that gradients actually move the right
parameters and target networks.

Run locally:
    uv run pytest hiro/tests/test_hiro_agent.py -v
"""
from __future__ import annotations

import copy

import pytest
import torch

from hiro.hiro_agent import HIROAgent, HIROConfig
from hiro.replay_buffer import HighLevelSample, LowLevelSample
from hiro.subgoal_space import HIRO_SUBGOAL_SPACES, OBJECT_CENTRIC_SUBGOAL_SPACES

OBS, SG, ACT = 42, 6, 8          # PickCube-v1 HIRO subgoal space
SCALE = 0.5


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make_agent(c: int = 5, num_candidates: int = 10) -> HIROAgent:
    sp = HIRO_SUBGOAL_SPACES["PickCube-v1"]   # obs_dim=42, dim=6
    cfg = HIROConfig(action_dim=ACT, c=c, subgoal_scale=SCALE,
                     num_candidates=num_candidates, device="cpu")
    return HIROAgent(sp, cfg)


def make_low(B: int) -> LowLevelSample:
    return LowLevelSample(
        obs=torch.randn(B, OBS),
        subgoal=(torch.rand(B, SG) * 2 - 1) * 0.4,
        action=torch.rand(B, ACT) * 2 - 1,
        reward=torch.randn(B),
        next_obs=torch.randn(B, OBS),
        next_subgoal=(torch.rand(B, SG) * 2 - 1) * 0.4,
        done=torch.zeros(B),
    )


def make_high(B: int, c: int, inter_len=None) -> HighLevelSample:
    if inter_len is None:
        inter_len = torch.full((B,), c, dtype=torch.long)
    return HighLevelSample(
        s0=torch.randn(B, OBS),
        g0=(torch.rand(B, SG) * 2 - 1) * 0.4,   # within +/- scale
        reward_sum=torch.randn(B),
        s_c=torch.randn(B, OBS),
        done=torch.zeros(B),
        inter_obs=torch.randn(B, c, OBS),
        inter_act=torch.rand(B, c, ACT) * 2 - 1,
        inter_len=inter_len,
    )


# ---------------------------------------------------------------------------
# Construction / action selection
# ---------------------------------------------------------------------------

class TestActionSelection:
    def test_select_action_shape_and_bounds(self):
        ag = make_agent()
        obs = torch.randn(16, OBS)
        sg = torch.zeros(16, SG)
        a = ag.select_action(obs, sg)
        assert a.shape == (16, ACT)
        assert torch.all(a.abs() <= 1.0)

    def test_select_action_deterministic(self):
        ag = make_agent()
        obs, sg = torch.randn(16, OBS), torch.zeros(16, SG)
        a1 = ag.select_action(obs, sg, deterministic=True)
        a2 = ag.select_action(obs, sg, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_select_subgoal_shape_and_bounds(self):
        ag = make_agent()
        g = ag.select_subgoal(torch.randn(16, OBS))
        assert g.shape == (16, SG)
        assert torch.all(g.abs() <= SCALE + 1e-6)


class TestWorkerRewardSelection:
    def test_object_centric_pickcube_rewards_reaching_pregrasp(self):
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(action_dim=ACT, c=5, subgoal_scale=SCALE, device="cpu")
        ag = HIROAgent(sp, cfg)
        ag.grasp_detector = lambda obs: obs[..., 18] > 0.5

        # tcp_to_obj = obs[36:39]; not grasped (obs[18]=0). Moving the hand
        # closer to the cube must be rewarded by the task-potential path.
        s = torch.zeros(1, OBS)
        s[:, 36] = 0.20
        sn = torch.zeros(1, OBS)
        sn[:, 36] = 0.10
        g = torch.zeros(1, SG)

        r = ag.worker_reward(s, g, sn)
        assert r.shape == (1,)
        assert r.item() > 0.0

    def test_object_centric_pickcube_lifting_beats_holding(self):
        """Agent-level guard for the bug that caused the plateau: once grasped,
        moving the cube toward the goal must beat holding it static."""
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(action_dim=ACT, c=5, subgoal_scale=SCALE, device="cpu")
        ag = HIROAgent(sp, cfg)
        ag.grasp_detector = lambda obs: obs[..., 18] > 0.5

        # grasped (obs[18]=1), gripper on cube (tcp_to_obj=0), cube 0.2 from goal.
        s = torch.zeros(1, OBS)
        s[:, 18] = 1.0
        s[:, 39] = 0.20            # obj_to_goal = obs[39:42]
        g = torch.zeros(1, SG)

        sn_hold = s.clone()
        sn_lift = s.clone()
        sn_lift[:, 39] = 0.10      # cube halved its distance to the goal
        r_hold = ag.worker_reward(s, g, sn_hold)
        r_lift = ag.worker_reward(s, g, sn_lift)
        assert r_lift.item() > r_hold.item()


class TestLatentInterface:
    def test_latent_object_centric_interface_has_requested_dim(self):
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(
            action_dim=ACT,
            c=5,
            subgoal_scale=SCALE,
            latent_subgoal_dim=4,
            latent_pretrain_steps=5,
            latent_pretrain_batch_size=32,
            device="cpu",
        )
        ag = HIROAgent(sp, cfg)
        assert ag.latent_enabled is True
        assert ag.subgoal_dim == 4

    def test_latent_transition_roundtrip_shape(self):
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(
            action_dim=ACT,
            c=5,
            subgoal_scale=SCALE,
            latent_subgoal_dim=4,
            latent_pretrain_steps=5,
            latent_pretrain_batch_size=32,
            device="cpu",
        )
        ag = HIROAgent(sp, cfg)
        obs = torch.randn(3, OBS)
        z = ag.select_subgoal(obs)
        next_obs = torch.randn(3, OBS)
        z_next = ag.subgoal_transition(obs, z, next_obs)
        assert z.shape == (3, 4)
        assert z_next.shape == (3, 4)

    def test_latent_worker_reward_decodes_geometric_goal(self):
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(
            action_dim=ACT,
            c=5,
            subgoal_scale=SCALE,
            latent_subgoal_dim=4,
            latent_pretrain_steps=5,
            latent_pretrain_batch_size=32,
            device="cpu",
        )
        ag = HIROAgent(sp, cfg)
        ag.grasp_detector = lambda obs: obs[..., 18] > 0.5
        s = torch.zeros(2, OBS)
        z = ag.select_subgoal(s)
        s_next = torch.zeros(2, OBS)
        r = ag.worker_reward(s, z, s_next)
        assert r.shape == (2,)

    def test_latent_off_policy_correction_returns_latent_shape(self):
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        latent_dim = 4
        cfg = HIROConfig(
            action_dim=ACT,
            c=5,
            subgoal_scale=SCALE,
            latent_subgoal_dim=latent_dim,
            latent_pretrain_steps=5,
            latent_pretrain_batch_size=32,
            device="cpu",
        )
        ag = HIROAgent(sp, cfg)
        batch = HighLevelSample(
            s0=torch.randn(8, OBS),
            g0=(torch.rand(8, latent_dim) * 2 - 1) * 0.9,
            reward_sum=torch.randn(8),
            s_c=torch.randn(8, OBS),
            done=torch.zeros(8),
            inter_obs=torch.randn(8, 5, OBS),
            inter_act=torch.rand(8, 5, ACT) * 2 - 1,
            inter_len=torch.full((8,), 5, dtype=torch.long),
        )
        g = ag.off_policy_correct(batch)
        assert g.shape == (8, latent_dim)
        assert torch.all(torch.isfinite(g))


# ---------------------------------------------------------------------------
# Off-policy correction
# ---------------------------------------------------------------------------

class TestOffPolicyCorrection:
    def test_shape_and_bounds(self):
        ag = make_agent(c=5)
        g = ag.off_policy_correct(make_high(32, c=5))
        assert g.shape == (32, SG)
        assert torch.all(torch.isfinite(g))
        assert torch.all(g.abs() <= SCALE + 1e-6)   # clipped to subgoal range

    def test_zero_length_returns_original_g0(self):
        # With no valid intermediate steps, every candidate scores 0, so argmax
        # picks candidate index 0 == the original g0.
        ag = make_agent(c=5)
        batch = make_high(16, c=5, inter_len=torch.zeros(16, dtype=torch.long))
        g = ag.off_policy_correct(batch)
        assert torch.allclose(g, batch.g0)

    def test_respects_partial_length_mask(self):
        # Should run and stay finite when segments have heterogeneous lengths.
        ag = make_agent(c=6)
        lens = torch.tensor([6, 1, 3, 6, 2, 0, 4, 5], dtype=torch.long)
        batch = make_high(8, c=6, inter_len=lens)
        g = ag.off_policy_correct(batch)
        assert g.shape == (8, SG)
        assert torch.all(torch.isfinite(g))

    def test_is_a_real_candidate(self):
        # The returned subgoal must be one of the generated candidates (it is
        # gathered from `cands`); here we at least confirm determinism w/ seed.
        ag = make_agent(c=4)
        batch = make_high(8, c=4)
        torch.manual_seed(123)
        g1 = ag.off_policy_correct(batch)
        torch.manual_seed(123)
        g2 = ag.off_policy_correct(batch)
        assert torch.allclose(g1, g2)


# ---------------------------------------------------------------------------
# SAC updates
# ---------------------------------------------------------------------------

class TestUpdates:
    def test_update_low_returns_finite_losses(self):
        ag = make_agent()
        out = ag.update_low(make_low(64))
        for k in ("q_loss", "actor_loss", "alpha_loss", "alpha"):
            assert k in out and torch.isfinite(torch.tensor(out[k]))

    def test_update_high_returns_finite_losses(self):
        ag = make_agent(c=5)
        out = ag.update_high(make_high(64, c=5))
        for k in ("q_loss", "actor_loss", "alpha_loss", "alpha"):
            assert k in out and torch.isfinite(torch.tensor(out[k]))

    def test_update_low_changes_worker_params(self):
        ag = make_agent()
        before = ag.nets.worker_actor.fc_mean.weight.detach().clone()
        ag.update_low(make_low(64))
        after = ag.nets.worker_actor.fc_mean.weight.detach()
        assert not torch.allclose(before, after)

    def test_update_high_changes_manager_params(self):
        ag = make_agent(c=5)
        before = ag.nets.manager_actor.fc_mean.weight.detach().clone()
        ag.update_high(make_high(64, c=5))
        after = ag.nets.manager_actor.fc_mean.weight.detach()
        assert not torch.allclose(before, after)

    def test_update_low_does_not_touch_manager(self):
        ag = make_agent()
        before = ag.nets.manager_actor.fc_mean.weight.detach().clone()
        ag.update_low(make_low(64))
        after = ag.nets.manager_actor.fc_mean.weight.detach()
        assert torch.allclose(before, after)

    def test_many_updates_stay_finite(self):
        ag = make_agent(c=5)
        for _ in range(10):
            lo = ag.update_low(make_low(64))
            hi = ag.update_high(make_high(64, c=5))
            assert all(torch.isfinite(torch.tensor(v)) for v in lo.values())
            assert all(torch.isfinite(torch.tensor(v)) for v in hi.values())


# ---------------------------------------------------------------------------
# Target networks
# ---------------------------------------------------------------------------

class TestTargets:
    def test_targets_initialized_equal_to_online(self):
        ag = make_agent()
        for p, tp in zip(ag.nets.worker_critic.parameters(),
                         ag.worker_critic_target.parameters()):
            assert torch.allclose(p, tp)
        for p, tp in zip(ag.nets.manager_critic.parameters(),
                         ag.manager_critic_target.parameters()):
            assert torch.allclose(p, tp)

    def test_targets_are_frozen(self):
        ag = make_agent()
        assert all(not p.requires_grad for p in ag.worker_critic_target.parameters())
        assert all(not p.requires_grad for p in ag.manager_critic_target.parameters())

    def test_soft_update_moves_target(self):
        ag = make_agent()
        before = [tp.detach().clone() for tp in ag.worker_critic_target.parameters()]
        ag.update_low(make_low(64))
        after = list(ag.worker_critic_target.parameters())
        # At least one target parameter should have moved toward the online net.
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_state_dict_roundtrip(self):
        ag = make_agent()
        for _ in range(3):
            ag.update_low(make_low(32))
        sd = copy.deepcopy(ag.state_dict())

        ag2 = make_agent()
        ag2.load_state_dict(sd)
        # Worker actor params must match exactly after load.
        for p, p2 in zip(ag.nets.worker_actor.parameters(),
                         ag2.nets.worker_actor.parameters()):
            assert torch.allclose(p, p2)
        assert torch.allclose(ag.log_alpha_low, ag2.log_alpha_low)
