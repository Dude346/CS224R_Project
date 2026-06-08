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
from hiro.subgoal_space import GRASP_DETECTORS, HIRO_SUBGOAL_SPACES, OBJECT_CENTRIC_SUBGOAL_SPACES

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


def make_place_agent(beta: float = 2.0, gamma_high: float = 0.8) -> HIROAgent:
    """Agent with Phase-2 manager place-PBRS enabled (PickCube cube->goal dims)."""
    sp = HIRO_SUBGOAL_SPACES["PickCube-v1"]
    cfg = HIROConfig(action_dim=ACT, c=5, subgoal_scale=SCALE, device="cpu",
                     gamma_high=gamma_high, place_pbrs_beta=beta,
                     place_residual_dims=(39, 40, 41))
    return HIROAgent(sp, cfg)


# ---------------------------------------------------------------------------
# Phase 2: manager place-PBRS shaping
# ---------------------------------------------------------------------------

class TestManagerPlacePBRS:
    def test_shaping_matches_formula(self):
        ag = make_place_agent(beta=2.0, gamma_high=0.8)
        B = 5
        s0, s_c = torch.randn(B, OBS), torch.randn(B, OBS)
        done = torch.tensor([0., 0., 1., 0., 1.])
        dims = torch.tensor([39, 40, 41])
        phi0 = -2.0 * s0[:, dims].norm(dim=-1)
        phic = -2.0 * s_c[:, dims].norm(dim=-1)
        expected = 0.8 * phic * (1.0 - done) - phi0
        got = ag._manager_place_shaping(s0, s_c, done)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_done_masks_next_potential(self):
        # When done=1, the gamma*Phi(s_c) term must vanish (Phi(terminal)=0) so
        # shaping reduces to -Phi(s0); policy-invariance for the episodic MDP.
        ag = make_place_agent(beta=1.5)
        s0, s_c = torch.randn(3, OBS), torch.randn(3, OBS)
        done = torch.ones(3)
        dims = torch.tensor([39, 40, 41])
        expected = 1.5 * s0[:, dims].norm(dim=-1)  # -Phi(s0) = +beta*||s0[dims]||
        assert torch.allclose(ag._manager_place_shaping(s0, s_c, done), expected, atol=1e-5)

    def test_disabled_by_default(self):
        # Default agent has no place dims => shaping path is off.
        ag = make_agent()
        assert ag._place_dims is None

    def test_update_high_runs_with_pbrs(self):
        ag = make_place_agent(beta=1.0)
        m = ag.update_high(make_high(8, c=5))
        for k in ("q_loss", "q_mean", "entropy"):
            assert k in m and torch.isfinite(torch.tensor(m[k]))


# ---------------------------------------------------------------------------
# Phase D: pre-grasp reach-to-cube PBRS bootstrap (worker_reward)
# ---------------------------------------------------------------------------

class TestReachBootstrap:
    K, G = 2.0, 0.8
    REACH = (36, 37, 38)   # PickCube tcp_to_obj_pos

    def _agent(self, weight: float) -> HIROAgent:
        sp = HIRO_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(action_dim=ACT, c=5, subgoal_scale=SCALE, device="cpu",
                         gamma_low=self.G, reach_pbrs_weight=weight, reach_dims=self.REACH)
        ag = HIROAgent(sp, cfg)
        ag.grasp_detector = GRASP_DETECTORS["PickCube-v1"]   # worker_reward needs it
        return ag

    def _phi(self, s):
        d = s[:, list(self.REACH)].norm(dim=-1)
        return self.K * (1.0 - torch.tanh(5.0 * d))

    def test_reach_shaping_matches_formula(self):
        # worker_reward does NOT touch the actor, so (weight=K) - (weight=0) isolates
        # the reach shaping = gamma_low*Phi(s') - Phi(s) exactly.
        a0, ak = self._agent(0.0), self._agent(self.K)
        B = 4
        s, s_next = torch.randn(B, OBS), torch.randn(B, OBS)
        g = (torch.rand(B, SG) * 2 - 1) * 0.1
        delta = ak.worker_reward(s, g, s_next) - a0.worker_reward(s, g, s_next)
        expected = self.G * self._phi(s_next) - self._phi(s)
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_approaching_positive_hovering_is_tax(self):
        a0, ak = self._agent(0.0), self._agent(self.K)
        g = torch.zeros(1, SG)

        def shaping(d_s, d_n):
            s, sn = torch.zeros(1, OBS), torch.zeros(1, OBS)
            s[0, list(self.REACH)] = torch.tensor([d_s, 0.0, 0.0])
            sn[0, list(self.REACH)] = torch.tensor([d_n, 0.0, 0.0])
            return (ak.worker_reward(s, g, sn) - a0.worker_reward(s, g, sn)).item()

        assert shaping(0.30, 0.02) > 0      # approaching the cube is rewarded
        assert shaping(0.01, 0.01) < 0      # hovering at the cube nets only the (gamma-1) tax => non-farmable

    def test_disabled_by_default(self):
        assert make_agent()._reach_dims is None


def test_load_manager_only_loads_manager():
    # Transfer: load_manager copies ONLY the manager (+ its temperature), leaving the
    # worker at its own (fresh, different) init.
    src, dst = make_agent(), make_agent()
    dst.load_manager(src.state_dict())
    for ps, pd in zip(src.nets.manager_actor.parameters(), dst.nets.manager_actor.parameters()):
        assert torch.allclose(ps, pd)
    assert torch.allclose(src.log_alpha_high, dst.log_alpha_high)
    # worker must NOT have been overwritten (dst keeps its own random worker)
    worker_differs = any(
        not torch.allclose(ps, pd)
        for ps, pd in zip(src.nets.worker_actor.parameters(), dst.nets.worker_actor.parameters())
    )
    assert worker_differs


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

    def test_apply_low_her_relabels_subgoal_and_reward(self):
        sp = OBJECT_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
        cfg = HIROConfig(action_dim=ACT, c=5, subgoal_scale=SCALE, device="cpu")
        ag = HIROAgent(sp, cfg)
        ag.grasp_detector = lambda obs: obs[..., 18] > 0.5

        obs = torch.zeros(2, OBS)
        next_obs = torch.zeros(2, OBS)
        obs[0, 36] = 0.20
        next_obs[0, 36] = 0.10
        # Make sample 0 have a future achieved object-centric state.
        future_obs = torch.zeros(2, OBS)
        future_obs[0, 36] = 0.05
        future_obs[0, 39] = 0.10
        batch = LowLevelSample(
            obs=obs,
            subgoal=torch.zeros(2, SG),
            action=torch.zeros(2, ACT),
            reward=torch.zeros(2),
            next_obs=next_obs,
            next_subgoal=torch.zeros(2, SG),
            done=torch.zeros(2),
        )
        mask = torch.tensor([True, False])
        relabeled = ag.apply_low_her(batch, future_obs, mask)
        expected_g0 = sp.project(future_obs[0:1]) - sp.project(obs[0:1])
        assert torch.allclose(relabeled.subgoal[0:1], expected_g0)
        assert torch.allclose(relabeled.subgoal[1], batch.subgoal[1])
        expected_next = ag.subgoal_transition(obs[0:1], expected_g0, next_obs[0:1])
        assert torch.allclose(relabeled.next_subgoal[0:1], expected_next)
        expected_reward = ag.worker_reward(obs[0:1], expected_g0, next_obs[0:1])
        assert torch.allclose(relabeled.reward[0:1], expected_reward)
        assert relabeled.reward[0].item() != ag.worker_reward(obs[0:1], batch.subgoal[0:1], next_obs[0:1]).item()


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

class TestPBRSGradientFlow:
    """End-to-end check: the PBRS shaping must actually reach the worker's loss.

    Build a small buffer of F->T grasp-transition rewards (using the actual
    agent.worker_reward path) and confirm: (a) those rewards include the PBRS
    bump, (b) update_low runs end-to-end on them without NaNs.
    """

    def test_pbrs_bump_in_buffer(self):
        from hiro.subgoal_space import GRASP_DETECTORS
        ag = make_agent()
        ag.cfg.pbrs_alpha = 0.5
        ag.cfg.gamma_low = 0.95
        ag.grasp_detector = GRASP_DETECTORS["PickCube-v1"]

        B = 32
        # Construct B transitions that are a true F -> T grasp, via the env grasp
        # bit obs[18] (ObsBitGraspReader): not grasped at s, grasped at s_next.
        s = torch.zeros(B, OBS)
        s_next = torch.zeros(B, OBS)
        s[:, 18] = 0.0      # not grasped at s
        s_next[:, 18] = 1.0  # grasped at s_next
        g = torch.zeros(B, SG)

        r = ag.worker_reward(s, g, s_next)
        # PBRS bump dominates a zero-residual transition: r ~= +γα = +0.475
        assert (r > 0.4).all() and (r < 0.5).all(), (
            f"PBRS grasp bump not visible in worker reward: r={r[0].item():.4f}, expected ~0.475"
        )

    def test_update_low_runs_on_pbrs_rewards(self):
        ag = make_agent()
        # Hand the worker a batch with realistic PBRS-shaped rewards (~+0.5 at grasp).
        batch = make_low(64)
        batch.reward.fill_(0.5)
        out = ag.update_low(batch)
        for k in ("q_loss", "actor_loss", "alpha_loss"):
            assert torch.isfinite(torch.tensor(out[k])), f"non-finite {k}: {out[k]}"


class TestHybridReward:
    """worker_extrinsic_weight blends env reward into the worker reward."""

    def test_w_zero_is_pure_intrinsic(self):
        ag = make_agent()
        ag.cfg.worker_extrinsic_weight = 0.0
        s, g, sn = torch.randn(8, OBS), torch.zeros(8, SG), torch.randn(8, OBS)
        env_r = torch.full((8,), 5.0)
        # With no detector, intrinsic = plain; env_reward must be ignored at w=0.
        r = ag.worker_reward(s, g, sn, env_r)
        assert torch.allclose(r, ag.sp.intrinsic_reward(s, g, sn))

    def test_hybrid_blend_convex(self):
        from hiro.subgoal_space import GRASP_DETECTORS
        ag = make_agent()
        ag.grasp_detector = GRASP_DETECTORS["PickCube-v1"]
        ag.cfg.worker_extrinsic_weight = 0.5
        s, g, sn = torch.randn(8, OBS), torch.zeros(8, SG), torch.randn(8, OBS)
        env_r = torch.full((8,), 1.0)

        # Recover the intrinsic part by querying at w=0, then check the blend.
        ag.cfg.worker_extrinsic_weight = 0.0
        intrinsic = ag.worker_reward(s, g, sn, env_r)
        ag.cfg.worker_extrinsic_weight = 0.5
        blended = ag.worker_reward(s, g, sn, env_r)
        expected = 0.5 * intrinsic + 0.5 * env_r
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_env_reward_none_falls_back_to_intrinsic(self):
        ag = make_agent()
        ag.cfg.worker_extrinsic_weight = 0.5
        s, g, sn = torch.randn(4, OBS), torch.zeros(4, SG), torch.randn(4, OBS)
        # No env_reward passed => no blend, returns intrinsic.
        r = ag.worker_reward(s, g, sn, None)
        assert torch.allclose(r, ag.sp.intrinsic_reward(s, g, sn))


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


# ---------------------------------------------------------------------------
# Body-independent manager (Option-B cross-robot transfer). The manager reads
# only obs[manager_input_dims] (9-D [tcp,cube,goal]) so its weights transfer
# across robots with different obs sizes. Default (no dims) = full obs, unchanged.
# ---------------------------------------------------------------------------

from hiro.subgoal_space import (  # noqa: E402
    CUBE_CENTRIC_SUBGOAL_SPACES,
    FETCH_CUBE_CENTRIC_SUBGOAL_SPACES,
    MANAGER_INPUT_DIMS,
    FETCH_MANAGER_INPUT_DIMS,
)


def _bi_agent(sp, mgr_dims):
    cfg = HIROConfig(action_dim=8, manager_input_dims=tuple(mgr_dims),
                     use_off_policy_correction=False, device="cpu")
    return HIROAgent(sp, cfg)


def test_bi_manager_slices_and_selects():
    sp = CUBE_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]     # obs_dim 42, sg 3
    ag = _bi_agent(sp, MANAGER_INPUT_DIMS["PickCube-v1"])  # 9-D manager input
    assert ag._mgr_dims == list(MANAGER_INPUT_DIMS["PickCube-v1"])
    # If the manager net were sized to the full 42-D obs, feeding the 9-D slice
    # would raise a shape error -- so a clean call proves the net is 9-D.
    g = ag.select_subgoal(torch.randn(4, 42), deterministic=True)
    assert g.shape == (4, sp.dim)


def test_bi_manager_transfers_across_obs_sizes():
    ag_panda = _bi_agent(CUBE_CENTRIC_SUBGOAL_SPACES["PickCube-v1"],
                          MANAGER_INPUT_DIMS["PickCube-v1"])        # obs 42
    ag_fetch = _bi_agent(FETCH_CUBE_CENTRIC_SUBGOAL_SPACES["PickCube-v1"],
                         FETCH_MANAGER_INPUT_DIMS["PickCube-v1"])   # obs 54
    # The whole point: a panda-trained body-independent manager loads onto the
    # fetch agent without a shape mismatch (both managers have a 9-D input).
    ag_fetch.load_manager(ag_panda.state_dict())
    g = ag_fetch.select_subgoal(torch.randn(2, 54), deterministic=True)
    assert g.shape == (2, 3)


def test_full_obs_manager_is_default():
    sp = CUBE_CENTRIC_SUBGOAL_SPACES["PickCube-v1"]
    ag = HIROAgent(sp, HIROConfig(action_dim=8, use_off_policy_correction=False, device="cpu"))
    assert ag._mgr_dims is None
    g = ag.select_subgoal(torch.randn(3, 42), deterministic=True)
    assert g.shape == (3, sp.dim)
