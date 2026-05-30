"""
hiro/tests/test_networks.py
----------------------------
Unit tests for hiro/networks.py.  Pure PyTorch, no env / GPU needed.

Run locally:
    uv run pytest hiro/tests/test_networks.py -v
"""
from __future__ import annotations

import pytest
import torch

from hiro.networks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    Critic,
    HIRONetworks,
    SquashedGaussianActor,
    TwinCritic,
    build_hiro_networks,
)

# Canonical dims for our two tasks (joint control => action_dim = 8).
PICK = dict(obs_dim=42, subgoal_dim=6, action_dim=8)   # HIRO or objcentric (both 6)
STACK = dict(obs_dim=48, subgoal_dim=9, action_dim=8)  # StackCube HIRO (9-D)

B = 16  # batch size used throughout


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# SquashedGaussianActor
# ---------------------------------------------------------------------------

class TestActor:
    def test_forward_shapes(self):
        actor = SquashedGaussianActor(in_dim=42 + 6, out_dim=8)
        mean, log_std = actor(torch.randn(B, 48))
        assert mean.shape == (B, 8)
        assert log_std.shape == (B, 8)

    def test_log_std_in_bounds(self):
        actor = SquashedGaussianActor(in_dim=20, out_dim=4)
        # Drive fc_logstd to extreme values; the tanh rescale must still clamp.
        with torch.no_grad():
            actor.fc_logstd.weight.mul_(1000.0)
            actor.fc_logstd.bias.fill_(1000.0)
        _, log_std = actor(torch.randn(B, 20))
        assert torch.all(log_std <= LOG_STD_MAX + 1e-5)
        assert torch.all(log_std >= LOG_STD_MIN - 1e-5)

    def test_get_action_shapes(self):
        actor = SquashedGaussianActor(in_dim=48, out_dim=8)
        action, log_prob, mean = actor.get_action(torch.randn(B, 48))
        assert action.shape == (B, 8)
        assert log_prob.shape == (B, 1)
        assert mean.shape == (B, 8)

    def test_action_within_bounds_default(self):
        # Default bounds [-1, 1] => action_scale=1, bias=0 => action in (-1, 1).
        actor = SquashedGaussianActor(in_dim=48, out_dim=8)
        action, _, mean = actor.get_action(torch.randn(B, 8 * 0 + 48))
        assert torch.all(action.abs() < 1.0)
        assert torch.all(mean.abs() < 1.0)

    def test_action_within_custom_scalar_bounds(self):
        # Manager-style symmetric bound: subgoal in (-0.5, 0.5).
        actor = SquashedGaussianActor(in_dim=42, out_dim=6, low=-0.5, high=0.5)
        action, _, _ = actor.get_action(torch.randn(B, 42))
        assert torch.all(action.abs() < 0.5)
        assert torch.allclose(actor.action_scale, torch.full((6,), 0.5))
        assert torch.allclose(actor.action_bias, torch.zeros(6))

    def test_action_within_per_dim_bounds(self):
        low = [-0.1, -0.2, -0.3]
        high = [0.1, 0.2, 0.3]
        actor = SquashedGaussianActor(in_dim=10, out_dim=3, low=low, high=high)
        action, _, _ = actor.get_action(torch.randn(B, 10))
        hi = torch.tensor(high)
        assert torch.all(action.abs() < hi)  # strict: tanh never reaches the bound

    def test_asymmetric_bounds_scale_and_bias(self):
        actor = SquashedGaussianActor(in_dim=10, out_dim=2, low=0.0, high=2.0)
        # scale=(2-0)/2=1, bias=(2+0)/2=1 => action in (0, 2).
        assert torch.allclose(actor.action_scale, torch.ones(2))
        assert torch.allclose(actor.action_bias, torch.ones(2))
        action, _, _ = actor.get_action(torch.randn(B, 10))
        assert torch.all(action > 0.0) and torch.all(action < 2.0)

    def test_log_prob_finite(self):
        actor = SquashedGaussianActor(in_dim=48, out_dim=8)
        _, log_prob, _ = actor.get_action(torch.randn(B, 48))
        assert torch.all(torch.isfinite(log_prob))

    def test_get_action_is_stochastic(self):
        actor = SquashedGaussianActor(in_dim=48, out_dim=8)
        x = torch.randn(B, 48)
        a1, _, _ = actor.get_action(x)
        a2, _, _ = actor.get_action(x)
        assert not torch.allclose(a1, a2)

    def test_eval_action_is_deterministic(self):
        actor = SquashedGaussianActor(in_dim=48, out_dim=8)
        x = torch.randn(B, 48)
        a1 = actor.get_eval_action(x)
        a2 = actor.get_eval_action(x)
        assert torch.allclose(a1, a2)

    def test_eval_action_within_bounds(self):
        actor = SquashedGaussianActor(in_dim=42, out_dim=6, low=-0.5, high=0.5)
        a = actor.get_eval_action(torch.randn(B, 42))
        assert torch.all(a.abs() < 0.5)

    def test_reparam_gradients_flow(self):
        # The reparameterization trick must let gradients reach actor params.
        actor = SquashedGaussianActor(in_dim=20, out_dim=4)
        action, log_prob, _ = actor.get_action(torch.randn(B, 20))
        (action.sum() + log_prob.sum()).backward()
        grads = [p.grad for p in actor.parameters() if p.requires_grad]
        assert any(g is not None and torch.any(g != 0) for g in grads)

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            SquashedGaussianActor(in_dim=10, out_dim=3, low=1.0, high=1.0)  # high<=low
        with pytest.raises(ValueError):
            SquashedGaussianActor(in_dim=10, out_dim=3, low=[0.0, 0.0], high=[1.0])  # wrong len


class TestActionLogProb:
    def test_consistent_with_get_action(self):
        # action_log_prob of a sampled action must match the log_prob get_action
        # returned for it (it inverts the same tanh squashing).
        actor = SquashedGaussianActor(in_dim=20, out_dim=4)
        x = torch.randn(B, 20)
        action, log_prob, _ = actor.get_action(x)
        recomputed = actor.action_log_prob(x, action)
        assert recomputed.shape == (B, 1)
        assert torch.allclose(log_prob, recomputed, atol=1e-3)

    def test_consistent_with_custom_scale(self):
        # Manager-style bounds must invert correctly too.
        actor = SquashedGaussianActor(in_dim=42, out_dim=6, low=-0.5, high=0.5)
        x = torch.randn(B, 42)
        action, log_prob, _ = actor.get_action(x)
        recomputed = actor.action_log_prob(x, action)
        assert torch.allclose(log_prob, recomputed, atol=1e-3)

    def test_handles_boundary_actions(self):
        # Actions exactly at the bound must not produce NaN/inf (atanh guard).
        actor = SquashedGaussianActor(in_dim=10, out_dim=3, low=-1.0, high=1.0)
        x = torch.randn(B, 10)
        action = torch.ones(B, 3)  # at the upper bound
        lp = actor.action_log_prob(x, action)
        assert torch.all(torch.isfinite(lp))


# ---------------------------------------------------------------------------
# Critic / TwinCritic
# ---------------------------------------------------------------------------

class TestCritic:
    def test_single_critic_shape(self):
        c = Critic(x_dim=48, a_dim=8)
        q = c(torch.randn(B, 48), torch.randn(B, 8))
        assert q.shape == (B, 1)

    def test_twin_returns_two_q(self):
        tc = TwinCritic(x_dim=48, a_dim=8)
        q1, q2 = tc(torch.randn(B, 48), torch.randn(B, 8))
        assert q1.shape == (B, 1) and q2.shape == (B, 1)

    def test_twin_q_are_independent(self):
        # Independent random init => the two heads should not be identical.
        tc = TwinCritic(x_dim=48, a_dim=8)
        x, a = torch.randn(B, 48), torch.randn(B, 8)
        q1, q2 = tc(x, a)
        assert not torch.allclose(q1, q2)

    def test_q_min_matches_elementwise_min(self):
        tc = TwinCritic(x_dim=48, a_dim=8)
        x, a = torch.randn(B, 48), torch.randn(B, 8)
        q1, q2 = tc(x, a)
        assert torch.allclose(tc.q_min(x, a), torch.min(q1, q2))

    def test_manager_critic_dims(self):
        # Manager critic: x = obs (no action concat beyond the subgoal).
        tc = TwinCritic(x_dim=48, a_dim=9)  # StackCube obs, 9-D subgoal
        q1, q2 = tc(torch.randn(B, 48), torch.randn(B, 9))
        assert q1.shape == (B, 1) and q2.shape == (B, 1)


# ---------------------------------------------------------------------------
# build_hiro_networks — the asymmetry wiring
# ---------------------------------------------------------------------------

class TestBuilder:
    @pytest.mark.parametrize("cfg", [PICK, STACK])
    def test_end_to_end_shapes(self, cfg):
        obs_dim, sg_dim, act_dim = cfg["obs_dim"], cfg["subgoal_dim"], cfg["action_dim"]
        nets = build_hiro_networks(obs_dim, sg_dim, act_dim, subgoal_scale=0.5)
        assert isinstance(nets, HIRONetworks)

        obs = torch.randn(B, obs_dim)
        subgoal = torch.randn(B, sg_dim)
        action = torch.randn(B, act_dim)

        # Worker actor: [obs, subgoal] -> action
        wa_in = torch.cat([obs, subgoal], dim=-1)
        a, lp, m = nets.worker_actor.get_action(wa_in)
        assert a.shape == (B, act_dim) and lp.shape == (B, 1) and m.shape == (B, act_dim)
        assert torch.all(a.abs() < 1.0)  # joint action bound

        # Worker critic: [obs, subgoal], action -> Q
        wq1, wq2 = nets.worker_critic(wa_in, action)
        assert wq1.shape == (B, 1) and wq2.shape == (B, 1)

        # Manager actor: obs -> subgoal (bounded by subgoal_scale)
        g, glp, gm = nets.manager_actor.get_action(obs)
        assert g.shape == (B, sg_dim) and glp.shape == (B, 1)
        assert torch.all(g.abs() < 0.5)

        # Manager critic: obs, subgoal -> Q
        mq1, mq2 = nets.manager_critic(obs, subgoal)
        assert mq1.shape == (B, 1) and mq2.shape == (B, 1)

    def test_manager_actor_input_is_obs_only(self):
        nets = build_hiro_networks(**PICK)
        assert nets.manager_actor.in_dim == PICK["obs_dim"]
        assert nets.manager_actor.out_dim == PICK["subgoal_dim"]

    def test_worker_actor_input_is_obs_plus_subgoal(self):
        nets = build_hiro_networks(**PICK)
        assert nets.worker_actor.in_dim == PICK["obs_dim"] + PICK["subgoal_dim"]
        assert nets.worker_actor.out_dim == PICK["action_dim"]

    def test_per_dim_subgoal_scale(self):
        # Different displacement bound per subgoal dim (e.g. tcp vs object).
        scale = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
        nets = build_hiro_networks(42, 6, 8, subgoal_scale=scale)
        g, _, _ = nets.manager_actor.get_action(torch.randn(B, 42))
        assert torch.all(g.abs() < torch.tensor(scale))

    def test_to_device_cpu_noop(self):
        # .to('cpu') should run without error and keep buffers on cpu.
        nets = build_hiro_networks(**PICK).to(torch.device("cpu"))
        assert nets.manager_actor.action_scale.device.type == "cpu"
