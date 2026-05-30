"""
hiro/networks.py
----------------
The neural-network building blocks for HIRO.  Both the high-level (manager)
and low-level (worker) policies are ordinary SAC actors/critics; they differ
ONLY in their input/output dimensions, so the same two classes
(`SquashedGaussianActor`, `TwinCritic`) are reused for both levels.

HIRO uses four networks total:

    worker actor   : [obs, subgoal]          -> action        (action_dim)
    worker critic  : [obs, subgoal], action  -> scalar Q       (twin)
    manager actor  : obs                      -> subgoal        (subgoal_dim)
    manager critic : obs, subgoal             -> scalar Q       (twin)

KEY ASYMMETRY (easy to get wrong): the manager's *action* IS the subgoal, so
the manager critic takes (obs, subgoal) -> Q, NOT (obs, subgoal, action).  We
express both critics with the same `TwinCritic(x_dim, a_dim)` class:

    worker critic  : x = [obs, subgoal] (x_dim = obs+subgoal),  a = action  (a_dim = action_dim)
    manager critic : x = obs            (x_dim = obs),           a = subgoal (a_dim = subgoal_dim)

Faithfulness
------------
The architecture and SAC math mirror ManiSkill's upstream baseline
(examples/baselines/sac/sac.py) exactly so HIRO's per-level SAC is identical
to the flat baseline we compare against:
  - 3 hidden layers of 256 units, ReLU, default PyTorch Linear init
  - log_std squashed via tanh into [LOG_STD_MIN, LOG_STD_MAX] = [-5, 2]
  - tanh-squashed Gaussian with action rescaling (action_scale / action_bias)
  - the log-prob correction  log_prob -= log(action_scale*(1 - tanh(x)^2) + 1e-6)

What lives elsewhere
--------------------
`log_alpha` (SAC's auto-temperature) and `target_entropy` are NOT defined here.
Upstream keeps `log_alpha` as a bare optimizer leaf tensor (zeros(1),
requires_grad=True), not a network submodule.  We follow that convention and
create one per level inside the agent (Step 5), so this module stays a pure
collection of nn.Modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import torch
import torch.nn as nn

# Squashing bounds for the policy log-std, copied verbatim from upstream sac.py
# (SpinUp / Denis Yarats convention).
LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0

ScaleLike = Union[float, Sequence[float], torch.Tensor]


def _mlp_trunk(in_dim: int, hidden_dim: int, n_hidden: int) -> nn.Sequential:
    """A stack of `n_hidden` (Linear -> ReLU) blocks ending in ReLU.

    For n_hidden=3, hidden_dim=256 this reproduces the upstream backbone:
        Linear(in, 256) ReLU  Linear(256,256) ReLU  Linear(256,256) ReLU
    """
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(nn.ReLU())
        last = hidden_dim
    return nn.Sequential(*layers)


def _as_dim_buffer(value: ScaleLike, dim: int, name: str) -> torch.Tensor:
    """Broadcast a scalar or length-`dim` value to a float32 tensor of shape (dim,)."""
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(dim).clone()
    elif t.shape != (dim,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({dim},), got shape {tuple(t.shape)}"
        )
    return t


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """A single Q-network: (x, a) -> scalar Q.

    Mirrors upstream `SoftQNetwork`: concatenate the conditioning vector `x`
    and the "action" `a`, then a 3x256 ReLU MLP to a scalar.

    For the worker:  x = [obs, subgoal], a = action.
    For the manager: x = obs,            a = subgoal.
    """

    def __init__(self, x_dim: int, a_dim: int, hidden_dim: int = 256, n_hidden: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_trunk(x_dim + a_dim, hidden_dim, n_hidden),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, a], dim=-1))


class TwinCritic(nn.Module):
    """Two independent Q-networks for SAC's clipped double-Q trick.

    Used identically by both levels (only the dims differ).
    """

    def __init__(self, x_dim: int, a_dim: int, hidden_dim: int = 256, n_hidden: int = 3):
        super().__init__()
        self.q1 = Critic(x_dim, a_dim, hidden_dim, n_hidden)
        self.q2 = Critic(x_dim, a_dim, hidden_dim, n_hidden)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x, a), self.q2(x, a)

    def q_min(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """min(Q1, Q2) — the conservative estimate SAC uses for targets/actor loss."""
        q1, q2 = self.forward(x, a)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Actor (tanh-squashed Gaussian) — shared by worker and manager
# ---------------------------------------------------------------------------

class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy, identical to upstream `Actor`.

    Parameterized by (in_dim, out_dim) plus the output bounds [low, high], which
    set `action_scale = (high-low)/2` and `action_bias = (high+low)/2`:

      - Worker: in_dim = obs+subgoal, out_dim = action_dim, low/high from the
        env action space (joint control is normalized to [-1, 1]).
      - Manager: in_dim = obs, out_dim = subgoal_dim, low/high = +/- subgoal_scale
        (a displacement bound, in the units of the subgoal space).

    `low`/`high` may be scalars or per-dimension length-`out_dim` vectors.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        low: ScaleLike = -1.0,
        high: ScaleLike = 1.0,
        hidden_dim: int = 256,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.backbone = _mlp_trunk(in_dim, hidden_dim, n_hidden)
        self.fc_mean = nn.Linear(hidden_dim, out_dim)
        self.fc_logstd = nn.Linear(hidden_dim, out_dim)

        low_t = _as_dim_buffer(low, out_dim, "low")
        high_t = _as_dim_buffer(high, out_dim, "high")
        if torch.any(high_t <= low_t):
            raise ValueError(f"require high > low elementwise; got low={low_t}, high={high_t}")
        # Registered buffers => moved by .to(device) and saved in state_dict.
        self.register_buffer("action_scale", (high_t - low_t) / 2.0)
        self.register_buffer("action_bias", (high_t + low_t) / 2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) of the pre-squash Gaussian."""
        h = self.backbone(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action via the reparameterization trick.

        Returns (action, log_prob, squashed_mean):
          - action       : sampled, tanh-squashed, rescaled to [low, high]
          - log_prob      : (..., 1) log-likelihood with the tanh Jacobian correction
          - squashed_mean : the deterministic (greedy) action
        """
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                       # reparameterized sample
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing the action bound (change of variables for tanh squashing).
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, squashed_mean

    def get_eval_action(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic (greedy) action: tanh(mean) rescaled to [low, high]."""
        h = self.backbone(x)
        mean = self.fc_mean(h)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def action_log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of a GIVEN action under the current policy at state x.

        Needed by HIRO's off-policy correction, which re-scores already-taken
        primitive actions under relabeled subgoals.  This inverts the tanh
        squashing (atanh) to recover the pre-squash sample, then evaluates the
        Gaussian density with the same tanh-Jacobian correction used in
        `get_action`, so the two are consistent.
        """
        mean, log_std = self(x)
        std = log_std.exp()
        # Invert action = tanh(x_t) * scale + bias  ->  y = tanh(x_t) in (-1, 1).
        y = (action - self.action_bias) / self.action_scale
        y = y.clamp(-1 + 1e-6, 1 - 1e-6)        # guard atanh against +/-inf
        x_t = torch.atanh(y)
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return log_prob.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# Convenience builder: wires the four nets with the correct dims in one place
# so the manager/worker asymmetry is documented exactly once.
# ---------------------------------------------------------------------------

@dataclass
class HIRONetworks:
    worker_actor: SquashedGaussianActor
    worker_critic: TwinCritic
    manager_actor: SquashedGaussianActor
    manager_critic: TwinCritic

    def to(self, device) -> "HIRONetworks":
        self.worker_actor.to(device)
        self.worker_critic.to(device)
        self.manager_actor.to(device)
        self.manager_critic.to(device)
        return self


def build_hiro_networks(
    obs_dim: int,
    subgoal_dim: int,
    action_dim: int,
    action_low: ScaleLike = -1.0,
    action_high: ScaleLike = 1.0,
    subgoal_scale: ScaleLike = 0.5,
    hidden_dim: int = 256,
    n_hidden: int = 3,
) -> HIRONetworks:
    """Construct the four HIRO networks with the correct (asymmetric) dims.

    Parameters
    ----------
    obs_dim, subgoal_dim, action_dim : int
        Dimensions of the flat observation, the subgoal vector, and the
        primitive action.
    action_low, action_high : scalar or per-dim
        Bounds of the env action space (joint control => [-1, 1]).
    subgoal_scale : scalar or per-dim
        Max |displacement| the manager may request; sets manager bounds to
        [-subgoal_scale, +subgoal_scale].
    """
    worker_actor = SquashedGaussianActor(
        in_dim=obs_dim + subgoal_dim, out_dim=action_dim,
        low=action_low, high=action_high, hidden_dim=hidden_dim, n_hidden=n_hidden,
    )
    worker_critic = TwinCritic(
        x_dim=obs_dim + subgoal_dim, a_dim=action_dim,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
    )

    scale_t = _as_dim_buffer(subgoal_scale, subgoal_dim, "subgoal_scale")
    manager_actor = SquashedGaussianActor(
        in_dim=obs_dim, out_dim=subgoal_dim,
        low=-scale_t, high=scale_t, hidden_dim=hidden_dim, n_hidden=n_hidden,
    )
    manager_critic = TwinCritic(
        x_dim=obs_dim, a_dim=subgoal_dim,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
    )
    return HIRONetworks(worker_actor, worker_critic, manager_actor, manager_critic)
