"""
hiro/hiro_agent.py
------------------
The HIROAgent ties the networks (Step 3) and buffers (Step 4) together: action
selection for both levels, the two SAC updates, and the off-policy correction
that is HIRO's signature contribution.

Both levels are ordinary SAC agents, so a single private `_sac_update` mirrors
the upstream ManiSkill SAC math (clipped double-Q targets, entropy-regularized
actor loss, autotuned temperature, soft target updates).  Worker and manager
differ only in:
  - worker  : state = [obs, subgoal],  action = primitive action,  reward = r_lo
  - manager : state =  obs,            action = subgoal,            reward = sum env reward
and in the manager's case the stored action (subgoal) is first relabeled by the
off-policy correction.

Off-policy correction (Nachum et al. 2018)
------------------------------------------
A stored high-level transition (s0, g0, R, s_c) was produced under an *old*
worker policy.  We relabel g0 with the candidate subgoal g~ that makes the
actually-taken primitive actions a_0..a_{c-1} most likely under the *current*
worker policy.  Candidates: the original g0, the achieved displacement
proj(s_c) - proj(s0), and (K-2) Gaussian samples around that displacement.  For
a candidate g~, the induced subgoal at intermediate step i follows the fixed
goal-transition telescoping g~_i = proj(s0) + g~ - proj(s_i); we score
sum_i log pi_worker(a_i | s_i, g~_i), mask steps past the segment length, and
pick the arg-max candidate.  Only the critic's input action is relabeled; the
manager actor update samples fresh subgoals as usual.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from hiro.networks import HIRONetworks, SubgoalLatentCodec, TwinCritic, build_hiro_networks
from hiro.replay_buffer import HighLevelSample, LowLevelSample
from hiro.subgoal_space import SubgoalSpace


@dataclass
class HIROConfig:
    action_dim: int
    c: int = 10
    subgoal_scale: float = 0.15         # |displacement| bound; scalar or per-dim
    action_low: float = -1.0
    action_high: float = 1.0
    gamma_low: float = 0.8
    gamma_high: float = 0.97            # longer-horizon manager for transport stability
    tau: float = 0.005                  # soft target update rate
    max_grad_norm: float = 10.0         # critic/actor grad clip; 0 disables
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    num_candidates: int = 10            # off-policy-correction candidates (incl. original + achieved)
    candidate_std_scale: float = 0.5    # Gaussian std = scale * subgoal_scale
    use_off_policy_correction: bool = True  # debug switch: False => manager trains on raw g0
    use_phased_reward: bool = True      # grasp-gate the object subgoal dims in the worker reward
    pbrs_alpha: float = 0.5             # PBRS grasp potential strength (Φ = α·is_grasped);
                                        # gives +γα at grasp, -α at drop, all policy-invariant
    # Hybrid worker reward: r = (1-w)*intrinsic + w*env_reward. w=0 => pure HIRO
    # (worker is task-agnostic, depends only on the subgoal). w>0 injects the env
    # reward (which contains place/grasp shaping) so the worker can make task
    # progress even under poor manager subgoals -- breaks the cold-start coupling.
    worker_extrinsic_weight: float = 0.0
    # Phase 2 (manager place-PBRS): ungated, policy-invariant potential added to the
    # MANAGER reward, Phi(s) = -beta*||cube-goal||. Shaping F = gamma_high*Phi(s_c)*(1-done)
    # - Phi(s0) rewards NET progress of the cube toward the goal regardless of grasp,
    # removing the drop-cliff from the manager's objective (the grasp-and-hold local
    # optimum). beta=0 disables. place_residual_dims = obs indices of the cube->goal
    # vector (set by the trainer from get_place_residual_dims(env_id)).
    place_pbrs_beta: float = 0.0
    place_residual_dims: tuple = ()
    # Phase D (reach bootstrap): potential-based pre-grasp shaping on the WORKER,
    # Phi_reach(s) = w * (1 - tanh(5*||tcp - cube||)), shaping = gamma_low*Phi(s') - Phi(s).
    # Object-anchored (rewards getting the gripper TO the cube -- the prerequisite
    # for grasping), so it cannot be hover-hacked the way an arbitrary TCP-target
    # term could. Potential-based => non-farmable (hovering at the cube nets the
    # small w*(gamma-1) tax, like the grasp PBRS). Naturally ~0 post-grasp because
    # the gripper stays on the cube (||tcp-cube||~=0 => Phi~=const). Lets a pure
    # intrinsic (w_env=0) worker bootstrap grasping with NO env reward, so the
    # manager-load-bearing test is clean. reach_dims = obs indices of tcp->object.
    reach_pbrs_weight: float = 0.0
    reach_dims: tuple = ()
    # Body-independent manager (cross-robot transfer): obs indices of the
    # object-centric features [tcp_xyz, cube_xyz, goal_xyz] the manager observes
    # INSTEAD of the full robot-specific obs. () => full obs (default; unchanged).
    # A manager trained on these body-independent dims transfers across robots with
    # different obs sizes (e.g. panda 42-D -> fetch 54-D); the worker stays full-obs.
    manager_input_dims: tuple = ()
    # Latent subgoal codec (latent_object_centric variant): >0 enables a learned
    # latent code over the geometric subgoal space (0 = disabled; default path).
    latent_subgoal_dim: int = 0
    latent_hidden_dim: int = 128
    latent_pretrain_steps: int = 1000
    latent_pretrain_batch_size: int = 256
    latent_pretrain_range: float = 2.0  # sample geometric subgoals over +/- range*scale (telescoped
                                        # residuals routinely exceed the manager's one-step bound)
    latent_lr: float = 1e-3
    low_her_ratio: float = 0.8          # fraction of worker batch relabeled with future achieved goals
    hidden_dim: int = 256
    n_hidden: int = 3
    device: str = "cpu"


class _LevelOptims:
    """Optimizers + autotuned temperature for one SAC level."""

    def __init__(self, actor, critic, log_alpha, policy_lr, q_lr, alpha_lr, target_entropy):
        self.actor = torch.optim.Adam(actor.parameters(), lr=policy_lr)
        self.critic = torch.optim.Adam(critic.parameters(), lr=q_lr)
        self.log_alpha = log_alpha
        self.alpha = torch.optim.Adam([log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy


class HIROAgent:
    def __init__(self, subgoal_space: SubgoalSpace, cfg: HIROConfig):
        self.sp = subgoal_space
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.c = cfg.c
        self.num_candidates = cfg.num_candidates

        obs_dim = subgoal_space.obs_dim
        self.base_subgoal_dim = subgoal_space.dim
        self.latent_enabled = cfg.latent_subgoal_dim > 0
        self.subgoal_dim = cfg.latent_subgoal_dim if self.latent_enabled else self.base_subgoal_dim
        sg_dim = self.subgoal_dim
        act_dim = cfg.action_dim

        # Body-independent manager: it observes only obs[manager_input_dims]
        # (object features), so its learned weights transfer across embodiments.
        # () => full obs (original behaviour). The worker always sees full obs.
        self._mgr_dims = list(cfg.manager_input_dims) if cfg.manager_input_dims else None
        manager_obs_dim = len(self._mgr_dims) if self._mgr_dims is not None else obs_dim

        base_scale = torch.as_tensor(cfg.subgoal_scale, dtype=torch.float32, device=self.device)
        if base_scale.ndim == 0:
            base_scale = base_scale.expand(self.base_subgoal_dim).clone()
        self.base_scale = base_scale

        # --- networks (online) ---
        self.nets: HIRONetworks = build_hiro_networks(
            obs_dim=obs_dim, subgoal_dim=sg_dim, action_dim=act_dim,
            action_low=cfg.action_low, action_high=cfg.action_high,
            subgoal_scale=1.0 if self.latent_enabled else cfg.subgoal_scale,
            hidden_dim=cfg.hidden_dim, n_hidden=cfg.n_hidden,
            manager_obs_dim=manager_obs_dim,
        ).to(self.device)

        self.latent_codec = None
        if self.latent_enabled:
            self.latent_codec = SubgoalLatentCodec(
                base_dim=self.base_subgoal_dim,
                latent_dim=sg_dim,
                latent_scale=1.0,
                hidden_dim=cfg.latent_hidden_dim,
                n_hidden=2,
            ).to(self.device)
            self._pretrain_latent_codec()
            for p in self.latent_codec.parameters():
                p.requires_grad_(False)

        # --- target critics (frozen copies) ---
        self.worker_critic_target = self._make_target(self.nets.worker_critic)
        self.manager_critic_target = self._make_target(self.nets.manager_critic)

        # --- autotuned temperatures (one log_alpha per level) ---
        self.log_alpha_low = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_high = torch.zeros(1, requires_grad=True, device=self.device)

        # Scale-corrected target entropy. The standard -dim heuristic assumes
        # actions in [-1, 1]; a tanh policy squashed to [-s, s] has its log-prob
        # shifted by -sum(log s) per the change-of-variables term, so the entropy
        # target must absorb +sum(log s) or the autotuned temperature blows up.
        # (Worker action_scale ~= 1 -> no change; manager scale << 1 -> big shift.)
        te_low = -float(act_dim) + self.nets.worker_actor.action_scale.log().sum().item()
        te_high = -float(sg_dim) + self.nets.manager_actor.action_scale.log().sum().item()
        self.low = _LevelOptims(
            self.nets.worker_actor, self.nets.worker_critic, self.log_alpha_low,
            cfg.policy_lr, cfg.q_lr, cfg.alpha_lr, target_entropy=te_low,
        )
        self.high = _LevelOptims(
            self.nets.manager_actor, self.nets.manager_critic, self.log_alpha_high,
            cfg.policy_lr, cfg.q_lr, cfg.alpha_lr, target_entropy=te_high,
        )

        # --- candidate noise for the off-policy correction (per-dim, on device) ---
        self.candidate_std = cfg.candidate_std_scale * self.base_scale  # (base_sg_dim,)

        # Optional grasp detector (set by the trainer from the env id). When present
        # AND the subgoal space has object dims, the worker uses the phase-aware
        # reward; otherwise it falls back to the plain intrinsic reward.
        self.grasp_detector = None

        # Phase 2: manager place-PBRS. Obs indices of the cube->goal vector; None
        # disables the shaping (also disabled when place_pbrs_beta <= 0).
        self._place_dims = (
            torch.tensor(cfg.place_residual_dims, dtype=torch.long, device=self.device)
            if cfg.place_residual_dims else None
        )
        # Phase D: pre-grasp reach bootstrap. Obs indices of the tcp->object vector.
        self._reach_dims = (
            torch.tensor(cfg.reach_dims, dtype=torch.long, device=self.device)
            if cfg.reach_dims else None
        )

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _make_target(self, critic: TwinCritic) -> TwinCritic:
        target = copy.deepcopy(critic)
        for p in target.parameters():
            p.requires_grad_(False)
        return target.to(self.device)

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    def _pretrain_latent_codec(self) -> None:
        if self.latent_codec is None or self.cfg.latent_pretrain_steps <= 0:
            return
        optim = torch.optim.Adam(self.latent_codec.parameters(), lr=self.cfg.latent_lr)
        B = self.cfg.latent_pretrain_batch_size
        rng = self.cfg.latent_pretrain_range
        latent_scale = self.latent_codec.latent_scale
        self.latent_codec.train()
        for _ in range(self.cfg.latent_pretrain_steps):
            # (1) reconstruction on geometric subgoals over a WIDER range than the
            #     manager's one-step bound, since telescoped residuals exceed it.
            g_base = (
                torch.rand(B, self.base_subgoal_dim, device=self.device) * 2.0 - 1.0
            ) * (self.base_scale * rng)
            recon = self.latent_codec.reconstruct(g_base)
            recon_loss = F.mse_loss(recon, g_base)
            # (2) latent cycle-consistency: the manager emits codes across the FULL
            #     [-latent_scale, latent_scale] cube, so `decode` must be a right
            #     inverse of `encode` there -- otherwise large parts of the manager's
            #     action space decode to extrapolated garbage subgoals.
            z = (torch.rand(B, self.subgoal_dim, device=self.device) * 2.0 - 1.0) * latent_scale
            z_cycle = self.latent_codec.encode(self.latent_codec.decode(z))
            cycle_loss = F.mse_loss(z_cycle, z)
            loss = recon_loss + cycle_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        self.latent_codec.eval()

    def encode_subgoal(self, g_base: torch.Tensor) -> torch.Tensor:
        if self.latent_codec is None:
            return g_base
        return self.latent_codec.encode(g_base)

    def decode_subgoal(self, g_interface: torch.Tensor) -> torch.Tensor:
        if self.latent_codec is None:
            return g_interface
        return self.latent_codec.decode(g_interface)

    # ------------------------------------------------------------------
    # subgoal-space delegates (so the trainer can call through the agent)
    # ------------------------------------------------------------------

    def subgoal_transition(self, s, g, s_next):
        g_base = self.decode_subgoal(g)
        g_next_base = self.sp.subgoal_transition(s, g_base, s_next)
        return self.encode_subgoal(g_next_base)

    def intrinsic_reward(self, s, g, s_next):
        g_base = self.decode_subgoal(g)
        return self.sp.intrinsic_reward(s, g_base, s_next)

    @torch.no_grad()
    def worker_reward(self, s, g, s_next, env_reward=None):
        """Worker reward used during rollout.

        Intrinsic part: phase-aware PBRS (the v5 design) when a grasp detector is
        set AND the subgoal space defines object dims AND pbrs_alpha > 0;
        otherwise the plain intrinsic reward (-||residual||).
            intrinsic = -||hand_residual||
                      + is_grasped(s) * (-||object_residual||)
                      + γ·α·is_grasped(s') − α·is_grasped(s)   (PBRS, policy-invariant)

        Hybrid blend (worker_extrinsic_weight = w > 0):
            r = (1 - w) * intrinsic + w * env_reward
        w=0 keeps the worker purely subgoal-driven (pure HIRO). w>0 lets the
        worker see the env's place/grasp shaping, breaking the manager-cold-start.
        """
        if (
            not self.cfg.use_phased_reward
            or self.grasp_detector is None
            or not self.sp.object_positions
            or self.cfg.pbrs_alpha <= 0.0
        ):
            intrinsic = self.intrinsic_reward(s, g, s_next)
        else:
            g_base = self.decode_subgoal(g)
            is_grasped_s = self.grasp_detector(s)
            is_grasped_s_next = self.grasp_detector(s_next)
            # Object-centric spaces use the farming-proof task-potential reward
            # (telescoping shaping that mirrors the env's reach/grasp/place reward);
            # all other spaces use the grasp-gated phased PBRS reward.
            if self.sp.reward_mode == "pickcube_task_potential":
                intrinsic = self.sp.task_potential_intrinsic_reward(
                    s, g_base, s_next, is_grasped_s, is_grasped_s_next,
                    reach_coef=1.0, grasp_coef=1.0, place_coef=1.0,
                    object_progress_coef=2.0, tanh_temp=5.0,
                )
            else:
                intrinsic = self.sp.phased_pbrs_intrinsic_reward(
                    s, g_base, s_next, is_grasped_s, is_grasped_s_next,
                    gamma=self.cfg.gamma_low, alpha=self.cfg.pbrs_alpha,
                )

        # Phase D: pre-grasp reach-to-cube PBRS bootstrap (object-anchored, non-farmable).
        #   Phi(s) = w_reach * (1 - tanh(5*||tcp-cube||));  shaping = gamma_low*Phi(s') - Phi(s)
        # Rewards approaching the cube; hovering at it nets only the w_reach*(gamma-1) tax;
        # ~0 once grasped (gripper stays on the cube). Added to the intrinsic so it also
        # bootstraps a pure (w_env=0) worker.
        if self.cfg.reach_pbrs_weight > 0.0 and self._reach_dims is not None:
            k = self.cfg.reach_pbrs_weight
            phi_s = k * (1.0 - torch.tanh(5.0 * torch.norm(s[..., self._reach_dims], dim=-1)))
            phi_next = k * (1.0 - torch.tanh(5.0 * torch.norm(s_next[..., self._reach_dims], dim=-1)))
            intrinsic = intrinsic + self.cfg.gamma_low * phi_next - phi_s

        w = self.cfg.worker_extrinsic_weight
        if w > 0.0 and env_reward is not None:
            return (1.0 - w) * intrinsic + w * env_reward
        return intrinsic

    # ------------------------------------------------------------------
    # action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs, subgoal, deterministic: bool = False):
        """Worker primitive action from (obs, subgoal)."""
        state = torch.cat([obs, subgoal], dim=-1)
        if deterministic:
            return self.nets.worker_actor.get_eval_action(state)
        action, _, _ = self.nets.worker_actor.get_action(state)
        return action

    def _mgr_in(self, obs):
        """Slice obs to the manager's (optionally body-independent) input.
        Identity when manager_input_dims is empty (full-obs manager, default)."""
        return obs[..., self._mgr_dims] if self._mgr_dims is not None else obs

    @torch.no_grad()
    def select_subgoal(self, obs, deterministic: bool = False):
        """Manager subgoal from obs."""
        obs_m = self._mgr_in(obs)
        if deterministic:
            return self.nets.manager_actor.get_eval_action(obs_m)
        subgoal, _, _ = self.nets.manager_actor.get_action(obs_m)
        return subgoal

    # ------------------------------------------------------------------
    # off-policy correction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def off_policy_correct(self, batch: HighLevelSample) -> torch.Tensor:
        """Return relabeled subgoals g~ of shape (B, subgoal_dim)."""
        sp = self.sp
        s0, s_c = batch.s0, batch.s_c
        inter_obs, inter_act, inter_len = batch.inter_obs, batch.inter_act, batch.inter_len
        B, c, obs_dim = inter_obs.shape
        act_dim = inter_act.shape[-1]
        K = self.num_candidates
        base_sg_dim = self.base_subgoal_dim
        iface_sg_dim = self.subgoal_dim

        proj_s0 = sp.project(s0)                  # (B, base_sg)
        proj_sc = sp.project(s_c)                 # (B, base_sg)
        diff = proj_sc - proj_s0                  # achieved displacement (B, base_sg)

        # Candidate subgoals: original, achieved, then Gaussian around achieved.
        cands = torch.empty(B, K, base_sg_dim, device=s0.device)
        if self.latent_enabled:
            cands[:, 0] = self.decode_subgoal(batch.g0)
        else:
            cands[:, 0] = batch.g0
        cands[:, 1] = diff
        if K > 2:
            noise = torch.randn(B, K - 2, base_sg_dim, device=s0.device) * self.candidate_std
            cands[:, 2:] = diff.unsqueeze(1) + noise
        # Clip to the GEOMETRIC subgoal range [-base_scale, base_scale] (per-dim).
        cands = torch.minimum(cands, self.base_scale)
        cands = torch.maximum(cands, -self.base_scale)

        # Induced subgoal at each intermediate step for each candidate:
        #   g~_{i,k} = proj(s0) + g~_k - proj(s_i)
        proj_inter = sp.project(inter_obs)        # (B, c, sg)
        induced = (
            proj_s0[:, None, None, :]
            + cands[:, :, None, :]
            - proj_inter[:, None, :, :]
        )                                          # (B, K, c, sg)

        if self.latent_enabled:
            latent_cands = self.encode_subgoal(cands.reshape(B * K, base_sg_dim)).reshape(B, K, iface_sg_dim)
            latent_induced = self.encode_subgoal(
                induced.reshape(B * K * c, base_sg_dim)
            ).reshape(B, K, c, iface_sg_dim)
        else:
            latent_cands = cands
            latent_induced = induced

        obs_exp = inter_obs[:, None, :, :].expand(B, K, c, obs_dim)   # (B,K,c,obs)
        state = torch.cat([obs_exp, latent_induced], dim=-1)          # (B,K,c,obs+sg)
        act_exp = inter_act[:, None, :, :].expand(B, K, c, act_dim)   # (B,K,c,act)

        logp = self.nets.worker_actor.action_log_prob(
            state.reshape(B * K * c, obs_dim + iface_sg_dim),
            act_exp.reshape(B * K * c, act_dim),
        ).reshape(B, K, c)                                            # (B,K,c)

        # Mask steps at or beyond the (possibly early-terminated) segment length.
        step_ids = torch.arange(c, device=s0.device)[None, :]        # (1,c)
        valid = (step_ids < inter_len[:, None]).float()              # (B,c)
        score = (logp * valid[:, None, :]).sum(dim=-1)               # (B,K)

        best = score.argmax(dim=1)                                   # (B,)
        return latent_cands[torch.arange(B, device=s0.device), best]  # (B, interface_sg)

    # ------------------------------------------------------------------
    # SAC updates
    # ------------------------------------------------------------------

    def _sac_update(self, *, actor, critic, critic_target, optims: _LevelOptims,
                    gamma: float, state, action, reward, next_state, done) -> dict:
        """One SAC gradient step for a single level. Shapes: reward/done (B,)."""
        alpha = optims.log_alpha.exp().item()
        reward = reward.unsqueeze(-1)               # (B,1)
        done = done.float().unsqueeze(-1)           # (B,1)

        # --- critic ---
        with torch.no_grad():
            next_action, next_logpi, _ = actor.get_action(next_state)
            q1_t, q2_t = critic_target(next_state, next_action)
            min_q_next = torch.min(q1_t, q2_t) - alpha * next_logpi   # (B,1)
            target_q = reward + (1.0 - done) * gamma * min_q_next     # (B,1)
        q1, q2 = critic(state, action)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        optims.critic.zero_grad()
        q_loss.backward()
        if self.cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.max_grad_norm)
        optims.critic.step()

        # --- actor ---
        pi, logpi, _ = actor.get_action(state)
        q1_pi, q2_pi = critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logpi - min_q_pi).mean()
        optims.actor.zero_grad()
        actor_loss.backward()
        if self.cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
        optims.actor.step()

        # --- temperature (autotune) ---
        with torch.no_grad():
            _, logpi_detached, _ = actor.get_action(state)
        alpha_loss = (-optims.log_alpha.exp() * (logpi_detached + optims.target_entropy)).mean()
        optims.alpha.zero_grad()
        alpha_loss.backward()
        optims.alpha.step()

        # --- soft target update ---
        self._soft_update(critic, critic_target)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": optims.log_alpha.exp().item(),
            # diagnostics: Q magnitude (divergence watch) + policy entropy (collapse watch)
            "q_mean": q1.mean().item(),
            "q_max": q1.detach().abs().max().item(),
            "entropy": (-logpi).mean().item(),
        }

    def update_low(self, batch: LowLevelSample) -> dict:
        """SAC update for the worker. State = [obs, subgoal]."""
        state = torch.cat([batch.obs, batch.subgoal], dim=-1)
        next_state = torch.cat([batch.next_obs, batch.next_subgoal], dim=-1)
        return self._sac_update(
            actor=self.nets.worker_actor,
            critic=self.nets.worker_critic,
            critic_target=self.worker_critic_target,
            optims=self.low,
            gamma=self.cfg.gamma_low,
            state=state, action=batch.action, reward=batch.reward,
            next_state=next_state, done=batch.done,
        )

    @torch.no_grad()
    def _manager_place_shaping(self, s0, s_c, done) -> torch.Tensor:
        """Phase 2 place-PBRS shaping for the manager reward (policy-invariant).

            Phi(s) = -beta * ||cube - goal||                       (= -beta*||s[place_dims]||)
            F      = gamma_high * Phi(s_c) * (1 - done) - Phi(s0)

        Ng et al. (1999): adding F = gamma*Phi(s') - Phi(s) to the reward leaves
        the optimal policy unchanged, so this only *re-shapes the gradient* toward
        moving the cube to the goal -- it does NOT introduce a reward the manager
        can farm. Phi(terminal)=0 is enforced via the (1-done) mask so the
        invariance holds for the episodic (success-terminating) MDP. Returns (B,).
        """
        dims = self._place_dims
        beta = self.cfg.place_pbrs_beta
        phi_s0 = -beta * torch.norm(s0[..., dims], dim=-1)     # (B,)
        phi_sc = -beta * torch.norm(s_c[..., dims], dim=-1)    # (B,)
        return self.cfg.gamma_high * phi_sc * (1.0 - done.float()) - phi_s0

    @torch.no_grad()
    def apply_low_her(
        self,
        batch: LowLevelSample,
        future_next_obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> LowLevelSample:
        """Relabel a worker batch with future achieved goals from the same episode.

        Hindsight goal in geometric subgoal space:
            g_her = project(s_future) - project(s)
        i.e. make the subgoal target equal to a state that was actually achieved
        later in the same episode. The relabeled next_subgoal and worker reward
        are then recomputed under the current worker reward definition.
        """
        if mask is None:
            mask = torch.ones(batch.obs.shape[0], dtype=torch.bool, device=batch.obs.device)
        if mask.ndim != 1 or mask.shape[0] != batch.obs.shape[0]:
            raise ValueError("mask must be shape (B,)")
        if not mask.any():
            return batch

        obs = batch.obs
        achieved = future_next_obs.to(obs.device)
        g_base = self.sp.project(achieved) - self.sp.project(obs)
        g_interface = self.encode_subgoal(g_base)

        subgoal = batch.subgoal.clone()
        next_subgoal = batch.next_subgoal.clone()
        reward = batch.reward.clone()

        subgoal[mask] = g_interface[mask]
        next_subgoal[mask] = self.subgoal_transition(obs[mask], g_interface[mask], batch.next_obs[mask])
        reward[mask] = self.worker_reward(obs[mask], g_interface[mask], batch.next_obs[mask])

        return LowLevelSample(
            obs=batch.obs,
            subgoal=subgoal,
            action=batch.action,
            reward=reward,
            next_obs=batch.next_obs,
            next_subgoal=next_subgoal,
            done=batch.done,
            t_inds=batch.t_inds,
            e_inds=batch.e_inds,
            episode_id=batch.episode_id,
            episode_step=batch.episode_step,
        )

    def update_high(self, batch: HighLevelSample) -> dict:
        """SAC update for the manager, after off-policy relabeling of the action.

        With cfg.use_off_policy_correction=False the manager trains on the raw
        stored subgoal g0 — a debugging knob to isolate whether the correction
        helps or hurts.

        When place_pbrs_beta>0, an ungated potential-based place reward is added
        to reward_sum (Phase 2) to remove the manager's grasp-and-hold cliff.
        """
        if self.cfg.use_off_policy_correction:
            action = self.off_policy_correct(batch)
        else:
            action = batch.g0
        reward = batch.reward_sum
        if self.cfg.place_pbrs_beta > 0.0 and self._place_dims is not None:
            reward = reward + self._manager_place_shaping(batch.s0, batch.s_c, batch.done)
        return self._sac_update(
            actor=self.nets.manager_actor,
            critic=self.nets.manager_critic,
            critic_target=self.manager_critic_target,
            optims=self.high,
            gamma=self.cfg.gamma_high,
            state=self._mgr_in(batch.s0), action=action, reward=reward,
            next_state=self._mgr_in(batch.s_c), done=batch.done,
        )

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        sd = {
            "worker_actor": self.nets.worker_actor.state_dict(),
            "worker_critic": self.nets.worker_critic.state_dict(),
            "worker_critic_target": self.worker_critic_target.state_dict(),
            "manager_actor": self.nets.manager_actor.state_dict(),
            "manager_critic": self.nets.manager_critic.state_dict(),
            "manager_critic_target": self.manager_critic_target.state_dict(),
            "log_alpha_low": self.log_alpha_low.detach(),
            "log_alpha_high": self.log_alpha_high.detach(),
        }
        if self.latent_codec is not None:
            sd["latent_codec"] = self.latent_codec.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.nets.worker_actor.load_state_dict(sd["worker_actor"])
        self.nets.worker_critic.load_state_dict(sd["worker_critic"])
        self.worker_critic_target.load_state_dict(sd["worker_critic_target"])
        self.nets.manager_actor.load_state_dict(sd["manager_actor"])
        self.nets.manager_critic.load_state_dict(sd["manager_critic"])
        self.manager_critic_target.load_state_dict(sd["manager_critic_target"])
        if self.latent_codec is not None and "latent_codec" in sd:
            self.latent_codec.load_state_dict(sd["latent_codec"])
        with torch.no_grad():
            self.log_alpha_low.copy_(sd["log_alpha_low"].to(self.device))
            self.log_alpha_high.copy_(sd["log_alpha_high"].to(self.device))

    def load_worker(self, sd: dict) -> None:
        """Warm-start ONLY the worker (low-level) from a checkpoint state_dict,
        leaving the manager at its fresh random init. Used for worker-pretraining:
        plug in an already-competent subgoal-following worker, then train the
        manager on top of it (collapses the HIRO cold-start)."""
        self.nets.worker_actor.load_state_dict(sd["worker_actor"])
        self.nets.worker_critic.load_state_dict(sd["worker_critic"])
        self.worker_critic_target.load_state_dict(sd["worker_critic_target"])
        with torch.no_grad():
            self.log_alpha_low.copy_(sd["log_alpha_low"].to(self.device))

    def load_manager(self, sd: dict) -> None:
        """Load ONLY the manager (high-level) from a checkpoint state_dict, leaving the
        worker at fresh random init. Used for embodiment TRANSFER: reuse a (body-
        independent, object-centric) manager and re-adapt only the worker on a new
        embodiment. Pair with freeze_manager to keep the transferred manager fixed."""
        self.nets.manager_actor.load_state_dict(sd["manager_actor"])
        self.nets.manager_critic.load_state_dict(sd["manager_critic"])
        self.manager_critic_target.load_state_dict(sd["manager_critic_target"])
        with torch.no_grad():
            self.log_alpha_high.copy_(sd["log_alpha_high"].to(self.device))
