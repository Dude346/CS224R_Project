"""
hiro/rollout.py
---------------
The per-env subgoal bookkeeping for HIRO's vectorized training loop, factored
out of the trainer so it can be unit-tested on CPU with hand-fed transitions
(no env, no GPU).  This is where HIRO's trickiest bugs live; isolating it is
deliberate.

State tracked per env (E = num_envs)
------------------------------------
  cur_subgoal    (E, sg)   subgoal used to pick the NEXT action; evolves within
                           a segment via the goal transition, resets at flush
  seg_start_obs  (E, obs)  s0: obs when the current segment's subgoal was issued
  seg_subgoal    (E, sg)   g0: the manager's issued subgoal (FIXED for the segment)
  seg_reward     (E,)      running sum of env reward in the current segment
  seg_len        (E,)      steps taken so far in the segment (0..c)
  seg_obs/seg_act(E,c,*)   intermediate (s_i, a_i) for off-policy correction

Per-step ordering (the bug-prone part), implemented in `record`:
  1. r_lo = intrinsic_reward(s, g, s')          (worker reward)
  2. g_next = subgoal_transition(s, g, s')       (telescoped subgoal)
  3. low_buffer.add(s, g, a, r_lo, s', g_next, done)
  4. store (s, a) at slot seg_len               (=> intermediates are s_0..s_{c-1})
  5. seg_reward += r_env;  seg_len += 1
  6. flush = (seg_len >= c) OR episode_end
  7. for flushing envs: push (s0, g0, seg_reward, s_c=s', intermediates, seg_len)
     to high_buffer, then sample a fresh subgoal and reset segment state
  8. non-flushing envs: cur_subgoal <- g_next

Two different "next obs" must be passed in:
  real_next_obs : the obs that *ends* this transition — the true terminal obs on
                  episode end (used for r_lo, the subgoal transition, the low
                  buffer's next_obs, and the high segment's s_c).
  resample_obs  : the obs the NEXT segment starts from — the new-episode obs after
                  an auto-reset (used to sample the new subgoal and as the next
                  seg_start_obs).  Equals real_next_obs when no reset happened.
Confusing these is the classic HIRO-on-vector-envs bug, so they are explicit.
"""
from __future__ import annotations

import torch


class HierarchicalRollout:
    def __init__(self, agent, low_buffer, high_buffer, num_envs: int, device="cpu"):
        self.agent = agent
        self.low = low_buffer
        self.high = high_buffer
        self.E = num_envs
        self.c = agent.c
        self.device = torch.device(device)

        self.obs_dim = agent.sp.obs_dim
        self.sg_dim = agent.sp.dim
        self.act_dim = agent.cfg.action_dim

        E, c, dev = num_envs, self.c, self.device
        self.cur_subgoal = torch.zeros(E, self.sg_dim, device=dev)
        self.seg_start_obs = torch.zeros(E, self.obs_dim, device=dev)
        self.seg_subgoal = torch.zeros(E, self.sg_dim, device=dev)
        self.seg_reward = torch.zeros(E, device=dev)
        self.seg_len = torch.zeros(E, dtype=torch.long, device=dev)
        self.seg_obs = torch.zeros(E, c, self.obs_dim, device=dev)
        self.seg_act = torch.zeros(E, c, self.act_dim, device=dev)

    @torch.no_grad()
    def start(self, obs: torch.Tensor) -> torch.Tensor:
        """Issue the first subgoal for every env. Returns cur_subgoal."""
        g = self.agent.select_subgoal(obs)
        self.cur_subgoal = g.clone()
        self.seg_subgoal = g.clone()
        self.seg_start_obs = obs.clone()
        self.seg_reward.zero_()
        self.seg_len.zero_()
        return self.cur_subgoal

    @torch.no_grad()
    def record(
        self,
        obs: torch.Tensor,            # (E, obs)  s_t  (state the action was taken in)
        action: torch.Tensor,         # (E, act)  a_t
        real_next_obs: torch.Tensor,  # (E, obs)  s_{t+1} for storage (terminal obs on episode end)
        resample_obs: torch.Tensor,   # (E, obs)  obs to start the next segment from
        r_env: torch.Tensor,          # (E,)      env reward
        episode_end: torch.Tensor,    # (E,) bool episode ended this step (flush trigger)
        bootstrap_done: torch.Tensor, # (E,)      SAC done/bootstrap mask stored in buffers
    ) -> torch.Tensor:
        """Process one vectorized transition; returns the updated cur_subgoal."""
        E, c = self.E, self.c
        ar = torch.arange(E, device=self.device)
        episode_end = episode_end.bool()

        # 1-3: low-level (worker) transition uses the CURRENT subgoal.
        r_lo = self.agent.intrinsic_reward(obs, self.cur_subgoal, real_next_obs)
        g_next = self.agent.subgoal_transition(obs, self.cur_subgoal, real_next_obs)
        self.low.add(obs, self.cur_subgoal, action, r_lo, real_next_obs, g_next, bootstrap_done)

        # 4: record the intermediate (s_i, a_i) at the current segment position.
        #    seg_len is in [0, c-1] here (it is reset to 0 on flush below), so this
        #    never writes out of bounds.
        self.seg_obs[ar, self.seg_len] = obs
        self.seg_act[ar, self.seg_len] = action

        # 5: accumulate reward, advance segment counter.
        self.seg_reward = self.seg_reward + r_env
        self.seg_len = self.seg_len + 1

        # 6: a segment ends when it reaches length c OR the episode ended.
        flush = (self.seg_len >= c) | episode_end

        # 7: flush completed segments to the high-level buffer (BEFORE resetting them).
        if flush.any():
            idx = flush.nonzero(as_tuple=True)[0]
            self.high.add_batch(
                self.seg_start_obs[idx],
                self.seg_subgoal[idx],
                self.seg_reward[idx],
                real_next_obs[idx],          # s_c = terminal obs of the segment
                bootstrap_done[idx],
                self.seg_obs[idx],
                self.seg_act[idx],
                self.seg_len[idx],
            )

        # 8: default subgoal update is the telescoped transition...
        self.cur_subgoal = g_next
        # ...but flushed envs get a fresh subgoal sampled from the next segment's start.
        if flush.any():
            new_g = self.agent.select_subgoal(resample_obs[idx])
            self.cur_subgoal[idx] = new_g
            self.seg_subgoal[idx] = new_g
            self.seg_start_obs[idx] = resample_obs[idx]
            self.seg_reward[idx] = 0.0
            self.seg_len[idx] = 0

        return self.cur_subgoal
