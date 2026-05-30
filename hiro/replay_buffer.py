"""
hiro/replay_buffer.py
---------------------
Replay buffers for the two HIRO levels.  Both are *dumb storage*: they hold
whatever tensors the trainer hands them and return uniform random samples.
All HIRO semantics (intrinsic reward, subgoal transition, off-policy
correction) live in the agent/trainer, not here.  This keeps the buffers easy
to reason about and test in isolation.

Two buffers
-----------
LowLevelBuffer (worker)
    One transition *per env, per timestep* — synchronous, so it mirrors the
    upstream ManiSkill ReplayBuffer layout: storage shaped
    (per_env_capacity, num_envs, dim) with a single shared write pointer.
    Stores (obs, subgoal, action, reward, next_obs, next_subgoal, done).
    The worker's SAC state is [obs, subgoal]; the trainer concatenates at
    update time.

HighLevelBuffer (manager)
    One transition *per env, every c steps OR at episode end* — these flushes
    are ASYNCHRONOUS across envs (episodes end at different steps), so a
    per-env layout with a shared pointer doesn't fit.  Instead this is a flat
    ring buffer written with a variable-size batched add: each step the trainer
    gathers however many envs flushed and appends that batch.
    Stores (s0, g0, reward_sum, s_c, done) plus the c intermediate (state,
    action) pairs needed for off-policy correction, padded to length c with a
    per-entry valid length.

Why store intermediates?  HIRO's off-policy correction relabels a stored
high-level action (subgoal) by re-evaluating the low-level log-prob of the
actually-taken primitive actions under the *current* worker policy.  That needs
the intermediate states s_0..s_{c-1} and actions a_0..a_{c-1}.  Episodes that
end early have length < c; the padded tail must be masked using `inter_len`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Sample containers
# ---------------------------------------------------------------------------

@dataclass
class LowLevelSample:
    obs: torch.Tensor           # (B, obs_dim)
    subgoal: torch.Tensor       # (B, subgoal_dim)
    action: torch.Tensor        # (B, action_dim)
    reward: torch.Tensor        # (B,)
    next_obs: torch.Tensor      # (B, obs_dim)
    next_subgoal: torch.Tensor  # (B, subgoal_dim)
    done: torch.Tensor          # (B,)


@dataclass
class HighLevelSample:
    s0: torch.Tensor          # (B, obs_dim)        obs when subgoal was issued
    g0: torch.Tensor          # (B, subgoal_dim)    the issued subgoal (manager action)
    reward_sum: torch.Tensor  # (B,)                summed env reward over the segment
    s_c: torch.Tensor         # (B, obs_dim)        obs c steps later (segment end)
    done: torch.Tensor        # (B,)
    inter_obs: torch.Tensor   # (B, c, obs_dim)     intermediate states s_0..s_{c-1} (padded)
    inter_act: torch.Tensor   # (B, c, action_dim)  intermediate actions a_0..a_{c-1} (padded)
    inter_len: torch.Tensor   # (B,)  long          number of valid intermediate steps (<= c)


# ---------------------------------------------------------------------------
# Low-level (worker) buffer — synchronous per-env ring, mirrors upstream SAC
# ---------------------------------------------------------------------------

class LowLevelBuffer:
    """Standard SAC replay buffer for the worker, vectorized over envs.

    Capacity is the total number of stored transitions; per-env capacity is
    `capacity // num_envs`.  Every `add` writes one transition for *all* envs.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        subgoal_dim: int,
        action_dim: int,
        storage_device: torch.device | str = "cpu",
        sample_device: torch.device | str = "cpu",
    ):
        self.num_envs = num_envs
        self.per_env_capacity = capacity // num_envs
        if self.per_env_capacity < 1:
            raise ValueError(f"capacity {capacity} too small for num_envs {num_envs}")
        self.capacity = self.per_env_capacity * num_envs
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)

        self.pos = 0
        self.full = False

        E, S = self.per_env_capacity, num_envs
        dev = self.storage_device
        self.obs = torch.zeros((E, S, obs_dim), device=dev)
        self.subgoal = torch.zeros((E, S, subgoal_dim), device=dev)
        self.action = torch.zeros((E, S, action_dim), device=dev)
        self.reward = torch.zeros((E, S), device=dev)
        self.next_obs = torch.zeros((E, S, obs_dim), device=dev)
        self.next_subgoal = torch.zeros((E, S, subgoal_dim), device=dev)
        self.done = torch.zeros((E, S), device=dev)

    def __len__(self) -> int:
        return (self.per_env_capacity if self.full else self.pos) * self.num_envs

    def add(
        self,
        obs: torch.Tensor,
        subgoal: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        next_subgoal: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Append one transition for every env. Leading dim of each arg = num_envs."""
        dev = self.storage_device
        i = self.pos
        self.obs[i] = obs.to(dev)
        self.subgoal[i] = subgoal.to(dev)
        self.action[i] = action.to(dev)
        self.reward[i] = reward.to(dev)
        self.next_obs[i] = next_obs.to(dev)
        self.next_subgoal[i] = next_subgoal.to(dev)
        self.done[i] = done.to(dev).float()

        self.pos += 1
        if self.pos == self.per_env_capacity:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> LowLevelSample:
        high = self.per_env_capacity if self.full else self.pos
        if high == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        t_inds = torch.randint(0, high, size=(batch_size,))
        e_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        dev = self.sample_device
        return LowLevelSample(
            obs=self.obs[t_inds, e_inds].to(dev),
            subgoal=self.subgoal[t_inds, e_inds].to(dev),
            action=self.action[t_inds, e_inds].to(dev),
            reward=self.reward[t_inds, e_inds].to(dev),
            next_obs=self.next_obs[t_inds, e_inds].to(dev),
            next_subgoal=self.next_subgoal[t_inds, e_inds].to(dev),
            done=self.done[t_inds, e_inds].to(dev),
        )


# ---------------------------------------------------------------------------
# High-level (manager) buffer — flat ring, asynchronous batched add
# ---------------------------------------------------------------------------

class HighLevelBuffer:
    """Flat ring buffer for manager transitions with off-policy-correction data.

    `capacity` is the total number of high-level transitions.  Because envs
    flush at different steps, `add_batch` appends a *variable* number M of
    transitions (one per env that flushed this step), wrapping the ring as
    needed.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        subgoal_dim: int,
        action_dim: int,
        c: int,
        storage_device: torch.device | str = "cpu",
        sample_device: torch.device | str = "cpu",
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.c = c
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)

        self.pos = 0
        self.full = False

        N, dev = capacity, self.storage_device
        self.s0 = torch.zeros((N, obs_dim), device=dev)
        self.g0 = torch.zeros((N, subgoal_dim), device=dev)
        self.reward_sum = torch.zeros((N,), device=dev)
        self.s_c = torch.zeros((N, obs_dim), device=dev)
        self.done = torch.zeros((N,), device=dev)
        self.inter_obs = torch.zeros((N, c, obs_dim), device=dev)
        self.inter_act = torch.zeros((N, c, action_dim), device=dev)
        self.inter_len = torch.zeros((N,), dtype=torch.long, device=dev)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add_batch(
        self,
        s0: torch.Tensor,          # (M, obs_dim)
        g0: torch.Tensor,          # (M, subgoal_dim)
        reward_sum: torch.Tensor,  # (M,)
        s_c: torch.Tensor,         # (M, obs_dim)
        done: torch.Tensor,        # (M,)
        inter_obs: torch.Tensor,   # (M, c, obs_dim)   padded
        inter_act: torch.Tensor,   # (M, c, action_dim) padded
        inter_len: torch.Tensor,   # (M,)              valid lengths (<= c)
    ) -> None:
        """Append M high-level transitions, wrapping the ring if necessary."""
        M = s0.shape[0]
        if M == 0:
            return
        if M > self.capacity:
            raise ValueError(f"batch size {M} exceeds buffer capacity {self.capacity}")

        end = self.pos + M
        if end <= self.capacity:
            self._write(slice(self.pos, end), slice(0, M),
                        s0, g0, reward_sum, s_c, done, inter_obs, inter_act, inter_len)
        else:
            first = self.capacity - self.pos          # fills to the end
            self._write(slice(self.pos, self.capacity), slice(0, first),
                        s0, g0, reward_sum, s_c, done, inter_obs, inter_act, inter_len)
            self._write(slice(0, end - self.capacity), slice(first, M),
                        s0, g0, reward_sum, s_c, done, inter_obs, inter_act, inter_len)
            self.full = True

        if end >= self.capacity:
            self.full = True
        self.pos = end % self.capacity

    def _write(self, dst: slice, src: slice,
               s0, g0, reward_sum, s_c, done, inter_obs, inter_act, inter_len) -> None:
        dev = self.storage_device
        self.s0[dst] = s0[src].to(dev)
        self.g0[dst] = g0[src].to(dev)
        self.reward_sum[dst] = reward_sum[src].to(dev)
        self.s_c[dst] = s_c[src].to(dev)
        self.done[dst] = done[src].to(dev).float()
        self.inter_obs[dst] = inter_obs[src].to(dev)
        self.inter_act[dst] = inter_act[src].to(dev)
        self.inter_len[dst] = inter_len[src].to(dev).long()

    def sample(self, batch_size: int) -> HighLevelSample:
        high = len(self)
        if high == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        inds = torch.randint(0, high, size=(batch_size,))
        dev = self.sample_device
        return HighLevelSample(
            s0=self.s0[inds].to(dev),
            g0=self.g0[inds].to(dev),
            reward_sum=self.reward_sum[inds].to(dev),
            s_c=self.s_c[inds].to(dev),
            done=self.done[inds].to(dev),
            inter_obs=self.inter_obs[inds].to(dev),
            inter_act=self.inter_act[inds].to(dev),
            inter_len=self.inter_len[inds].to(dev),
        )
