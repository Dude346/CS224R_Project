"""
hiro/smoke_pipeline.py
----------------------
End-to-end smoke test for the HIRO pipeline on a *controllable* mini-PickCube
env (known dynamics, grasping required, success reachable) running on CPU.

Unlike the unit tests (which check wiring/consistency), this drives the REAL
HIROAgent + LowLevelBuffer + HighLevelBuffer + HierarchicalRollout + SAC updates
through a closed loop, exactly like hiro/trainer.py minus gym/W&B. The point is
to answer empirically: does each level actually LEARN?

It runs several scenarios and prints a verdict table:

  1. worker-only reach   : fixed random subgoals, can the worker drive the
                           intrinsic reward toward 0? (isolates worker SAC)
  2. manager full task   : full hierarchy on mini-PickCube with the CURRENT
                           phased/PBRS worker reward -> does grasp_rate rise and
                           does the task get solved?
  3. ablations           : plain intrinsic reward, bigger pbrs_alpha, and a
                           worker reward that includes the env reaching signal,
                           to localize WHY (if) the full task stalls.

Run:
    .venv/bin/python -m hiro.smoke_pipeline
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from hiro.hiro_agent import HIROAgent, HIROConfig
from hiro.replay_buffer import HighLevelBuffer, LowLevelBuffer
from hiro.rollout import HierarchicalRollout
from hiro.subgoal_space import ObsBitGraspReader, SubgoalSpace

# ---------------------------------------------------------------------------
# Mini-PickCube: a faithful but tiny vectorized env (pure torch, CPU).
#
# obs (dim=10):  [ tcp(0:3), cube(3:6), goal(6:9), is_grasped(9) ]
# action (dim=4): [ dx, dy, dz, grip ]   all in [-1, 1]
#
# Dynamics:
#   tcp'  = clip(tcp + STEP * action[:3], workspace)
#   grasp'= (grip > 0) AND ||tcp' - cube|| < GRASP_RADIUS     (must close near cube)
#   cube' = tcp'  if grasp' else cube                          (cube follows hand)
# Env reward (dense, normalized-ish, the MANAGER's reward):
#   r = 1 - tanh(5*||cube-goal||)         in (0,1], 1 at the goal
# success: ||cube - goal|| < SUCCESS_RADIUS
#
# This is the smallest env that needs the full pick-and-place competency:
# reach the cube, CLOSE the gripper, carry the cube to the goal.
# ---------------------------------------------------------------------------

STEP = 0.1
GRASP_RADIUS = 0.10
SUCCESS_RADIUS = 0.05
HORIZON = 50
OBS_DIM = 10
ACT_DIM = 4


class MiniPickCube:
    def __init__(self, num_envs: int, device="cpu", seed: int = 0):
        self.E = num_envs
        self.device = torch.device(device)
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.tcp = torch.zeros(num_envs, 3)
        self.cube = torch.zeros(num_envs, 3)
        self.goal = torch.zeros(num_envs, 3)
        self.grasped = torch.zeros(num_envs)
        self.t = torch.zeros(num_envs, dtype=torch.long)

    def _rand(self, n, lo, hi):
        return torch.rand(n, 3, generator=self.g) * (hi - lo) + lo

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        self.tcp[idx] = self._rand(n, -0.3, 0.3)
        self.cube[idx] = self._rand(n, -0.3, 0.3)
        self.goal[idx] = self._rand(n, -0.3, 0.3)
        self.grasped[idx] = 0.0
        self.t[idx] = 0

    def _obs(self):
        return torch.cat(
            [self.tcp, self.cube, self.goal, self.grasped.unsqueeze(-1)], dim=-1
        )

    def reset(self):
        self._reset_idx(torch.arange(self.E))
        return self._obs()

    def step(self, action):
        action = action.clamp(-1, 1)
        self.tcp = (self.tcp + STEP * action[:, :3]).clamp(-0.5, 0.5)
        near = (self.tcp - self.cube).norm(dim=-1) < GRASP_RADIUS
        close = action[:, 3] > 0
        self.grasped = (near & close).float()
        carry = self.grasped.bool()
        self.cube[carry] = self.tcp[carry]

        dist = (self.cube - self.goal).norm(dim=-1)
        reward = 1.0 - torch.tanh(5.0 * dist)
        success = (dist < SUCCESS_RADIUS)

        self.t += 1
        truncated = self.t >= HORIZON
        obs = self._obs()

        # auto-reset truncated envs; return the pre-reset obs as terminal obs
        terminal_obs = obs.clone()
        if truncated.any():
            self._reset_idx(truncated.nonzero(as_tuple=True)[0])
        next_obs = self._obs()
        return next_obs, reward, terminal_obs, truncated, success


def mini_subgoal_space() -> SubgoalSpace:
    return SubgoalSpace(
        indices=[0, 1, 2, 3, 4, 5],  # tcp_xyz + cube_xyz
        obs_dim=OBS_DIM,
        label="mini tcp+cube",
        object_dims=(3, 4, 5),
    )


# ---------------------------------------------------------------------------
# Training loop (mirrors hiro/trainer.py, env-agnostic, CPU)
# ---------------------------------------------------------------------------

@dataclass
class SmokeCfg:
    total_steps: int = 60_000
    num_envs: int = 16
    learning_starts: int = 2_000
    training_freq: int = 32
    utd: float = 1.0
    batch_size: int = 256
    c: int = 10
    subgoal_scale: float = 0.20
    gamma_low: float = 0.95
    gamma_high: float = 0.95
    use_phased_reward: bool = True
    pbrs_alpha: float = 0.1
    reach_coef: float = 1.0           # dense pre-grasp object-reach term (the fix)
    reach_temp: float = 5.0
    use_off_policy_correction: bool = True
    log_every: int = 10_000


def build_agent(scfg: SmokeCfg) -> HIROAgent:
    sp = mini_subgoal_space()
    cfg = HIROConfig(
        action_dim=ACT_DIM, c=scfg.c, subgoal_scale=scfg.subgoal_scale,
        action_low=-1.0, action_high=1.0,
        gamma_low=scfg.gamma_low, gamma_high=scfg.gamma_high,
        use_off_policy_correction=scfg.use_off_policy_correction,
        use_phased_reward=scfg.use_phased_reward, pbrs_alpha=scfg.pbrs_alpha,
        reach_coef=scfg.reach_coef, reach_temp=scfg.reach_temp,
        device="cpu",
    )
    agent = HIROAgent(sp, cfg)
    if scfg.use_phased_reward:
        agent.grasp_detector = ObsBitGraspReader(obs_index=9)
    return agent


@torch.no_grad()
def evaluate(agent: HIROAgent, scfg: SmokeCfg, seed: int = 999, episodes: int = 4):
    env = MiniPickCube(scfg.num_envs, seed=seed)
    obs = env.reset()
    cur_g = agent.select_subgoal(obs, deterministic=True)
    since = torch.zeros(scfg.num_envs, dtype=torch.long)
    succ_any = torch.zeros(scfg.num_envs, dtype=torch.bool)
    succ_counts, ep_done = [], 0
    for _ in range(HORIZON * episodes):
        action = agent.select_action(obs, cur_g, deterministic=True)
        prev = obs
        obs, _, _, truncated, success = env.step(action)
        cur_g = agent.subgoal_transition(prev, cur_g, obs)
        succ_any |= success
        since += 1
        reissue = (since >= scfg.c) | truncated
        if reissue.any():
            idx = reissue.nonzero(as_tuple=True)[0]
            cur_g[idx] = agent.select_subgoal(obs[idx], deterministic=True)
            since[idx] = 0
        if truncated.any():
            ti = truncated.nonzero(as_tuple=True)[0]
            succ_counts.append(succ_any[ti].float())
            succ_any[ti] = False
            ep_done += ti.numel()
    return torch.cat(succ_counts).mean().item() if succ_counts else 0.0


def run(scfg: SmokeCfg, label: str, worker_reward_mode: str = "default") -> dict:
    """worker_reward_mode: 'default' uses agent.worker_reward; 'env_reach' adds the
    env reaching signal to the worker reward to test the grasp-incentive hypothesis."""
    torch.manual_seed(0)
    agent = build_agent(scfg)
    sp = agent.sp
    env = MiniPickCube(scfg.num_envs, seed=0)

    low = LowLevelBuffer(200_000, scfg.num_envs, OBS_DIM, sp.dim, ACT_DIM)
    high = HighLevelBuffer(50_000, OBS_DIM, sp.dim, ACT_DIM, scfg.c)
    rollout = HierarchicalRollout(agent, low, high, scfg.num_envs, device="cpu")

    # Monkeypatch the worker reward for the ablation, leaving the pipeline intact.
    if worker_reward_mode == "env_reach":
        base_worker_reward = agent.worker_reward

        def reach_aug(s, g, s_next):
            r = base_worker_reward(s, g, s_next)
            # add the env's reaching signal so the worker is pulled to the cube
            tcp = s_next[..., 0:3]
            cube = s_next[..., 3:6]
            return r + (1.0 - torch.tanh(5.0 * (tcp - cube).norm(dim=-1)))

        agent.worker_reward = reach_aug  # type: ignore

    if worker_reward_mode == "grasp_dense":
        base_worker_reward = agent.worker_reward
        det = agent.grasp_detector

        def grasp_aug(s, g, s_next):
            # DENSE grasp signal: continuously reward being grasped, so the worker
            # has a gradient toward (and a reason to keep) closing the gripper.
            r = base_worker_reward(s, g, s_next)
            return r + 0.5 * det(s_next).float()

        agent.worker_reward = grasp_aug  # type: ignore

    obs = env.reset()
    cur_g = rollout.start(obs)
    grad_steps = max(1, int(scfg.training_freq * scfg.utd))
    steps_per_env = max(1, scfg.training_freq // scfg.num_envs)
    high_updates = max(1, grad_steps // scfg.c)

    gstep = 0
    started = False
    hist = []
    grasp_acc, intr_acc, n_acc = 0.0, 0.0, 0
    while gstep < scfg.total_steps:
        for _ in range(steps_per_env):
            if not started:
                action = torch.rand(scfg.num_envs, ACT_DIM) * 2 - 1
            else:
                action = agent.select_action(obs, cur_g)
            next_obs, reward, terminal_obs, truncated, success = env.step(action)
            episode_end = truncated.bool()
            bootstrap_done = torch.zeros(scfg.num_envs)  # truncation => keep bootstrapping
            cur_g = rollout.record(
                obs, action, terminal_obs, next_obs, reward, episode_end, bootstrap_done
            )
            # cheap running diagnostics from the freshest low transition
            grasp_acc += agent.grasp_detector(next_obs).float().mean().item() if agent.grasp_detector else 0.0
            n_acc += 1
            obs = next_obs
            gstep += scfg.num_envs

        if gstep >= scfg.learning_starts:
            started = True
            for _ in range(grad_steps):
                lm = agent.update_low(low.sample(scfg.batch_size))
            intr_acc += lm["q_loss"]
            if len(high) >= scfg.batch_size:
                for _ in range(high_updates):
                    agent.update_high(high.sample(scfg.batch_size))

        if gstep % scfg.log_every < scfg.training_freq and started:
            succ = evaluate(agent, scfg)
            ls = low.sample(min(scfg.batch_size, len(low)))
            det = agent.grasp_detector
            grasp_rate = det(ls.next_obs).float().mean().item() if det else float("nan")
            mean_intr = ls.reward.mean().item()
            mgr = high.sample(min(scfg.batch_size, len(high))).reward_sum.mean().item() if len(high) else float("nan")
            hist.append((gstep, succ, grasp_rate, mean_intr, mgr))
            print(f"  [{label:22s} step {gstep:>6}] eval_success={succ:.3f}  "
                  f"grasp_rate={grasp_rate:.3f}  mean_worker_r={mean_intr:+.3f}  mgr_r={mgr:+.2f}")

    final_succ = evaluate(agent, scfg, episodes=8)
    return {"label": label, "final_success": final_succ, "hist": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--scenario", default="all",
                    choices=["all", "default", "no_reach", "no_pbrs", "big_pbrs",
                             "env_reach", "no_phase", "grasp_dense"])
    args = ap.parse_args()

    results = []

    def maybe(name, scfg, mode="default"):
        if args.scenario in ("all", name):
            print(f"\n=== scenario: {name} ===")
            results.append(run(scfg, name, worker_reward_mode=mode))

    # default now includes the reach-term fix (reach_coef=1.0); no_reach is the OLD
    # broken reward for side-by-side comparison.
    maybe("default", SmokeCfg(total_steps=args.steps, pbrs_alpha=0.1, reach_coef=1.0))
    maybe("no_reach", SmokeCfg(total_steps=args.steps, pbrs_alpha=0.1, reach_coef=0.0))
    maybe("no_phase", SmokeCfg(total_steps=args.steps, use_phased_reward=False))
    maybe("big_pbrs", SmokeCfg(total_steps=args.steps, pbrs_alpha=1.0))
    maybe("env_reach", SmokeCfg(total_steps=args.steps, pbrs_alpha=0.1), mode="env_reach")
    maybe("grasp_dense", SmokeCfg(total_steps=args.steps, pbrs_alpha=0.1), mode="grasp_dense")

    print("\n================ VERDICT ================")
    for r in results:
        print(f"  {r['label']:12s}  final_eval_success = {r['final_success']:.3f}")
    print("=========================================")


if __name__ == "__main__":
    main()
