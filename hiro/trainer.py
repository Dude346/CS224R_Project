"""
hiro/trainer.py
---------------
The HIRO training loop.  Thin glue around the tested pieces:
  - env setup mirrors ManiSkill's upstream sac.py (same wrappers, obs/bootstrap
    handling) so HIRO is compared against the flat baseline on equal footing
  - the per-env subgoal bookkeeping is delegated to HierarchicalRollout (tested)
  - the SAC updates + off-policy correction live in HIROAgent (tested)

Everything that is genuinely novel/fiddly is in the unit-tested modules; this
file is orchestration + logging + eval + checkpointing.  It can only be fully
validated on a GPU container (Vulkan), so it is intentionally kept simple.

Diagnostics are logged from day one (eval success/return, per-level q/actor/
alpha losses, mean intrinsic reward, mean manager reward, mean subgoal norm,
buffer sizes) so that "HIRO isn't learning" is debugged by reading the W&B
dashboard, not by adding prints and re-running.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from hiro.hiro_agent import HIROAgent, HIROConfig
from hiro.replay_buffer import HighLevelBuffer, LowLevelBuffer
from hiro.rollout import HierarchicalRollout
from hiro.subgoal_space import get_grasp_detector, get_subgoal_space, normalize_subgoal_variant


@dataclass
class TrainArgs:
    env_id: str = "PickCube-v1"
    subgoal_variant: str = "hybrid"
    robot_uids: str = "panda"
    control_mode: str = "pd_joint_delta_pos"
    seed: int = 1
    total_timesteps: int = 1_000_000
    num_envs: int = 32
    num_eval_envs: int = 16
    num_eval_steps: int = 200
    eval_freq: int = 50_000           # in env steps
    learning_starts: int = 4_000
    training_freq: int = 64           # env steps between update phases
    utd: float = 0.5                  # gradient steps per env step collected
    batch_size: int = 512
    low_buffer_size: int = 1_000_000
    high_buffer_size: int = 200_000
    # HIRO hyperparameters
    c: int = 10
    subgoal_scale: float = 0.15
    gamma_low: float = 0.8
    gamma_high: float = 0.97
    tau: float = 0.005
    use_phased_reward: bool = True
    pbrs_alpha: float = 0.1            # PBRS grasp potential strength (replaces grasp_bonus)
    latent_subgoal_dim: int = 6        # only used by latent_object_centric; 6 == base
                                       # object-centric dim, so no lossy compression
                                       # (drop below 6 only to deliberately bottleneck)
    # partial_reset=True: episodes end on TRUE termination (env success), not just
    # horizon. Defaults ON for HIRO (avoids per-step PBRS hold-cost drag after
    # success). Flat SAC baselines used False to match upstream.
    partial_reset: bool = False
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    num_candidates: int = 10
    use_off_policy_correction: bool = True
    # infra / logging
    exp_name: Optional[str] = None
    wandb_project: str = "cs224r-hiro"
    track: bool = False
    capture_video: bool = True
    save_freq_steps: int = 200_000
    buffer_device: str = "cuda"
    device: str = "cuda"
    output_root: str = "runs"


# ---------------------------------------------------------------------------
# Env construction (mirrors upstream sac.py)
# ---------------------------------------------------------------------------

def build_envs(args: TrainArgs, run_name: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers tasks
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    if args.robot_uids != "panda":
        import weak_panda  # noqa: F401 — registers custom robots on import

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="gpu",
        control_mode=args.control_mode, robot_uids=args.robot_uids,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs,
                    reconfiguration_freq=None, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=1,
                         human_render_camera_configs=dict(shader_pack="default"),
                         **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_envs = RecordEpisode(
            eval_envs, output_dir=f"{args.output_root}/{run_name}/videos",
            save_trajectory=False, save_video=True,
            max_steps_per_video=args.num_eval_steps, video_fps=30,
        )
    # partial_reset=True (default for HIRO) => TRAINING envs end and auto-reset on
    # TRUE env termination (e.g. PickCube success), not just at the 50-step horizon.
    # Avoids the per-step shaped-reward drag on successful trajectories.
    # Set False to match the flat SAC baseline's "always run to horizon" behaviour.
    #
    # EVAL envs always run to horizon (ignore_terminations=True). Two reasons:
    # (1) ManiSkill asserts reconfiguration_freq=0 under partial_reset, and we
    #     want reconfiguration_freq=1 on eval to randomize object poses per eval;
    # (2) measuring success_once over a fixed 50-step horizon gives a stable,
    #     comparable metric across runs -- the PBRS post-success drag is purely
    #     a training-time issue and irrelevant when we're not learning from reward.
    ignore_term_train = not args.partial_reset
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=ignore_term_train, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)
    return envs, eval_envs


# ---------------------------------------------------------------------------
# Greedy evaluation (no buffer writes; deterministic worker + manager)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(agent: HIROAgent, eval_envs, args: TrainArgs) -> dict:
    E = args.num_eval_envs
    c = args.c
    device = agent.device

    obs, _ = eval_envs.reset(seed=args.seed)  # fixed seed => reproducible eval episodes
    cur_g = agent.select_subgoal(obs, deterministic=True)
    steps_since = torch.zeros(E, dtype=torch.long, device=device)

    metrics = defaultdict(list)
    for _ in range(args.num_eval_steps):
        action = agent.select_action(obs, cur_g, deterministic=True)
        prev_obs = obs
        obs, _, terminations, truncations, infos = eval_envs.step(action)

        # transition the subgoal so it tracks the moving target
        cur_g = agent.subgoal_transition(prev_obs, cur_g, obs)
        steps_since += 1
        episode_end = (truncations | terminations).bool()
        reissue = (steps_since >= c) | episode_end
        if reissue.any():
            idx = reissue.nonzero(as_tuple=True)[0]
            cur_g[idx] = agent.select_subgoal(obs[idx], deterministic=True)
            steps_since[idx] = 0

        if "final_info" in infos:
            mask = infos["_final_info"]
            for k, v in infos["final_info"]["episode"].items():
                metrics[k].append(v[mask].float())

    return {k: torch.cat(v).mean().item() for k, v in metrics.items() if v}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: TrainArgs) -> str:
    import random
    from torch.utils.tensorboard import SummaryWriter

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    variant_key = normalize_subgoal_variant(args.subgoal_variant)
    run_name = args.exp_name or (
        f"hiro_{args.env_id.replace('-', '_')}_{variant_key}_"
        f"{args.robot_uids}_{args.control_mode}_c{args.c}_seed{args.seed}_{args.total_timesteps}steps"
    )
    run_dir = f"{args.output_root}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    envs, eval_envs = build_envs(args, run_name)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    low = np.asarray(envs.single_action_space.low).reshape(-1)
    high = np.asarray(envs.single_action_space.high).reshape(-1)

    sp = get_subgoal_space(args.env_id, variant_key)
    assert sp.obs_dim == obs_dim, f"subgoal space obs_dim {sp.obs_dim} != env obs_dim {obs_dim}"
    latent_subgoal_dim = args.latent_subgoal_dim if variant_key == "latent_object_centric" else 0

    cfg = HIROConfig(
        action_dim=act_dim, c=args.c, subgoal_scale=args.subgoal_scale,
        action_low=low.tolist(), action_high=high.tolist(),
        gamma_low=args.gamma_low, gamma_high=args.gamma_high, tau=args.tau,
        policy_lr=args.policy_lr, q_lr=args.q_lr, alpha_lr=args.alpha_lr,
        num_candidates=args.num_candidates,
        use_off_policy_correction=args.use_off_policy_correction,
        use_phased_reward=args.use_phased_reward, pbrs_alpha=args.pbrs_alpha,
        latent_subgoal_dim=latent_subgoal_dim,
        device=args.device,
    )
    agent = HIROAgent(sp, cfg)
    if args.use_phased_reward:
        agent.grasp_detector = get_grasp_detector(args.env_id)
        print(f"phased worker reward (PBRS): detector={'set' if agent.grasp_detector else 'NONE'} "
              f"| subgoal_variant={variant_key} "
              f"| object_dims={sp.object_positions} | pbrs_alpha={args.pbrs_alpha} "
              f"| gamma_low={args.gamma_low}")

    low_buf = LowLevelBuffer(args.low_buffer_size, args.num_envs, obs_dim, agent.subgoal_dim, act_dim,
                             storage_device=args.buffer_device, sample_device=args.device)
    high_buf = HighLevelBuffer(args.high_buffer_size, obs_dim, agent.subgoal_dim, act_dim, args.c,
                               storage_device=args.buffer_device, sample_device=args.device)
    rollout = HierarchicalRollout(agent, low_buf, high_buf, args.num_envs, device=args.device)

    # logging
    writer = SummaryWriter(run_dir)
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args), save_code=True)

    def run_eval(step: int) -> dict:
        em = evaluate(agent, eval_envs, args)
        for k, v in em.items():
            writer.add_scalar(f"eval/{k}", v, step)
        if args.track:
            import wandb
            wandb.log({f"eval/{k}": v for k, v in em.items()}, step=step)
        print(f"[step {step:>9}] eval success_once={em.get('success_once', float('nan')):.3f}  "
              f"return={em.get('return', float('nan')):.2f}")
        return em

    grad_steps = max(1, int(args.training_freq * args.utd))
    steps_per_env = max(1, args.training_freq // args.num_envs)
    high_updates = max(1, grad_steps // args.c)   # match the c-slower high data rate

    obs, _ = envs.reset(seed=args.seed)
    cur_subgoal = rollout.start(obs)
    global_step = 0
    learning_started = False
    last_eval = 0
    last_save = 0
    run_eval(0)   # baseline (random-policy) point so the curve starts at step 0

    while global_step < args.total_timesteps:
        # ---- collect rollout ----
        for _ in range(steps_per_env):
            if not learning_started:
                action = torch.rand((args.num_envs, act_dim), device=device) * 2 - 1
            else:
                action = agent.select_action(obs, cur_subgoal)

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            episode_end = (truncations | terminations).bool()
            real_next_obs = next_obs.clone()
            # bootstrap_done = 1 only on TRUE env terminations (success), so the
            # worker doesn't extrapolate Q past a real terminal state. Under
            # ignore_terminations=True (partial_reset=False) the wrapper forces
            # terminations to all-False, so this is all-zeros == "always bootstrap"
            # (matches the old behaviour). Under partial_reset=True, this stops
            # the worker from bootstrapping past success — critical when shaping
            # rewards (PBRS) charge per-step costs.
            bootstrap_done = terminations.float()
            if "final_info" in infos:
                need_final = episode_end
                real_next_obs[need_final] = infos["final_observation"][need_final]

            cur_subgoal = rollout.record(
                obs, action, real_next_obs, next_obs, reward, episode_end, bootstrap_done,
            )
            obs = next_obs
            global_step += args.num_envs

            # train-side episode metrics (env reward the policy actually collects).
            # Logged to BOTH TensorBoard and W&B so the dashboard mirrors the SAC
            # pipeline (eval/* AND train/* curves).
            if "final_info" in infos:
                ep = infos["final_info"]["episode"]
                m = infos["_final_info"]
                train_log = {}
                for k, v in ep.items():
                    train_log[f"train/{k}"] = v[m].float().mean().item()
                for tag, val in train_log.items():
                    writer.add_scalar(tag, val, global_step)
                if args.track:
                    import wandb
                    wandb.log(train_log, step=global_step)

        # ---- updates ----
        if global_step >= args.learning_starts:
            learning_started = True
            low_metrics, high_metrics = {}, {}
            for _ in range(grad_steps):
                low_metrics = agent.update_low(low_buf.sample(args.batch_size))
            if len(high_buf) >= args.batch_size:
                for _ in range(high_updates):
                    high_metrics = agent.update_high(high_buf.sample(args.batch_size))

            if global_step % 1000 < args.training_freq:
                # data diagnostics
                ls = low_buf.sample(min(args.batch_size, len(low_buf)))
                data_log = {
                    "data/mean_intrinsic_reward": ls.reward.mean().item(),
                    "data/mean_subgoal_norm": ls.subgoal.norm(dim=-1).mean().item(),
                    "data/low_buffer": len(low_buf),
                    "data/high_buffer": len(high_buf),
                }
                # grasp rate: is the worker actually grasping? (key signal for the phased reward)
                if agent.grasp_detector is not None:
                    data_log["data/grasp_rate"] = agent.grasp_detector(ls.next_obs).float().mean().item()
                if len(high_buf) > 0:
                    hs = high_buf.sample(min(args.batch_size, len(high_buf)))
                    data_log["data/mean_manager_reward"] = hs.reward_sum.mean().item()

                for k, v in low_metrics.items():
                    writer.add_scalar(f"low/{k}", v, global_step)
                for k, v in high_metrics.items():
                    writer.add_scalar(f"high/{k}", v, global_step)
                for k, v in data_log.items():
                    writer.add_scalar(k, v, global_step)
                if args.track:
                    import wandb
                    wandb.log({**{f"low/{k}": v for k, v in low_metrics.items()},
                               **{f"high/{k}": v for k, v in high_metrics.items()},
                               **data_log}, step=global_step)

        # ---- eval ----
        if global_step - last_eval >= args.eval_freq:
            last_eval = global_step
            run_eval(global_step)

        # ---- checkpoint ----
        if global_step - last_save >= args.save_freq_steps:
            last_save = global_step
            torch.save({"agent": agent.state_dict(), "step": global_step},
                       f"{run_dir}/ckpt_{global_step}.pt")

    run_eval(global_step)   # guaranteed final eval so end-of-training perf is always logged
    torch.save({"agent": agent.state_dict(), "step": global_step}, f"{run_dir}/final_ckpt.pt")
    writer.close()
    envs.close()
    eval_envs.close()
    print(f"Training complete. Artifacts in {run_dir}")
    return run_name
