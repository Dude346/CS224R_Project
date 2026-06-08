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
from hiro.oracle import make_oracle
from hiro.subgoal_space import (
    get_grasp_detector,
    get_manager_input_dims,
    get_place_residual_dims,
    get_reach_dims,
    get_subgoal_space,
    normalize_subgoal_variant,
)


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
    num_eval_steps: int = 50
    render_size: int = 512            # human-render camera resolution (square px)
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
    subgoal_mode: str = "hiro"        # "hiro" (tcp+object) | "cube_centric" (object-only, Phase 5)
    gamma_low: float = 0.95
    gamma_high: float = 0.95           # was 0.8; low value made manager too myopic (0.8^4=0.41 at end)
    tau: float = 0.005
    use_phased_reward: bool = True
    pbrs_alpha: float = 0.5            # PBRS grasp potential strength (replaces grasp_bonus)
    worker_extrinsic_weight: float = 0.0  # hybrid: r_worker = (1-w)*intrinsic + w*env_reward
    place_pbrs_beta: float = 0.0       # Phase 2: manager place-PBRS strength (Phi=-beta*||cube-goal||); 0=off
    reach_pbrs_weight: float = 0.0     # Phase D: pre-grasp reach-to-cube PBRS bootstrap on the worker; 0=off
    # Exp 1 (curriculum anneal): hold worker_extrinsic_weight (w) until grasping bootstraps
    # (train grasp_rate > anneal_w_grasp_threshold), THEN linearly anneal w->0 over the
    # remaining budget. Tests whether the manager retains control as the env floor is removed.
    anneal_w: bool = False
    anneal_w_grasp_threshold: float = 0.4
    manager_utd_mult: int = 1          # D2: multiply manager gradient steps per cycle (more manager training)
    # Diagnostic mode: replace the manager with a hardcoded oracle that reads the
    # cube/goal positions from obs. Only the WORKER trains. Tests whether the
    # worker SAC can solve the task given perfect subgoals (isolates worker bugs
    # from manager/cold-start issues).
    oracle_manager: bool = False
    # Diagnostic mode: constant ZERO subgoal (no manager, no re-issue, no
    # transition). With worker_extrinsic_weight=1.0 this makes the worker a plain
    # flat-SAC agent inside our pipeline -- isolates "our SAC code/rollout" from
    # "the hierarchy" as the cause of any failure.
    flat_worker: bool = False
    # Worker-pretraining: path to a checkpoint to warm-start ONLY the worker
    # (the manager still trains from scratch). Collapses the HIRO cold-start by
    # starting with an already-competent subgoal-following worker.
    pretrained_worker_ckpt: str = ""
    # Resume: path to a checkpoint to load the FULL agent (worker + manager +
    # targets + temperatures) and continue training. Replay buffers are NOT
    # restored (they refill from the loaded policy). Use learning_starts=0 so the
    # trained policy collects data immediately instead of random warmup.
    resume_from_ckpt: str = ""
    # Transfer: load ONLY the manager (high level) from a checkpoint; the worker starts
    # fresh. Pair with freeze_manager to reuse a body-independent object-centric manager
    # and re-adapt only the worker on a new embodiment.
    pretrained_manager_ckpt: str = ""
    freeze_manager: bool = False      # don't train the manager (skip update_high); it still acts
    # Body-independent manager: "full" (default) = manager reads the full obs (original);
    # "object" = manager reads only [tcp,cube,goal] (9-D), so its learned weights transfer
    # across robots with different obs sizes (panda<->fetch). Enables the Option-B
    # learned-manager cross-embodiment transfer.
    manager_input_mode: str = "full"
    # Latent subgoal codec (used ONLY by the latent_object_centric variant; the
    # trainer forces it to 0 for every other variant, so this default is safe for
    # the default path).
    latent_subgoal_dim: int = 6
    # Hindsight (HER) relabeling of worker subgoals. 0.0 = off (default path); the
    # latent / object-centric runs enable it with --low-her-ratio 0.8.
    low_her_ratio: float = 0.0
    # Dense pre-grasp hand->object reach term in phased PBRS (reward-tuning branch).
    # 0.0 = off (default path preserved); the reach-fix runs set --reach-coef 1.0.
    reach_coef: float = 0.0
    reach_temp: float = 5.0            # tanh sharpness of the reach term
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

    if "panda" in args.robot_uids and not args.robot_uids.startswith("panda"):
        import weak_panda  # noqa: F401 — registers custom robots on import

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="gpu",
        control_mode=args.control_mode, robot_uids=args.robot_uids,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs,
                    reconfiguration_freq=None, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=1,
                         human_render_camera_configs=dict(shader_pack="default", width=args.render_size, height=args.render_size),
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
    # Grasp diagnostics over the GREEDY eval rollout (far more informative than the
    # buffer-wide grasp_rate, which is swamped by random exploration transitions).
    grasp_step_count = 0
    grasp_step_total = 0
    ever_grasped = torch.zeros(E, dtype=torch.bool, device=device)

    # Phase-2 campaign diagnostics. place_dims = obs indices of the cube->goal
    # vector (extra.obj_to_goal_pos = goal - cube), so ||obs[place_dims]|| is the
    # cube-to-goal distance and its direction is "where the cube should go".
    place_dims = get_place_residual_dims(args.env_id, args.robot_uids)
    place_dims_t = torch.tensor(place_dims, device=device, dtype=torch.long) if place_dims else None
    obj_pos = agent.sp.object_positions  # positions within the subgoal that target the cube
    place_dist_sum, place_dist_n = 0.0, 0
    cube_at_goal_count = 0               # steps with cube within 5 cm of goal
    align_sum, align_n = 0.0, 0          # cube-subgoal vs cube->goal cosine, grasped steps only
    for _ in range(args.num_eval_steps):
        action = agent.select_action(obs, cur_g, deterministic=True)
        prev_obs = obs

        # --- manager intent diagnostic (uses the action-time subgoal + prev_obs) ---
        if place_dims_t is not None and obj_pos:
            to_goal = prev_obs[..., place_dims_t]                 # (E,3) goal - cube
            d = torch.norm(to_goal, dim=-1)                       # (E,)
            place_dist_sum += float(d.sum().item()); place_dist_n += E
            cube_at_goal_count += int((d < 0.05).sum().item())
            g_obj = cur_g[..., obj_pos]                           # (E,3) commanded cube displacement
            if agent.grasp_detector is not None:
                gmask = agent.grasp_detector(prev_obs)
                if gmask.any():
                    cos = torch.nn.functional.cosine_similarity(
                        g_obj[gmask], to_goal[gmask], dim=-1, eps=1e-6)
                    align_sum += float(cos.sum().item()); align_n += int(gmask.sum().item())

        obs, _, terminations, truncations, infos = eval_envs.step(action)

        if agent.grasp_detector is not None:
            gd = agent.grasp_detector(obs)
            grasp_step_count += int(gd.sum().item())
            grasp_step_total += E
            ever_grasped |= gd

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

    out = {k: torch.cat(v).mean().item() for k, v in metrics.items() if v}
    if agent.grasp_detector is not None and grasp_step_total > 0:
        out["greedy_grasp_rate"] = grasp_step_count / grasp_step_total      # fraction of eval steps grasped
        out["ever_grasped"] = ever_grasped.float().mean().item()            # fraction of eval envs that ever grasped
    if place_dist_n > 0:
        out["place_dist"] = place_dist_sum / place_dist_n                   # mean ||cube-goal|| (m); lower=better
        out["cube_at_goal_rate"] = cube_at_goal_count / place_dist_n        # fraction of steps cube within 5cm of goal
    if align_n > 0:
        out["subgoal_goal_align"] = align_sum / align_n                     # cos(cube-subgoal, cube->goal) | grasped; >0 = manager commands transport
    return out


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

    # Two subgoal-space selection schemes coexist: the object/latent variants are
    # built from the obs layout (variant_key), while the geometric modes (hiro /
    # cube_centric) are routed per robot (panda / fetch).
    if variant_key in ("object_centric", "latent_object_centric"):
        sp = get_subgoal_space(args.env_id, variant_key, obs_dim=obs_dim)
    else:
        sp = get_subgoal_space(args.env_id, args.subgoal_mode, args.robot_uids)
    assert sp.obs_dim == obs_dim, f"subgoal space obs_dim {sp.obs_dim} != env obs_dim {obs_dim}"
    if args.subgoal_mode != "hiro":
        print(f"SUBGOAL MODE = {args.subgoal_mode}: {sp!r} indices={sp.indices} object_dims={sp.object_dims}")
    latent_subgoal_dim = args.latent_subgoal_dim if variant_key == "latent_object_centric" else 0

    # Body-independent manager input (Option-B cross-robot transfer). "object" =>
    # manager reads only [tcp,cube,goal] (9-D); "full" (default) => full obs.
    mgr_dims: tuple = ()
    if args.manager_input_mode == "object":
        md = get_manager_input_dims(args.env_id, args.robot_uids)
        if not md:
            raise RuntimeError(
                f"manager_input_mode='object' but no manager_input_dims for "
                f"{args.env_id!r}/{args.robot_uids!r}")
        mgr_dims = tuple(md)
        print(f"BODY-INDEPENDENT MANAGER on: input=obs[{list(mgr_dims)}] "
              f"(tcp,cube,goal 9-D) -- manager weights transfer across robots.")

    cfg = HIROConfig(
        action_dim=act_dim, c=args.c, subgoal_scale=args.subgoal_scale,
        action_low=low.tolist(), action_high=high.tolist(),
        gamma_low=args.gamma_low, gamma_high=args.gamma_high, tau=args.tau,
        policy_lr=args.policy_lr, q_lr=args.q_lr, alpha_lr=args.alpha_lr,
        num_candidates=args.num_candidates,
        use_off_policy_correction=args.use_off_policy_correction,
        use_phased_reward=args.use_phased_reward, pbrs_alpha=args.pbrs_alpha,
        worker_extrinsic_weight=args.worker_extrinsic_weight,
        place_pbrs_beta=args.place_pbrs_beta,
        place_residual_dims=tuple(get_place_residual_dims(args.env_id, args.robot_uids) or ())
            if args.place_pbrs_beta > 0.0 else (),
        reach_pbrs_weight=args.reach_pbrs_weight,
        reach_dims=tuple(get_reach_dims(args.env_id, args.robot_uids) or ())
            if args.reach_pbrs_weight > 0.0 else (),
        manager_input_dims=mgr_dims,
        low_her_ratio=args.low_her_ratio,
        latent_subgoal_dim=latent_subgoal_dim,
        reach_coef=args.reach_coef, reach_temp=args.reach_temp,
        device=args.device,
    )
    agent = HIROAgent(sp, cfg)
    if args.reach_pbrs_weight > 0.0:
        if not cfg.reach_dims:
            raise RuntimeError(f"reach_pbrs_weight>0 but no reach_dims for env {args.env_id!r}")
        print(f"WORKER reach-PBRS bootstrap on: Phi={args.reach_pbrs_weight}*(1-tanh(5*||tcp-cube||)) "
              f"(tcp->obj dims={cfg.reach_dims}, gamma_low={args.gamma_low})")
    if args.place_pbrs_beta > 0.0:
        if not cfg.place_residual_dims:
            raise RuntimeError(f"place_pbrs_beta>0 but no place_residual_dims for env {args.env_id!r}")
        print(f"MANAGER place-PBRS on: Phi=-{args.place_pbrs_beta}*||cube-goal|| "
              f"(cube->goal dims={cfg.place_residual_dims}, gamma_high={args.gamma_high})")
    if args.use_phased_reward:
        agent.grasp_detector = get_grasp_detector(args.env_id, args.robot_uids)
        print(f"phased worker reward (PBRS): detector={'set' if agent.grasp_detector else 'NONE'} "
              f"| subgoal_variant={variant_key} "
              f"| object_dims={sp.object_positions} | pbrs_alpha={args.pbrs_alpha} "
              f"| reach_coef={args.reach_coef} reach_temp={args.reach_temp} "
              f"| gamma_low={args.gamma_low}")
    if args.worker_extrinsic_weight > 0.0:
        print(f"HYBRID worker reward: r = {1-args.worker_extrinsic_weight:.2f}*intrinsic "
              f"+ {args.worker_extrinsic_weight:.2f}*env_reward")

    if args.oracle_manager:
        if agent.grasp_detector is None:
            raise RuntimeError("oracle_manager requires a grasp detector for the env")
        oracle = make_oracle(args.env_id, args.subgoal_scale, agent.grasp_detector,
                             cube_only=(args.subgoal_mode == "cube_centric"),
                             robot_uids=args.robot_uids)
        agent.select_subgoal = oracle           # monkey-patch: ignore manager_actor
        print(f"ORACLE MANAGER on -- worker-only diagnostic. Manager will NOT be trained.")

    if args.flat_worker:
        # Constant zero subgoal that never transitions => worker sees [obs, 0...0]
        # every step == plain flat SAC in our pipeline. Use with w=1.0.
        sg_dim = sp.dim
        agent.select_subgoal = lambda obs, deterministic=False: torch.zeros(
            obs.shape[0], sg_dim, device=device)
        agent.subgoal_transition = lambda s, g, s_next: torch.zeros_like(g)
        print("FLAT WORKER on -- constant zero subgoal (flat-SAC-equivalent in our pipeline).")

    if args.pretrained_worker_ckpt:
        ckpt = torch.load(args.pretrained_worker_ckpt, map_location=args.device, weights_only=False)
        agent.load_worker(ckpt["agent"])
        print(f"PRETRAINED WORKER loaded from {args.pretrained_worker_ckpt} "
              f"(checkpoint step {ckpt.get('step', '?')}). Manager starts FRESH.")

    if args.resume_from_ckpt:
        ckpt = torch.load(args.resume_from_ckpt, map_location=args.device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        print(f"RESUMED full agent (worker + manager) from {args.resume_from_ckpt} "
              f"(checkpoint step {ckpt.get('step', '?')}); training {args.total_timesteps} more steps. "
              f"Replay buffers start empty and refill from the loaded policy.")

    if args.pretrained_manager_ckpt:
        ckpt = torch.load(args.pretrained_manager_ckpt, map_location=args.device, weights_only=False)
        agent.load_manager(ckpt["agent"])
        print(f"PRETRAINED MANAGER loaded from {args.pretrained_manager_ckpt} "
              f"(checkpoint step {ckpt.get('step', '?')}). Worker starts FRESH (random init).")
    if args.freeze_manager:
        print("FREEZE MANAGER on -- manager acts but is NOT trained (update_high skipped). "
              "Transfer setup: reuse manager, adapt only the worker.")

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
        print(f"[step {step:>9}] w={agent.cfg.worker_extrinsic_weight:.2f}  "
              f"eval success_once={em.get('success_once', float('nan')):.3f}  "
              f"return={em.get('return', float('nan')):.2f}  "
              f"reward={em.get('reward', float('nan')):.3f}  "
              f"ever_grasped={em.get('ever_grasped', float('nan')):.2f}  "
              f"place_dist={em.get('place_dist', float('nan')):.3f}  "
              f"cube@goal={em.get('cube_at_goal_rate', float('nan')):.3f}  "
              f"sg_align={em.get('subgoal_goal_align', float('nan')):+.3f}")
        return em

    grad_steps = max(1, int(args.training_freq * args.utd))
    steps_per_env = max(1, args.training_freq // args.num_envs)
    high_updates = max(1, (grad_steps // args.c) * args.manager_utd_mult)   # c-slower high data rate; *mult for D2
    if args.manager_utd_mult != 1:
        print(f"MANAGER UTD x{args.manager_utd_mult}: {high_updates} manager grad-steps/cycle (worker={grad_steps})")

    obs, _ = envs.reset(seed=args.seed)
    cur_subgoal = rollout.start(obs)
    global_step = 0
    learning_started = False
    last_eval = 0
    last_save = 0
    # Exp 1 anneal state (no-op unless args.anneal_w).
    w_start = args.worker_extrinsic_weight
    anneal_started = False
    anneal_start_step = 0
    run_eval(0)   # baseline (random-policy) point so the curve starts at step 0

    while global_step < args.total_timesteps:
        # ---- Exp 1: curriculum-anneal the env-reward weight w. Hold w_start until grasping
        #      bootstraps; once triggered, linearly anneal w->0 over the remaining budget so
        #      the manager must take over transport. Mutates cfg (read by worker_reward).
        if args.anneal_w:
            if anneal_started:
                frac = max(0.0, 1.0 - (global_step - anneal_start_step) / max(1, args.total_timesteps - anneal_start_step))
                agent.cfg.worker_extrinsic_weight = w_start * frac
            else:
                agent.cfg.worker_extrinsic_weight = w_start
        # ---- collect rollout ----
        for _ in range(steps_per_env):
            if not learning_started:
                action = torch.rand((args.num_envs, act_dim), device=device) * 2 - 1
            else:
                action = agent.select_action(obs, cur_subgoal)

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            episode_end = (truncations | terminations).bool()
            real_next_obs = next_obs.clone()
            bootstrap_done = torch.zeros(args.num_envs, device=device)
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
                low_batch = low_buf.sample(args.batch_size)
                if args.low_her_ratio > 0.0:
                    her_mask = torch.rand(args.batch_size, device=device) < args.low_her_ratio
                    future_next_obs = low_buf.sample_future_next_obs(
                        low_batch.t_inds.cpu(),
                        low_batch.e_inds.cpu(),
                        low_batch.episode_id.cpu(),
                        low_batch.episode_step.cpu(),
                    ).to(device)
                    low_batch = agent.apply_low_her(low_batch, future_next_obs, her_mask)
                low_metrics = agent.update_low(low_batch)
            if (not args.oracle_manager) and (not args.flat_worker) and (not args.freeze_manager) and len(high_buf) >= args.batch_size:
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
                    "data/low_her_ratio": args.low_her_ratio,
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
                # Exp 1 (anneal_w): once grasping bootstraps, begin annealing w->0.
                grasp_rate = data_log.get("data/grasp_rate")
                if (args.anneal_w and not anneal_started and grasp_rate is not None
                        and grasp_rate > args.anneal_w_grasp_threshold):
                    anneal_started = True
                    anneal_start_step = global_step
                    print(f"[anneal_w] grasp_rate {grasp_rate:.2f} > {args.anneal_w_grasp_threshold} "
                          f"at step {global_step}: annealing w {w_start}->0 over remaining budget")
                data_log["data/worker_w"] = agent.cfg.worker_extrinsic_weight
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
