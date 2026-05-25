"""Side-by-side comparison of robot_uids='panda' and robot_uids='weakpanda'
envs to find exactly where they diverge. With WeakPanda.gripper_force_limit
monkey-patched to stock Panda's value, the envs SHOULD be behaviorally
identical. We compare:

  - Initial observations after reset(seed=100)
  - Agent class attributes (gripper_stiffness/damping/force_limit, etc.)
  - Action space bounds
  - Initial qpos / qvel of the robot

Whichever fields diverge tell us where the custom-robot path is leaking.
"""
from __future__ import annotations

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("custom-robot-diagnostic")


@app.function(image=image, gpu="T4", timeout=10 * 60, volumes={"/output": volume})
def diagnose(env_id: str = "StackCube-v1") -> dict:
    import sys

    import numpy as np
    import torch
    import gymnasium as gym

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)
    import mani_skill.envs  # noqa: F401
    import weak_panda  # noqa: F401

    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam

    # Monkey-patch: WeakPanda's gripper_force_limit == stock Panda's
    weak_panda.WeakPanda.gripper_force_limit = Panda.gripper_force_limit
    weak_panda.WeakPandaWristCam.gripper_force_limit = PandaWristCam.gripper_force_limit
    print(f"WeakPanda.gripper_force_limit (post-patch): {weak_panda.WeakPanda.gripper_force_limit}")
    print(f"Panda.gripper_force_limit:                  {Panda.gripper_force_limit}")

    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    def build(robot_uids: str):
        env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu",
                          control_mode="pd_ee_delta_pos")
        env = gym.make(env_id, num_envs=1, reconfiguration_freq=1,
                       human_render_camera_configs=dict(shader_pack="default"),
                       robot_uids=robot_uids, **env_kwargs)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
        return env

    print("\n=== Building panda env ===")
    env_p = build("panda")
    obs_p, info_p = env_p.reset(seed=100)
    agent_p = env_p.unwrapped.agent

    print("\n=== Building weakpanda env (with monkey-patch) ===")
    env_w = build("weakpanda")
    obs_w, info_w = env_w.reset(seed=100)
    agent_w = env_w.unwrapped.agent

    result: dict = {}

    print("\n" + "=" * 70)
    print("OBSERVATION COMPARISON (after reset, seed=100)")
    print("=" * 70)
    obs_p_t = obs_p if torch.is_tensor(obs_p) else torch.as_tensor(obs_p)
    obs_w_t = obs_w if torch.is_tensor(obs_w) else torch.as_tensor(obs_w)
    print(f"panda obs:      shape={tuple(obs_p_t.shape)} mean={obs_p_t.mean().item():.4f}")
    print(f"weakpanda obs:  shape={tuple(obs_w_t.shape)} mean={obs_w_t.mean().item():.4f}")
    if obs_p_t.shape != obs_w_t.shape:
        print(f"  ❌ SHAPE MISMATCH")
        result["obs_shape_match"] = False
    else:
        diff = (obs_p_t - obs_w_t).abs()
        max_diff = diff.max().item()
        result["obs_max_abs_diff"] = max_diff
        if max_diff < 1e-6:
            print(f"  ✅ Observations are bit-equal.")
        else:
            print(f"  ❌ Observations differ. Max abs diff: {max_diff:.6f}")
            nonzero_idx = (diff > 1e-6).nonzero(as_tuple=False)
            print(f"  Indices where they differ: {nonzero_idx.flatten().tolist()[:30]}")
            for idx in nonzero_idx.flatten().tolist()[:10]:
                idx = int(idx)
                print(f"    [{idx}]: panda={obs_p_t.flatten()[idx].item():.4f}, "
                      f"weakpanda={obs_w_t.flatten()[idx].item():.4f}")

    print("\n" + "=" * 70)
    print("AGENT CLASS / INSTANCE COMPARISON")
    print("=" * 70)
    print(f"panda agent type:       {type(agent_p).__name__}")
    print(f"weakpanda agent type:   {type(agent_w).__name__}")
    for attr in ["arm_stiffness", "arm_damping", "arm_force_limit",
                 "gripper_stiffness", "gripper_damping", "gripper_force_limit"]:
        vp = getattr(agent_p, attr, "MISSING")
        vw = getattr(agent_w, attr, "MISSING")
        match = "✅" if vp == vw else "❌"
        print(f"  {match} {attr}: panda={vp} | weakpanda={vw}")
        result[f"attr_{attr}_match"] = vp == vw

    print("\n" + "=" * 70)
    print("ACTION SPACE COMPARISON")
    print("=" * 70)
    print(f"panda action space:     {env_p.single_action_space}")
    print(f"weakpanda action space: {env_w.single_action_space}")

    print("\n" + "=" * 70)
    print("INITIAL ROBOT QPOS COMPARISON")
    print("=" * 70)
    qpos_p = agent_p.robot.get_qpos()
    qpos_w = agent_w.robot.get_qpos()
    print(f"panda qpos:     {qpos_p.cpu().numpy().flatten().tolist()}")
    print(f"weakpanda qpos: {qpos_w.cpu().numpy().flatten().tolist()}")
    qpos_diff = (qpos_p - qpos_w).abs().max().item()
    print(f"max abs diff: {qpos_diff:.6f}")
    result["qpos_max_abs_diff"] = qpos_diff

    print("\n" + "=" * 70)
    print("BUILD & STEP 1 ACTION")
    print("=" * 70)
    # Apply same deterministic action: zeros
    a = torch.zeros(env_p.single_action_space.shape, device="cuda").unsqueeze(0)
    obs_p2, rew_p, _, _, _ = env_p.step(a)
    obs_w2, rew_w, _, _, _ = env_w.step(a)
    obs_diff2 = ((obs_p2 if torch.is_tensor(obs_p2) else torch.as_tensor(obs_p2))
                 - (obs_w2 if torch.is_tensor(obs_w2) else torch.as_tensor(obs_w2))).abs().max().item()
    rew_p_v = rew_p.item() if torch.is_tensor(rew_p) else float(rew_p)
    rew_w_v = rew_w.item() if torch.is_tensor(rew_w) else float(rew_w)
    print(f"After identical zero-action step:")
    print(f"  panda reward:     {rew_p_v:.4f}")
    print(f"  weakpanda reward: {rew_w_v:.4f}")
    print(f"  obs max abs diff: {obs_diff2:.6f}")
    result["step1_reward_panda"] = rew_p_v
    result["step1_reward_weakpanda"] = rew_w_v
    result["step1_obs_max_abs_diff"] = obs_diff2

    env_p.close()
    env_w.close()
    return result


@app.local_entrypoint()
def main(env_id: str = "StackCube-v1"):
    result = diagnose.remote(env_id=env_id)
    print("\n\n=== DIAGNOSTIC SUMMARY ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
