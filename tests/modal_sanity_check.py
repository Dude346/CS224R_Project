"""1.0-multiplier sanity check for the weak-gripper pipeline.

Exercises the full custom-robot pipeline via robot_uids="weakpanda", but
monkey-patches WeakPanda.gripper_force_limit back to stock Panda's value
(100) BEFORE the env is built. This makes the perturbation a mathematical
no-op while still going through every line of the pipeline:
  - agent registration via @register_agent (done at weak_panda import)
  - PICK_CUBE_CONFIGS / SUPPORTED_ROBOTS membership
  - gym.make(...) lookup of "weakpanda" in REGISTERED_AGENTS
  - WeakPanda instantiation
  - _controller_configs reading self.gripper_force_limit (now monkey-patched
    to the stock value)
  - SAPIEN setting up the gripper joint actuator's force cap

Expected result: success rate should match stock-panda performance on the
same eval setup. With n=100, seed=100, EE control on StackCube, stock panda
gives ~0.12 in our prior evals — so this sanity check should land near 0.12
if the pipeline is sound. A result near 0.0 (which is what real weakpanda
gave us early on) would indicate a silent bug in the pipeline.

Does NOT modify any existing files. Reuses the image and volume from
modal_train_sac.py; reuses local checkpoint bytes by default.
"""
from __future__ import annotations

from pathlib import Path

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("sanity-check")


@app.function(
    image=image,
    gpu="T4",
    timeout=15 * 60,
    volumes={"/output": volume},
)
def sanity_eval(
    checkpoint_data: bytes,
    env_id: str = "StackCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    num_eval_envs: int = 100,
    num_eval_steps: int = 50,
    seed: int = 100,
) -> dict:
    import io
    import sys

    import numpy as np
    import torch
    import torch.nn as nn
    import gymnasium as gym

    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)
    import mani_skill.envs  # noqa: F401
    import weak_panda  # registers WeakPanda with scale=DEFAULT_WEAK_PANDA_FORCE_SCALE

    # === THE SANITY-CHECK CORE: reset the perturbation to a no-op (multiplier=1.0) ===
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam

    pre_patch_value = weak_panda.WeakPanda.gripper_force_limit
    weak_panda.WeakPanda.gripper_force_limit = Panda.gripper_force_limit
    weak_panda.WeakPandaWristCam.gripper_force_limit = PandaWristCam.gripper_force_limit
    print("=" * 70)
    print(f"MONKEY-PATCH applied BEFORE env build:")
    print(f"  WeakPanda.gripper_force_limit: {pre_patch_value} → {Panda.gripper_force_limit}")
    print(f"  WeakPandaWristCam.gripper_force_limit reset to stock as well.")
    print(f"  Multiplier is now effectively 1.0 (no actual perturbation).")
    print("=" * 70)

    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    env_kwargs["control_mode"] = control_mode
    env = gym.make(
        env_id,
        num_envs=num_eval_envs,
        reconfiguration_freq=1,
        human_render_camera_configs=dict(shader_pack="default"),
        robot_uids="weakpanda",  # custom-robot pipeline (now neutralized)
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, num_eval_envs, ignore_terminations=True, record_metrics=True
    )

    class Actor(nn.Module):
        def __init__(self, env):
            super().__init__()
            obs_dim = int(np.array(env.single_observation_space.shape).prod())
            act_dim = int(np.prod(env.single_action_space.shape))
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.fc_mean = nn.Linear(256, act_dim)
            self.fc_logstd = nn.Linear(256, act_dim)
            h = env.single_action_space.high
            low = env.single_action_space.low
            self.register_buffer("action_scale", torch.tensor((h - low) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor((h + low) / 2.0, dtype=torch.float32))

        def get_eval_action(self, x):
            h = self.backbone(x)
            mean = self.fc_mean(h)
            return torch.tanh(mean) * self.action_scale + self.action_bias

    device = "cuda"
    actor = Actor(env).to(device)
    ckpt = torch.load(io.BytesIO(checkpoint_data), map_location=device, weights_only=False)
    missing, unexpected = actor.load_state_dict(ckpt["actor"], strict=False)
    if missing or unexpected:
        print(f"WARNING: state_dict mismatch. missing={missing}, unexpected={unexpected}")
    actor.eval()

    obs, _ = env.reset(seed=seed)
    success_once_or = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
    total_return = torch.zeros(num_eval_envs, dtype=torch.float32, device=device)
    fi_success_vals: list[float] = []

    with torch.no_grad():
        for step in range(num_eval_steps):
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
            action = actor.get_eval_action(obs)
            obs, reward, terminations, truncations, info = env.step(action)
            total_return += reward.to(device) if torch.is_tensor(reward) else torch.as_tensor(reward, device=device)
            if "success" in info:
                sf = info["success"]
                if not torch.is_tensor(sf):
                    sf = torch.as_tensor(sf, device=device)
                success_once_or = success_once_or | sf.bool()
            if "final_info" in info and "episode" in info["final_info"]:
                ep = info["final_info"]["episode"]
                if "success_once" in ep:
                    v = ep["success_once"]
                    fi_success_vals.extend(v.float().cpu().tolist() if torch.is_tensor(v) else [float(v)])

    or_rate = success_once_or.float().mean().item()
    fi_rate = float(np.mean(fi_success_vals)) if fi_success_vals else None
    mean_ret = total_return.mean().item()

    print("\n" + "=" * 70)
    print("SANITY CHECK RESULT")
    print("=" * 70)
    print(f"robot_uids:                 weakpanda (gripper_force_limit monkey-patched to {Panda.gripper_force_limit})")
    print(f"env_id:                     {env_id}")
    print(f"control_mode:               {control_mode}")
    print(f"num_eval_envs:              {num_eval_envs}")
    print(f"num_eval_steps:             {num_eval_steps}")
    print(f"seed:                       {seed}")
    print(f"success_once (OR-method):   {or_rate:.4f}")
    print(f"success_once (final_info):  {fi_rate if fi_rate is None else f'{fi_rate:.4f}'}")
    print(f"mean_return:                {mean_ret:.3f}")
    print("=" * 70)
    print("INTERPRETATION:")
    print("  Compare against stock-panda result with same n/seed/env_setup (was ~0.12).")
    print("  If this is ~0.12 (within GPU-sim variance): pipeline is sound.")
    print("  If this is ~0.0 (catastrophically lower): silent bug exists in pipeline.")
    print("=" * 70)

    env.close()
    return {
        "robot_uids": "weakpanda_monkeypatched_to_1.0",
        "env_id": env_id,
        "control_mode": control_mode,
        "num_envs": num_eval_envs,
        "success_once_or": or_rate,
        "success_once_final_info": fi_rate,
        "mean_return": mean_ret,
    }


@app.local_entrypoint()
def main(
    checkpoint_path: str = "checkpoints/sac_stackcube_v1_panda_seed1_1000000_ee.pt",
    env_id: str = "StackCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    num_eval_envs: int = 100,
    num_eval_steps: int = 50,
    seed: int = 100,
):
    data = Path(checkpoint_path).read_bytes()
    print(f"Read {len(data)} bytes from {checkpoint_path}\n")
    result = sanity_eval.remote(
        checkpoint_data=data,
        env_id=env_id,
        control_mode=control_mode,
        num_eval_envs=num_eval_envs,
        num_eval_steps=num_eval_steps,
        seed=seed,
    )
    print(f"\nFinal returned result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
