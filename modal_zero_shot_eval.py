from __future__ import annotations

from pathlib import Path

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("zero-shot-eval")


@app.function(
    image=image,
    gpu="T4",
    timeout=10 * 60,
    volumes={"/output": volume},
)
def zero_shot_eval(
    checkpoint_data: bytes,
    env_id: str = "StackCube-v1",
    robot_uids: str = "dampedpanda",
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
    import mani_skill.envs  # noqa: F401  -- registers ManiSkill envs
    import weak_panda  # noqa: F401  -- registers WeakPanda + DampedPanda

    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    print(f"Building env: {env_id} | robot_uids={robot_uids} | control_mode={control_mode}")

    # Mirror ManiSkill SAC's training-time eval env setup byte-for-byte.
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    env_kwargs["control_mode"] = control_mode
    env = gym.make(
        env_id,
        num_envs=num_eval_envs,
        reconfiguration_freq=1,
        human_render_camera_configs=dict(shader_pack="default"),
        robot_uids=robot_uids,
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
    fi_return_vals: list[float] = []

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
                if "return" in ep:
                    v = ep["return"]
                    fi_return_vals.extend(v.float().cpu().tolist() if torch.is_tensor(v) else [float(v)])

    or_rate = success_once_or.float().mean().item()
    fi_rate = float(np.mean(fi_success_vals)) if fi_success_vals else None
    print(f"RESULT: robot_uids={robot_uids} | "
          f"success_once_or={or_rate:.4f} | "
          f"success_once_final_info={fi_rate if fi_rate is None else f'{fi_rate:.4f}'} | "
          f"mean_return={total_return.mean().item():.3f} | n={num_eval_envs}")

    env.close()
    return {
        "robot_uids": robot_uids,
        "env_id": env_id,
        "control_mode": control_mode,
        "num_episodes": num_eval_envs,
        "success_once_or": or_rate,
        "success_once_final_info": fi_rate,
        "mean_return": total_return.mean().item(),
        "fi_return": float(np.mean(fi_return_vals)) if fi_return_vals else None,
    }


@app.local_entrypoint()
def main(
    checkpoint_path: str = "checkpoints/sac_stackcube_v1_panda_seed1_1000000_ee.pt",
    env_id: str = "StackCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    num_eval_envs: int = 100,
    num_eval_steps: int = 50,
):
    data = Path(checkpoint_path).read_bytes()
    print(f"Read {len(data)} bytes from {checkpoint_path}\n")

    robot_uids_to_test = ["panda", "weakpanda", "dampedpanda"]
    summary = []

    for robot_uids in robot_uids_to_test:
        print(f"\n{'='*70}\nEvaluating robot_uids={robot_uids}\n{'='*70}")
        result = zero_shot_eval.remote(
            checkpoint_data=data,
            env_id=env_id,
            robot_uids=robot_uids,
            control_mode=control_mode,
            num_eval_envs=num_eval_envs,
            num_eval_steps=num_eval_steps,
        )
        summary.append(result)
        print(f"  → success_once_final_info={result['success_once_final_info']}, "
              f"mean_return={result['mean_return']:.3f}")

    print("\n" + "=" * 80)
    print(f"ZERO-SHOT EVAL SUMMARY ({env_id} | {control_mode} | "
          f"{num_eval_envs} eps × {num_eval_steps} steps)")
    print("=" * 80)
    print(f"{'robot_uids':<22} {'success_OR':>12} {'success_FI':>12} {'mean_return':>14}")
    print("-" * 80)
    for row in summary:
        sor = row["success_once_or"]
        sfi = row["success_once_final_info"]
        sfi_str = f"{sfi:.4f}" if sfi is not None else "n/a"
        print(f"{row['robot_uids']:<22} {sor:>12.4f} {sfi_str:>12} {row['mean_return']:>14.3f}")
    print("=" * 80)
