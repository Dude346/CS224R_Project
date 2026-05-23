"""Panda-only debug eval that mirrors ManiSkill SAC's training-time eval setup
byte-for-byte. Goal: figure out why my zero-shot panda eval got 0.12 success
when training reached 0.85 on the same checkpoint."""
from __future__ import annotations

from pathlib import Path

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("debug-eval")


@app.function(
    image=image,
    gpu="T4",
    timeout=10 * 60,
    volumes={"/output": volume},
)
def debug_eval(
    checkpoint_data: bytes,
    env_id: str = "StackCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    num_eval_envs: int = 16,
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
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    # === Step 1: inspect the checkpoint ===
    ckpt = torch.load(io.BytesIO(checkpoint_data), map_location="cuda", weights_only=False)
    print("=" * 60)
    print("STEP 1: CHECKPOINT INSPECTION")
    print("=" * 60)
    print(f"Top-level keys: {list(ckpt.keys())}")
    actor_state = ckpt["actor"]
    for k, v in actor_state.items():
        if hasattr(v, "shape"):
            print(f"  actor.{k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # === Step 2: build env mirroring SAC eval exactly ===
    print("\n" + "=" * 60)
    print("STEP 2: ENV BUILD (mirroring SAC eval byte-for-byte)")
    print("=" * 60)
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    env_kwargs["control_mode"] = control_mode
    env = gym.make(
        env_id,
        num_envs=num_eval_envs,
        reconfiguration_freq=1,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, num_eval_envs, ignore_terminations=True, record_metrics=True
    )
    print(f"obs space: {env.single_observation_space}")
    print(f"action space: {env.single_action_space}")
    print(f"action.high: {env.single_action_space.high}")
    print(f"action.low:  {env.single_action_space.low}")

    # === Step 3: build Actor and load ===
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

    print("\n" + "=" * 60)
    print("STEP 3: ACTOR LOAD")
    print("=" * 60)
    actor = Actor(env).to("cuda")
    missing, unexpected = actor.load_state_dict(ckpt["actor"], strict=False)
    print(f"missing: {missing}")
    print(f"unexpected: {unexpected}")
    print(f"action_scale: {actor.action_scale.cpu().numpy()}")
    print(f"action_bias:  {actor.action_bias.cpu().numpy()}")
    actor.eval()

    # === Step 4: rollout with full diagnostics ===
    print("\n" + "=" * 60)
    print("STEP 4: ROLLOUT")
    print("=" * 60)
    obs, info_reset = env.reset(seed=seed)
    print(f"reset obs: shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"  stats: mean={obs.mean().item():.3f} std={obs.std().item():.3f} "
          f"min={obs.min().item():.3f} max={obs.max().item():.3f}")

    success_once_or = torch.zeros(num_eval_envs, dtype=torch.bool, device="cuda")
    total_return = torch.zeros(num_eval_envs, dtype=torch.float32, device="cuda")
    final_info_success: list[float] = []
    final_info_return: list[float] = []

    with torch.no_grad():
        for step in range(num_eval_steps):
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, device="cuda", dtype=torch.float32)
            action = actor.get_eval_action(obs)
            obs, reward, terminations, truncations, info = env.step(action)
            total_return += reward.to("cuda") if torch.is_tensor(reward) else torch.as_tensor(reward, device="cuda")

            if "success" in info:
                sf = info["success"]
                if not torch.is_tensor(sf):
                    sf = torch.as_tensor(sf, device="cuda")
                success_once_or = success_once_or | sf.bool()

            if step in (0, 1, 24, 48, 49):
                print(f"[step {step:>2}] reward_mean={reward.mean().item():.3f} "
                      f"info_keys={sorted(info.keys())}")

            if "final_info" in info:
                fi = info["final_info"]
                print(f"\n[step {step}] FINAL_INFO detected; keys: {list(fi.keys())}")
                if "episode" in fi:
                    print(f"  episode keys: {list(fi['episode'].keys())}")
                    if "success_once" in fi["episode"]:
                        v = fi["episode"]["success_once"]
                        if torch.is_tensor(v):
                            print(f"  success_once: shape={tuple(v.shape)}, "
                                  f"mean={v.float().mean().item():.4f}, vals={v.tolist()}")
                            final_info_success.extend(v.float().cpu().tolist())
                        else:
                            print(f"  success_once: {v}")
                            final_info_success.append(float(v))
                    if "return" in fi["episode"]:
                        v = fi["episode"]["return"]
                        if torch.is_tensor(v):
                            print(f"  return: mean={v.float().mean().item():.3f}")
                            final_info_return.extend(v.float().cpu().tolist())

    print("\n" + "=" * 60)
    print("STEP 5: FINAL RESULTS")
    print("=" * 60)
    or_rate = success_once_or.float().mean().item()
    print(f"success_once via OR-of-info['success']:  {or_rate:.4f}")
    if final_info_success:
        fi_arr = np.array(final_info_success)
        print(f"success_once via final_info (n={len(fi_arr)}): {fi_arr.mean():.4f}")
    else:
        print("success_once via final_info: NONE COLLECTED (final_info never fired)")
    print(f"mean_return (per-step accumulation): {total_return.mean().item():.3f}")
    if final_info_return:
        print(f"mean_return via final_info (n={len(final_info_return)}): "
              f"{np.mean(final_info_return):.3f}")

    env.close()
    return {
        "or_success": or_rate,
        "fi_success": float(np.mean(final_info_success)) if final_info_success else None,
        "mean_return": total_return.mean().item(),
        "fi_return": float(np.mean(final_info_return)) if final_info_return else None,
        "n_final_info_events": len(final_info_success),
    }


@app.local_entrypoint()
def main(
    checkpoint_path: str = "checkpoints/sac_stackcube_v1_panda_seed1_1000000_ee.pt",
    env_id: str = "StackCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    seed: int = 100,
):
    data = Path(checkpoint_path).read_bytes()
    print(f"Read {len(data)} bytes from {checkpoint_path}\n")
    result = debug_eval.remote(
        checkpoint_data=data,
        env_id=env_id,
        control_mode=control_mode,
        seed=seed,
    )
    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k}: {v}")
