"""
inspect_env.py
--------------
Step 1 of the HIRO implementation: understand the observation and action spaces
for PickCube-v1 and StackCube-v1 before writing any policy or buffer code.

Run via the Modal wrapper (macOS Vulkan doesn't work locally):
    uv run modal run modal_inspect_hiro_env.py

What this prints for each task + control mode:
  - The raw dict observation from env.unwrapped.get_obs():
      every named field (e.g. agent.qpos, extra.tcp_pose),
      its shape, and exactly which slice of the flat vector it occupies
  - Value ranges (min / mean / max) over a short random rollout
  - The action space: total dim and per-dimension bounds
"""
from __future__ import annotations

import sys
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers ManiSkill tasks with gymnasium
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers: walk a nested dict observation (from env.unwrapped.get_obs())
# ---------------------------------------------------------------------------

def _to_numpy_1d(val: Any) -> np.ndarray:
    """Convert a tensor/array (possibly with a leading num_envs=1 dim) to 1-D float32."""
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]          # strip the num_envs=1 batch dim
    return arr.reshape(-1)


def _walk_dict(node: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    """
    Recursively walk a nested dict observation.
    Returns list of (dotted_key, 1d_numpy_array).
    """
    entries = []
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{prefix}.{k}" if prefix else k
            entries.extend(_walk_dict(v, prefix=child))
    else:
        entries.append((prefix, _to_numpy_1d(node)))
    return entries


def _flatten_dict_obs(obs: Any) -> np.ndarray:
    """Flatten any nested-dict or tensor obs to a 1-D float32 vector."""
    if isinstance(obs, dict):
        parts = [_flatten_dict_obs(v) for v in obs.values()]
        return np.concatenate(parts)
    return _to_numpy_1d(obs)


# ---------------------------------------------------------------------------
# Main inspection routine
# ---------------------------------------------------------------------------

def inspect(
    env_id: str,
    control_mode: str,
    n_random_steps: int = 50,
    sim_backend: str = "physx_cpu",
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ENV: {env_id}   |   CONTROL MODE: {control_mode}   |   SIM: {sim_backend}")
    print(sep)

    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode=control_mode,
        robot_uids="panda",
        sim_backend=sim_backend,
        num_envs=1,
    )
    env.reset(seed=0)

    # ------------------------------------------------------------------
    # 1. Get named field structure via env.unwrapped.get_obs().
    #    The top-level env wraps everything into a flat Box, but the
    #    unwrapped ManiSkill env returns a proper nested dict.
    # ------------------------------------------------------------------
    print("\n--- Observation structure (from env.unwrapped.get_obs()) ---")
    raw_obs = env.unwrapped.get_obs()

    if not isinstance(raw_obs, dict):
        print(f"  [note] unwrapped obs is not a dict — got {type(raw_obs).__name__}, shape={np.asarray(raw_obs).shape}")
        print(f"  Flat dim: {_to_numpy_1d(raw_obs).shape[0]}")
    else:
        leaves = _walk_dict(raw_obs)
        offset = 0
        slice_map: list[tuple[str, int, int]] = []
        for key, arr in leaves:
            n = arr.shape[0]
            slice_map.append((key, offset, offset + n))
            print(f"  {key:<50}  shape=({n},)   flat_slice=[{offset}:{offset+n}]")
            offset += n
        print(f"\n  >> Total flattened observation dim: {offset}")

        # ------------------------------------------------------------------
        # 2. Value ranges from a short random rollout.
        # ------------------------------------------------------------------
        print("\n--- Observed value ranges (min / mean / max over random rollout) ---")
        flat_samples: list[np.ndarray] = []
        for _ in range(n_random_steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            sample = _flatten_dict_obs(env.unwrapped.get_obs())
            flat_samples.append(sample)
            if terminated or truncated:
                env.reset()

        arr_all = np.stack(flat_samples, axis=0)  # (n_steps, obs_dim)

        for key, start, end in slice_map:
            chunk = arr_all[:, start:end]          # (n_steps, field_dim)
            mn   = chunk.min(axis=0)
            mx   = chunk.max(axis=0)
            mean = chunk.mean(axis=0)
            n_dims = end - start
            if n_dims <= 4:
                for i in range(n_dims):
                    print(f"  {key}[{i}]  min={mn[i]:+.4f}  mean={mean[i]:+.4f}  max={mx[i]:+.4f}")
            else:
                print(
                    f"  {key:<50}  "
                    f"min={mn.min():+.4f}  mean={mean.mean():+.4f}  max={mx.max():+.4f}"
                    f"  ({n_dims} dims)"
                )

    # ------------------------------------------------------------------
    # 3. Action space.
    # ------------------------------------------------------------------
    print("\n--- Action space ---")
    aspace = env.action_space
    low  = np.asarray(aspace.low ).reshape(-1)
    high = np.asarray(aspace.high).reshape(-1)
    print(f"  total action dim : {low.shape[0]}")
    print(f"  dtype            : {aspace.dtype}")
    for i, (lo, hi) in enumerate(zip(low, high)):
        print(f"  action[{i}]  low={lo:+.4f}  high={hi:+.4f}")
    print()

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks         = ["PickCube-v1", "StackCube-v1"]
    control_modes = ["pd_joint_delta_pos", "pd_ee_delta_pos"]
    sim_backend   = "physx_cpu"

    if len(sys.argv) > 1:
        task_args = [a for a in sys.argv[1:] if "-v" in a]
        mode_args = [a for a in sys.argv[1:] if "pd_" in a]
        sim_args  = [a for a in sys.argv[1:] if "physx" in a or a == "gpu"]
        if task_args:   tasks         = task_args
        if mode_args:   control_modes = mode_args
        if sim_args:    sim_backend   = sim_args[0]

    for task in tasks:
        for mode in control_modes:
            try:
                inspect(task, mode, sim_backend=sim_backend)
            except Exception as exc:
                print(f"\n[SKIP] {task} / {mode}: {exc}", file=sys.stderr)
