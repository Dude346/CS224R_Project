"""
hiro/verify_subgoal_space.py
-----------------------------
Verifies that the hardcoded subgoal indices in HIRO_SUBGOAL_SPACES are correct
for the actual ManiSkill3 environments.  Must be run before building anything
on top of these indices.

Run via Modal (ManiSkill needs a GPU container):
    uv run modal run modal_verify_subgoal_space.py

Three assertions per env:
  A1 — Each pose block is [xyz, quaternion]: the quaternion slice has L2 norm ≈ 1.
  A2 — tcp_pose starts at the expected index (19 for PickCube, 18 for StackCube).
  A3 — The chosen subgoal_indices select position dims, not quaternion dims:
         confirmed by (a) checking no index falls in a quaternion block, and
         (b) printing the selected values so they can be visually confirmed as
         plausible xyz coordinates (not unit-norm quaternion components).
"""
from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym

from hiro.subgoal_space import HIRO_SUBGOAL_SPACES


def _to_numpy(obs) -> np.ndarray:
    """Squeeze leading num_envs=1 batch dim and return flat float32 array."""
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    arr = np.asarray(obs, dtype=np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.reshape(-1)


def _assert_quat_norm(obs: np.ndarray, start: int, label: str) -> None:
    """Assert obs[start:start+4] is a unit quaternion."""
    q = obs[start : start + 4]
    norm = np.linalg.norm(q)
    print(f"     {label} quat obs[{start}:{start+4}] = {q}  norm={norm:.5f}")
    assert abs(norm - 1.0) < 0.01, (
        f"FAIL: {label} quaternion norm = {norm:.5f}, expected ≈ 1.0"
    )


def verify_pick_cube(sim_backend: str = "physx_cpu") -> None:
    env_id = "PickCube-v1"
    space  = HIRO_SUBGOAL_SPACES[env_id]

    env = gym.make(
        env_id, obs_mode="state", control_mode="pd_joint_delta_pos",
        robot_uids="panda", sim_backend=sim_backend, num_envs=1,
    )
    obs, _ = env.reset(seed=42)
    s = _to_numpy(obs)
    env.close()

    print(f"\n{'='*60}")
    print(f"  {env_id}  —  obs_dim={len(s)},  subgoal={space}")
    print(f"{'='*60}")
    assert len(s) == space.obs_dim, (
        f"obs_dim mismatch: got {len(s)}, expected {space.obs_dim}"
    )

    # ------------------------------------------------------------------
    # A1: Each pose block is [xyz (3), quaternion (4)].
    #     Verify by checking quaternion slice norm ≈ 1.
    #     tcp_pose  : obs[19:26]  ->  xyz=[19:22], quat=[22:26]
    #     obj_pose  : obs[29:36]  ->  xyz=[29:32], quat=[32:36]
    # ------------------------------------------------------------------
    print("\n[A1] Pose block structure: each block is [xyz(3), quat(4)]")
    _assert_quat_norm(s, 22, "tcp_pose")
    _assert_quat_norm(s, 32, "obj_pose ")
    print("     PASS")

    # ------------------------------------------------------------------
    # A2: tcp_pose starts at index 19.
    #     obs[18] = is_grasped (scalar, should be binary).
    #     obs[19:22] = tcp xyz (plausible EE position near the table).
    # ------------------------------------------------------------------
    print("\n[A2] tcp_pose starts at index 19  (obs[18] = is_grasped)")
    is_grasped = s[18]
    tcp_xyz    = s[19:22]
    print(f"     obs[18] = {is_grasped:.4f}  (is_grasped; 0=open, 1=grasping)")
    print(f"     obs[19:22] = {tcp_xyz}  (tcp xyz)")
    assert is_grasped in (0.0, 1.0) or abs(is_grasped - round(is_grasped)) < 0.01, (
        f"FAIL: obs[18]={is_grasped} does not look like a binary is_grasped flag"
    )
    print("     PASS")

    # ------------------------------------------------------------------
    # A3: subgoal_indices=[19,20,21, 29,30,31] select xyz, not quaternions.
    #     (a) No index falls in a quaternion block.
    #     (b) Print selected values — confirm they look like xyz coordinates.
    # ------------------------------------------------------------------
    print(f"\n[A3] Subgoal indices {space.indices} select xyz only (no quaternion dims)")
    quat_blocks = set(range(22, 26)) | set(range(32, 36))
    for idx in space.indices:
        assert idx not in quat_blocks, (
            f"FAIL: index {idx} falls inside a quaternion block {quat_blocks}"
        )
    subgoal_vals = s[space.indices]
    print(f"     tcp_xyz  obs[19:22] = {s[19:22]}")
    print(f"     cube_xyz obs[29:32] = {s[29:32]}")
    print(f"     subgoal values      = {subgoal_vals}")
    print("     (Values should be plausible Cartesian coordinates, not unit-norm)")
    print("     PASS")

    print(f"\n✓  All assertions passed for {env_id}\n")


def verify_stack_cube(sim_backend: str = "physx_cpu") -> None:
    env_id = "StackCube-v1"
    space  = HIRO_SUBGOAL_SPACES[env_id]

    env = gym.make(
        env_id, obs_mode="state", control_mode="pd_joint_delta_pos",
        robot_uids="panda", sim_backend=sim_backend, num_envs=1,
    )
    obs, _ = env.reset(seed=42)
    s = _to_numpy(obs)
    env.close()

    print(f"\n{'='*60}")
    print(f"  {env_id}  —  obs_dim={len(s)},  subgoal={space}")
    print(f"{'='*60}")
    assert len(s) == space.obs_dim, (
        f"obs_dim mismatch: got {len(s)}, expected {space.obs_dim}"
    )

    # ------------------------------------------------------------------
    # A1: Three pose blocks, each [xyz(3), quat(4)].
    #     tcp_pose  : obs[18:25] -> xyz=[18:21], quat=[21:25]
    #     cubeA_pose: obs[25:32] -> xyz=[25:28], quat=[28:32]
    #     cubeB_pose: obs[32:39] -> xyz=[32:35], quat=[35:39]
    # ------------------------------------------------------------------
    print("\n[A1] Pose block structure: each block is [xyz(3), quat(4)]")
    _assert_quat_norm(s, 21, "tcp_pose  ")
    _assert_quat_norm(s, 28, "cubeA_pose")
    _assert_quat_norm(s, 35, "cubeB_pose")
    print("     PASS")

    # ------------------------------------------------------------------
    # A2: tcp_pose starts at index 18 (no is_grasped field in StackCube).
    # ------------------------------------------------------------------
    print("\n[A2] tcp_pose starts at index 18  (no is_grasped in StackCube)")
    tcp_xyz = s[18:21]
    print(f"     obs[18:21] = {tcp_xyz}  (tcp xyz)")
    print("     (Quaternion norm check in A1 confirms block starts at 18)")
    print("     PASS")

    # ------------------------------------------------------------------
    # A3: subgoal_indices=[18,19,20, 25,26,27, 32,33,34] select xyz only.
    # ------------------------------------------------------------------
    print(f"\n[A3] Subgoal indices {space.indices} select xyz only (no quaternion dims)")
    quat_blocks = set(range(21, 25)) | set(range(28, 32)) | set(range(35, 39))
    for idx in space.indices:
        assert idx not in quat_blocks, (
            f"FAIL: index {idx} falls inside a quaternion block {quat_blocks}"
        )
    subgoal_vals = s[space.indices]
    print(f"     tcp_xyz   obs[18:21] = {s[18:21]}")
    print(f"     cubeA_xyz obs[25:28] = {s[25:28]}")
    print(f"     cubeB_xyz obs[32:35] = {s[32:35]}")
    print(f"     subgoal values       = {subgoal_vals}")
    print("     (Values should be plausible Cartesian coordinates, not unit-norm)")
    print("     PASS")

    print(f"\n✓  All assertions passed for {env_id}\n")
