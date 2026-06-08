"""
modal_inspect_fetch_obs.py
--------------------------
Scope the Fetch obs/action layout for a HIRO subgoal-space mapping.
Prints obs_dim, action dim, available control modes, and locates the
cube / goal / tcp xyz triples inside the flat state observation by
value-matching against the env's ground-truth poses.

    uv run modal run modal_inspect_fetch_obs.py --env-id PickCube-v1
"""
from __future__ import annotations

import modal

from modal_train_sac import image  # reuse SAC image (CUDA+Vulkan+ManiSkill)

app = modal.App("inspect-fetch-obs")


@app.function(image=image, gpu="T4", timeout=600)
def inspect(env_id: str = "PickCube-v1", robot_uids: str = "fetch") -> dict:
    import numpy as np
    import torch
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    out: dict = {"env_id": env_id, "robot_uids": robot_uids}

    # Discover available control modes for this robot without committing to one.
    try:
        probe = gym.make(env_id, robot_uids=robot_uids, num_envs=1, sim_backend="physx_cpu")
        out["control_modes"] = list(probe.unwrapped.agent.supported_control_modes) \
            if hasattr(probe.unwrapped.agent, "supported_control_modes") \
            else list(getattr(probe.unwrapped, "SUPPORTED_CONTROL_MODES", []))
        probe.close()
    except Exception as e:
        out["control_modes_err"] = repr(e)

    # Pick joint-delta if available, else whatever the env defaults to.
    cm = "pd_joint_delta_pos"
    env = gym.make(env_id, robot_uids=robot_uids, obs_mode="state",
                   control_mode=cm, num_envs=1, sim_backend="physx_cpu")
    out["control_mode"] = env.unwrapped.control_mode
    out["action_space"] = str(env.action_space)
    obs, _ = env.reset(seed=0)
    arr = (obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs)).cpu().numpy().reshape(-1)
    out["obs_dim"] = int(arr.shape[0])

    u = env.unwrapped
    truth = {}
    for name, getter in [
        ("cube_xyz", lambda: u.cube.pose.p),
        ("cubeA_xyz", lambda: getattr(u, "cubeA", u.cube).pose.p),
        ("cubeB_xyz", lambda: getattr(u, "cubeB", u.cube).pose.p),
        ("goal_xyz", lambda: u.goal_site.pose.p),
        ("tcp_xyz", lambda: u.agent.tcp.pose.p),
    ]:
        try:
            truth[name] = getter().cpu().numpy().reshape(-1)[:3]
        except Exception:
            pass

    def find(vec):
        for i in range(len(arr) - 2):
            if np.allclose(arr[i:i + 3], vec, atol=1e-3):
                return i
        return None

    out["located_indices"] = {n: find(v) for n, v in truth.items()}
    out["truth_xyz"] = {n: v.tolist() for n, v in truth.items()}
    env.close()
    return out


@app.local_entrypoint()
def main(env_id: str = "PickCube-v1") -> None:
    import json
    print("FETCH_OBS_RESULT:", json.dumps(inspect.remote(env_id), indent=2, default=str))
