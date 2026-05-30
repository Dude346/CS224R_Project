"""Fact-check: does the HIRO PickCube+panda env start at the correct robot pose,
or is it the zero-qpos / flat-T-pose bug we hit before with custom robots?

The qpos bug only ever affected CUSTOM robot uids (weakpanda, etc.) because
TableSceneBuilder.initialize hardcodes a uid check. Stock 'panda' is in that
list, so it should start at the proper keyframe. This verifies that directly.

    uv run modal run modal_check_hiro_qpos.py
"""
from __future__ import annotations

import modal

from modal_train_sac import image, volume

app = modal.App("check-hiro-qpos")


@app.function(image=image, gpu="T4", timeout=10 * 60, volumes={"/output": volume})
def check() -> dict:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.agents.robots.panda.panda import Panda

    env = gym.make(
        "PickCube-v1", num_envs=1, obs_mode="state",
        control_mode="pd_joint_delta_pos", robot_uids="panda", sim_backend="gpu",
    )
    obs, _ = env.reset(seed=1)
    qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy().flatten().tolist()

    # Panda's canonical rest keyframe (what a correctly-initialized arm looks like).
    kf = Panda.keyframes["rest"].qpos
    kf = kf.tolist() if hasattr(kf, "tolist") else list(kf)

    obs_t = obs if not hasattr(obs, "cpu") else obs.cpu()
    import numpy as np
    obs_arr = np.asarray(obs_t).flatten()

    env.close()
    max_abs = max(abs(q) for q in qpos)
    return {
        "qpos": [round(q, 4) for q in qpos],
        "expected_keyframe": [round(float(q), 4) for q in kf],
        "qpos_max_abs": round(max_abs, 4),
        "looks_like_zero_tpose": max_abs < 0.1,   # the bug signature
        "tcp_xyz_obs[19:22]": [round(float(x), 4) for x in obs_arr[19:22]],
        "cube_xyz_obs[29:32]": [round(float(x), 4) for x in obs_arr[29:32]],
    }


@app.local_entrypoint()
def main():
    r = check.remote()
    print("\n=== HIRO PickCube+panda initial qpos check ===")
    print(f"  qpos            : {r['qpos']}")
    print(f"  expected keyframe: {r['expected_keyframe']}")
    print(f"  qpos max |abs|  : {r['qpos_max_abs']}")
    print(f"  tcp_xyz (obs)   : {r['tcp_xyz_obs[19:22]']}")
    print(f"  cube_xyz (obs)  : {r['cube_xyz_obs[29:32]']}")
    print()
    if r["looks_like_zero_tpose"]:
        print("  *** RED FLAG: qpos is ~zero -> the flat-T-pose init BUG is present ***")
    else:
        print("  OK: qpos matches the panda rest keyframe -> NOT the init bug.")
