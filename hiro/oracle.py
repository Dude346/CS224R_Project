"""
hiro/oracle.py
--------------
Hardcoded "perfect manager" subgoal generators used by the worker-in-isolation
diagnostic.  Reads the env's privileged obs (positions of TCP, cube, goal) and
returns the displacement subgoal a perfectly-trained manager would output.

If the worker can solve PickCube under these oracle subgoals, HIRO's structure
is fine and the failure mode is the actual manager (or the cold-start coupling).
If it cannot, the worker SAC itself has a bug that must be found before any
more reward shaping.
"""
from __future__ import annotations

import torch


# PickCube-v1 obs layout (state mode, joint control, panda):
#   [19:22] tcp_xyz, [29:32] cube_xyz, [26:29] goal_pos (fixed per episode)
# StackCube-v1 obs layout:
#   [18:21] tcp_xyz, [25:28] cubeA_xyz, [32:35] cubeB_xyz (goal = on top of cubeB)
PICKCUBE_LAYOUT = {"tcp": (19, 22), "obj": (29, 32), "goal": (26, 29)}
STACKCUBE_LAYOUT = {"tcp": (18, 21), "obj": (25, 28), "goal": (32, 35)}


def make_oracle(env_id: str, scale: float, grasp_detector):
    """Return an oracle subgoal function: obs (B, obs_dim) -> g (B, subgoal_dim).

    Pre-grasp:  g_tcp = (cube - tcp);              g_obj = 0  (cube can't be moved anyway)
    Post-grasp: g_tcp = (cube - tcp) (track cube); g_obj = (goal - cube)

    Each component is clamped per-dim to ±scale so the magnitude matches what a
    learned manager would ever output. For StackCube we add a small +z lift to
    the goal so cubeA targets above cubeB, not into it.
    """
    if env_id == "PickCube-v1":
        layout = PICKCUBE_LAYOUT
        z_lift = 0.0
    elif env_id == "StackCube-v1":
        layout = STACKCUBE_LAYOUT
        z_lift = 0.04   # ~one cube height above cubeB so cubeA lands ON it
    else:
        raise ValueError(f"no oracle defined for env {env_id!r}")

    tcp_s, tcp_e = layout["tcp"]
    obj_s, obj_e = layout["obj"]
    goal_s, goal_e = layout["goal"]
    s = float(scale)

    @torch.no_grad()
    def oracle(obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        tcp = obs[..., tcp_s:tcp_e]
        obj = obs[..., obj_s:obj_e]
        goal = obs[..., goal_s:goal_e].clone()
        goal[..., 2] = goal[..., 2] + z_lift     # raise z target for stacking
        is_grasped = grasp_detector(obs).to(obs.dtype).unsqueeze(-1)
        g_tcp = obj - tcp                         # always aim TCP at the object
        g_obj = is_grasped * (goal - obj)         # only set object subgoal once grasped
        g = torch.cat([g_tcp, g_obj], dim=-1)     # (B, 6)
        return g.clamp(-s, s)

    return oracle
