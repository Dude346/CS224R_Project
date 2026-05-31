"""
hiro/tests/test_oracle.py
--------------------------
Unit tests for hiro/oracle.py — confirm the oracle returns physically sensible
subgoals before we use it for the worker-in-isolation diagnostic.
"""
from __future__ import annotations

import pytest
import torch

from hiro.oracle import make_oracle
from hiro.subgoal_space import GRASP_DETECTORS


SCALE = 0.15


def _build_pickcube_obs(B, tcp_xyz, cube_xyz, goal_xyz, cube_z=None, is_grasped=False):
    """Make a synthetic PickCube obs with controlled positions."""
    obs = torch.zeros(B, 42)
    obs[:, 19:22] = torch.tensor(tcp_xyz)
    obs[:, 29:32] = torch.tensor(cube_xyz)
    obs[:, 26:29] = torch.tensor(goal_xyz)
    if cube_z is not None:
        obs[:, 31] = cube_z
    # set tcp_to_obj = cube - tcp (used by the positional detector / oracle geometry)
    obs[:, 36:39] = obs[:, 29:32] - obs[:, 19:22]
    # PickCube grasp detection now reads the TRUE env grasp bit at obs[18].
    if is_grasped:
        obs[:, 18] = 1.0
    return obs


class TestPickCubeOracle:
    def test_pregrasp_aims_tcp_at_cube_no_cube_subgoal(self):
        det = GRASP_DETECTORS["PickCube-v1"]
        oracle = make_oracle("PickCube-v1", SCALE, det)

        # TCP 0.30m away from cube on x; cube on the table (z=0.02 < lift threshold)
        obs = _build_pickcube_obs(1, [0.0, 0.0, 0.2], [0.30, 0.0, 0.02], [0.20, 0.0, 0.15], cube_z=0.02)
        # Confirm we're in the "not grasped" branch
        assert bool(det(obs)[0]) is False

        g = oracle(obs)
        assert g.shape == (1, 6)
        # g_tcp = clip(cube - tcp) = clip([0.3, 0, -0.18]) per-dim to ±0.15 => [0.15, 0, -0.15]
        assert torch.allclose(g[0, :3], torch.tensor([0.15, 0.0, -0.15]))
        # g_cube must be 0 because is_grasped = False
        assert torch.allclose(g[0, 3:], torch.zeros(3))

    def test_postgrasp_sets_cube_subgoal_toward_goal(self):
        det = GRASP_DETECTORS["PickCube-v1"]
        oracle = make_oracle("PickCube-v1", SCALE, det)

        # is_grasped = True via the TRUE env grasp bit obs[18] (ObsBitGraspReader).
        # cube at (0.1, 0, 0.10); goal at (0.20, 0, 0.20)
        obs = torch.zeros(1, 42)
        obs[0, 19:22] = torch.tensor([0.10, 0.0, 0.10])
        obs[0, 29:32] = torch.tensor([0.10, 0.0, 0.10])
        obs[0, 26:29] = torch.tensor([0.20, 0.0, 0.20])
        obs[0, 36:39] = obs[0, 29:32] - obs[0, 19:22]   # TCP at cube
        obs[0, 18] = 1.0                                 # true grasp bit set
        assert bool(det(obs)[0]) is True

        g = oracle(obs)
        # g_tcp = clip(cube - tcp) = 0 (TCP is already at cube)
        assert torch.allclose(g[0, :3], torch.zeros(3))
        # g_cube = clip(goal - cube) = clip([0.10, 0, 0.10]) per-dim to ±0.15 => unchanged
        assert torch.allclose(g[0, 3:], torch.tensor([0.10, 0.0, 0.10]))

    def test_clamping_to_scale(self):
        det = GRASP_DETECTORS["PickCube-v1"]
        oracle = make_oracle("PickCube-v1", SCALE, det)

        # Huge displacement: should saturate at ±SCALE per dim
        obs = _build_pickcube_obs(1, [0.0, 0.0, 0.2], [1.0, -1.0, 0.02], [0.20, 0.0, 0.15], cube_z=0.02)
        g = oracle(obs)
        assert torch.all(g[0, :3].abs() <= SCALE + 1e-6)
        assert torch.all(g[0, 3:].abs() <= SCALE + 1e-6)
        # Direction preserved
        assert g[0, 0] == SCALE              # +x (cube to the right of tcp)
        assert g[0, 1] == -SCALE             # -y (cube to the left)

    def test_batched(self):
        det = GRASP_DETECTORS["PickCube-v1"]
        oracle = make_oracle("PickCube-v1", SCALE, det)
        obs = torch.zeros(4, 42)
        # Different positions per env
        obs[:, 19:22] = torch.tensor([[0.0, 0, 0.2]] * 4)
        obs[:, 29:32] = torch.tensor([[0.1, 0, 0.02], [-0.1, 0, 0.02],
                                       [0.0, 0.1, 0.02], [0.0, -0.1, 0.02]])
        obs[:, 26:29] = torch.tensor([[0.2, 0, 0.15]] * 4)
        obs[:, 36:39] = obs[:, 29:32] - obs[:, 19:22]
        obs[:, 31] = 0.02
        g = oracle(obs)
        assert g.shape == (4, 6)
        # All pre-grasp so cube subgoal is zero across batch
        assert torch.allclose(g[:, 3:], torch.zeros(4, 3))
        # tcp subgoal points in different directions per env
        assert torch.all(g[:, :3].abs().sum(dim=-1) > 0)
