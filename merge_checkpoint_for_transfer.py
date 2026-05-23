"""Merge a SAC checkpoint trained on one task into a checkpoint shaped for a
different task (e.g., PickCube → StackCube). Layers whose shapes match are
copied; layers whose input dim differs (because obs_dim changed) keep fresh
random initialization.

The actor and Q-network architectures are reconstructed to exactly match
ManiSkill's SAC script, so the output checkpoint loads cleanly via the
script's strict=True load_state_dict.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


def build_target_actor(obs_dim: int, act_dim: int) -> nn.Module:
    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.fc_mean = nn.Linear(256, act_dim)
            self.fc_logstd = nn.Linear(256, act_dim)
            self.register_buffer("action_scale", torch.ones(act_dim, dtype=torch.float32))
            self.register_buffer("action_bias", torch.zeros(act_dim, dtype=torch.float32))

    return Actor()


def build_target_qnet(obs_dim: int, act_dim: int) -> nn.Module:
    class SoftQNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )

    return SoftQNetwork()


def merge_state_dicts(target_state: dict, source_state: dict, name: str) -> dict:
    merged: list[str] = []
    skipped: list[tuple[str, str, str]] = []
    for key, target_tensor in target_state.items():
        if key in source_state and source_state[key].shape == target_tensor.shape:
            target_state[key] = source_state[key].clone()
            merged.append(key)
        else:
            src_shape = (
                tuple(source_state[key].shape) if key in source_state else "MISSING"
            )
            skipped.append((key, str(src_shape), str(tuple(target_tensor.shape))))
    print(f"\n[{name}] merged {len(merged)} keys, skipped {len(skipped)}")
    for k in merged:
        print(f"  copy {k}")
    for k, sshape, tshape in skipped:
        print(f"  skip {k}: source={sshape}, target={tshape}  (using fresh random init)")
    return target_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source checkpoint path (e.g., PickCube panda EE)")
    parser.add_argument("--output", required=True, help="Output merged checkpoint path")
    parser.add_argument("--target-obs-dim", type=int, required=True, help="Target env obs_dim (e.g., 48 for StackCube state)")
    parser.add_argument("--target-act-dim", type=int, default=4, help="Target env act_dim (default 4 for pd_ee_delta_pos)")
    args = parser.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    assert src_path.exists(), f"Source checkpoint not found: {src_path}"

    print(f"Loading source: {src_path}")
    source = torch.load(src_path, map_location="cpu", weights_only=False)
    print(f"Source top-level keys: {list(source.keys())}")

    print(f"\nBuilding target architecture (obs_dim={args.target_obs_dim}, act_dim={args.target_act_dim})")
    target_actor = build_target_actor(args.target_obs_dim, args.target_act_dim)
    target_qf1 = build_target_qnet(args.target_obs_dim, args.target_act_dim)
    target_qf2 = build_target_qnet(args.target_obs_dim, args.target_act_dim)

    new_actor = merge_state_dicts(target_actor.state_dict(), source["actor"], "actor")
    new_qf1 = merge_state_dicts(target_qf1.state_dict(), source["qf1"], "qf1")
    new_qf2 = merge_state_dicts(target_qf2.state_dict(), source["qf2"], "qf2")

    new_ckpt = {
        "actor": new_actor,
        "qf1": new_qf1,
        "qf2": new_qf2,
        "log_alpha": source.get("log_alpha", torch.tensor([0.0], requires_grad=True)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote merged checkpoint → {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
