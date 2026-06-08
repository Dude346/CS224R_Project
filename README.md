# Object-Centric Hierarchical RL for Embodiment Transfer

Stanford CS224R final project.

We study goal-conditioned hierarchical RL on ManiSkill manipulation tasks. The
agent follows HIRO — a high-level *manager* that proposes subgoals every `c`
steps and a low-level *worker* that acts to reach them, both trained with SAC —
and we compare several subgoal parameterizations (full state, object-centric,
and a learned latent) against a flat SAC baseline. The question we focus on is
whether hierarchy and object-centric subgoals buy sample efficiency or transfer
across robot embodiments (Panda, weakened/damped Panda variants, and Fetch).

Authors: Ashwin Mahendran, Anjali Sreenivas, Arianna Cao.

## Layout

```
hiro/                  HIRO implementation
  hiro_agent.py          manager/worker SAC + off-policy subgoal correction
  subgoal_space.py       subgoal parameterizations + grasp readers
  networks.py            actor/critic networks + latent subgoal codec
  replay_buffer.py       low- and high-level replay (with HER relabeling)
  rollout.py             vectorized segment bookkeeping
  trainer.py             training loop, eval, logging, checkpointing
  oracle.py              scripted manager used for diagnostics
  tests/                 unit tests
modal_train_hiro.py    HIRO training entry point (Modal)
modal_train_sac.py     flat SAC baseline entry point (Modal)
weak_panda.py          perturbed Panda embodiments for the transfer experiments
experiments/figures/   plotting scripts and figures used in the report
```

## Setup

```bash
uv sync
```

Training runs are launched on [Modal](https://modal.com) (GPU + Vulkan for
ManiSkill rendering); metrics log to Weights & Biases when `--track` is set.

## Training

Flat SAC baseline:

```bash
uv run modal run modal_train_sac.py --env-id PickCube-v1 --total-timesteps 1000000 --track
```

HIRO:

```bash
uv run modal run modal_train_hiro.py --env-id PickCube-v1 --total-timesteps 1000000 --track
```

The subgoal space is chosen with `--subgoal-mode {hiro,cube_centric}` (geometric
spaces) or `--subgoal-variant {object_centric,latent_object_centric}`. Use
`--robot-uids` together with `--manager-input-mode object` for the
cross-embodiment transfer runs.

## Tests

```bash
uv run pytest
```

## Report

Methods, experiments, and results are written up in the project report
(`CS_224R_Project_Final_Report.pdf`).
