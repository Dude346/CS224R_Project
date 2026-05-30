"""Dump StackCube + weakdampedpanda raw eval/success_once curves.
Compares from-scratch vs fine-tune side-by-side."""
from __future__ import annotations

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("inspect-weakdamped")

RUNS = [
    ("StackCube weakdampedpanda FROMSCRATCH",
     "sac_StackCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKDAMPED_force075_damping15_fromscratch"),
    ("StackCube weakdampedpanda FINETUNE",
     "ft_sac_StackCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKDAMPED_force075_damping15_finetune"),
]


@app.function(image=image, timeout=10 * 60, volumes={"/output": volume})
def inspect() -> dict:
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for label, run_name in RUNS:
        run_dir = f"/output/runs/{run_name}"
        tb_files = sorted(glob.glob(f"{run_dir}/events.out.tfevents.*"))
        if not tb_files:
            out[label] = {"error": "no tb files"}
            continue
        ea = EventAccumulator(tb_files[-1])
        ea.Reload()
        vals = ea.Scalars("eval/success_once")
        out[label] = {"series": [(int(v.step), float(v.value)) for v in vals]}
    return out


@app.local_entrypoint()
def main():
    out = inspect.remote()
    for label, info in out.items():
        print("\n" + "=" * 90)
        print(label)
        print("=" * 90)
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue
        for step, val in info["series"]:
            bar = "█" * int(val * 40)
            print(f"  step={step:>9}   value={val:.4f}   {bar}")
