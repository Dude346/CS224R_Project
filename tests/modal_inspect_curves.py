"""Dump the raw eval/success_once series for a few runs to verify what the
TB events actually contain. The user reports the W&B plots look similar
between scratch and finetune; my earlier analysis claimed fine-tune
reached 0.9 at step 0. One of those is wrong — let's see the raw numbers.
"""
from __future__ import annotations

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("inspect-curves")

RUNS_TO_INSPECT = [
    ("PickCube weakpanda fromscratch",
     "sac_PickCube_v1_weakpanda_pd_joint_delta_pos_seed1_500000steps_WEAKGRIPPER_force075_fromscratch"),
    ("PickCube weakpanda finetune",
     "ft_sac_PickCube_v1_weakpanda_pd_joint_delta_pos_seed1_500000steps_WEAKGRIPPER_force075_finetune"),
    ("StackCube weakpanda fromscratch",
     "sac_StackCube_v1_weakpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKGRIPPER_force075_fromscratch"),
    ("StackCube weakpanda finetune",
     "ft_sac_StackCube_v1_weakpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKGRIPPER_force075_finetune"),
    ("StackCube dampedpanda fromscratch",
     "sac_StackCube_v1_dampedpanda_pd_joint_delta_pos_seed1_1000000steps_DAMPED_damping15_fromscratch"),
    ("StackCube dampedpanda finetune",
     "ft_sac_StackCube_v1_dampedpanda_pd_joint_delta_pos_seed1_1000000steps_DAMPED_damping15_finetune"),
]


@app.function(image=image, timeout=10 * 60, volumes={"/output": volume})
def inspect() -> dict:
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for label, run_name in RUNS_TO_INSPECT:
        run_dir = f"/output/runs/{run_name}"
        tb_files = sorted(glob.glob(f"{run_dir}/events.out.tfevents.*"))
        if not tb_files:
            out[label] = {"error": "no_tb_files", "run_dir": run_dir}
            continue
        ea = EventAccumulator(tb_files[-1])
        ea.Reload()
        tags = sorted(ea.Tags().get("scalars", []))
        # Find the eval-success tag
        success_tag = None
        for cand in ["eval/success_once", "eval/success_once_mean", "charts/eval_success_once"]:
            if cand in tags:
                success_tag = cand
                break
        if success_tag is None:
            for t in tags:
                if "success_once" in t.lower():
                    success_tag = t
                    break
        info = {
            "all_tags": tags,
            "selected_tag": success_tag,
        }
        if success_tag:
            vals = ea.Scalars(success_tag)
            info["series"] = [(int(v.step), float(v.value)) for v in vals]
        out[label] = info
    return out


@app.local_entrypoint()
def main():
    out = inspect.remote()
    for label, info in out.items():
        print("\n" + "=" * 90)
        print(label)
        print("=" * 90)
        if "error" in info:
            print(f"  ERROR: {info['error']} (dir: {info.get('run_dir')})")
            continue
        print(f"  tags found: {info['all_tags'][:8]}{' ...' if len(info['all_tags']) > 8 else ''}")
        print(f"  using tag:  {info['selected_tag']}")
        print(f"  series ({len(info['series'])} points):")
        for step, val in info["series"]:
            bar = "█" * int(val * 40)
            print(f"    step={step:>9}   value={val:.4f}   {bar}")
