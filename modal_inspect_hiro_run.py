"""Dump all scalar series for a HIRO run's TB events to diagnose the collapse.

Reads /output/runs/<run_name> on the volume (CPU only, no GPU).

    uv run modal run modal_inspect_hiro_run.py
"""
from __future__ import annotations

import modal

from modal_train_sac import image, volume

app = modal.App("inspect-hiro-run")

DEFAULT_RUN = "hiro_diag_w100_oracle_pipelinetest_250k"

# Tags we care about for diagnosing a reward collapse.
KEY_TAGS = [
    "eval/success_once", "eval/reward", "eval/return",
    "eval/ever_grasped", "eval/greedy_grasp_rate", "eval/episode_len",
    "train/reward", "train/return", "train/success_once",
    "low/q_loss", "low/actor_loss", "low/alpha", "low/alpha_loss",
    "high/q_loss", "high/actor_loss", "high/alpha", "high/alpha_loss",
    "data/mean_intrinsic_reward", "data/mean_manager_reward",
    "data/mean_subgoal_norm", "data/low_buffer", "data/high_buffer",
    "data/grasp_rate",
]


@app.function(image=image, timeout=10 * 60, volumes={"/output": volume})
def inspect(run_name: str) -> dict:
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = f"/output/runs/{run_name}"
    tb_files = sorted(glob.glob(f"{run_dir}/events.out.tfevents.*"))
    if not tb_files:
        return {"error": f"no tb files under {run_dir}", "listing": glob.glob(f"{run_dir}/*")}

    ea = EventAccumulator(tb_files[-1])
    ea.Reload()
    available = sorted(ea.Tags().get("scalars", []))
    out = {"available_tags": available, "series": {}}
    for tag in KEY_TAGS:
        if tag in available:
            vals = ea.Scalars(tag)
            out["series"][tag] = [(int(v.step), float(v.value)) for v in vals]
    return out


@app.local_entrypoint()
def main(run_name: str = DEFAULT_RUN):
    out = inspect.remote(run_name)
    if "error" in out:
        print("ERROR:", out["error"])
        print("dir listing:", out.get("listing"))
        return
    print("available tags:", out["available_tags"])
    for tag, series in out["series"].items():
        if not series:
            continue
        steps = [s for s, _ in series]
        vals = [v for _, v in series]
        # sparse print: first, peak, last, plus a few midpoints
        print("\n" + "=" * 70)
        print(f"{tag}   ({len(series)} pts)")
        print(f"  first  step={steps[0]:>8}  val={vals[0]:.5f}")
        peak_i = max(range(len(vals)), key=lambda i: vals[i])
        min_i = min(range(len(vals)), key=lambda i: vals[i])
        print(f"  peak   step={steps[peak_i]:>8}  val={vals[peak_i]:.5f}")
        print(f"  min    step={steps[min_i]:>8}  val={vals[min_i]:.5f}")
        print(f"  last   step={steps[-1]:>8}  val={vals[-1]:.5f}")
        # full coarse series (every ~10th point)
        stride = max(1, len(series) // 15)
        print("  series:", "  ".join(f"{s//1000}k:{v:.3f}" for s, v in series[::stride]))
