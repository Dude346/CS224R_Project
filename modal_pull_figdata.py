"""
Pull logged scalar series for the paper figures from the Modal volume (CPU, no GPU).
Reads each run's latest TB events file and dumps the requested tags as JSON locally.

    uv run modal run modal_pull_figdata.py --runs "<r1>,<r2>" --tags "<t1>,<t2>"
"""
from __future__ import annotations
import modal
from modal_train_sac import image, volume

app = modal.App("pull-figdata")


@app.function(image=image, timeout=15 * 60, volumes={"/output": volume})
def pull(run_names: list[str], tags: list[str]) -> dict:
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out: dict = {}
    for rn in run_names:
        run_dir = f"/output/runs/{rn}"
        tb = sorted(glob.glob(f"{run_dir}/events.out.tfevents.*"))
        if not tb:
            out[rn] = {"_error": f"no tb under {run_dir}"}
            continue
        ea = EventAccumulator(tb[-1]); ea.Reload()
        avail = sorted(ea.Tags().get("scalars", []))
        rec = {"_available": avail}
        for t in tags:
            if t in avail:
                rec[t] = [(int(v.step), float(v.value)) for v in ea.Scalars(t)]
        out[rn] = rec
    return out


@app.local_entrypoint()
def main(runs: str, tags: str) -> None:
    import json, os
    out = pull.remote(runs.split(","), tags.split(","))
    os.makedirs("experiments/figures", exist_ok=True)
    with open("experiments/figures/figdata.json", "w") as f:
        json.dump(out, f)
    for rn, d in out.items():
        if "_error" in d:
            print(f"{rn}: ERROR {d['_error']}")
        else:
            pulled = [k for k in d if not k.startswith("_")]
            print(f"{rn}: {len(d['_available'])} tags avail | pulled {pulled}")
