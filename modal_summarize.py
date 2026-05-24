"""Summarize the 12 perturbation-matrix runs from TB events on the volume.

Focus: SAMPLE EFFICIENCY — first step at which eval/success_once crosses
0.5 / 0.75 / 0.9 thresholds. If fine-tune is helping, those thresholds
should be reached at EARLIER steps than from-scratch.
"""
from __future__ import annotations

import modal

from modal_train_sac import image, volume, REMOTE_ROOT

app = modal.App("summarize")

RUN_NAMES = [
    ("PickCube", "weakpanda", "fromscratch",
     "sac_PickCube_v1_weakpanda_pd_joint_delta_pos_seed1_500000steps_WEAKGRIPPER_force075_fromscratch"),
    ("PickCube", "dampedpanda", "fromscratch",
     "sac_PickCube_v1_dampedpanda_pd_joint_delta_pos_seed1_500000steps_DAMPED_damping15_fromscratch"),
    ("PickCube", "weakdampedpanda", "fromscratch",
     "sac_PickCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_500000steps_WEAKDAMPED_force075_damping15_fromscratch"),
    ("StackCube", "weakpanda", "fromscratch",
     "sac_StackCube_v1_weakpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKGRIPPER_force075_fromscratch"),
    ("StackCube", "dampedpanda", "fromscratch",
     "sac_StackCube_v1_dampedpanda_pd_joint_delta_pos_seed1_1000000steps_DAMPED_damping15_fromscratch"),
    ("StackCube", "weakdampedpanda", "fromscratch",
     "sac_StackCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKDAMPED_force075_damping15_fromscratch"),
    ("PickCube", "weakpanda", "finetune",
     "ft_sac_PickCube_v1_weakpanda_pd_joint_delta_pos_seed1_500000steps_WEAKGRIPPER_force075_finetune"),
    ("PickCube", "dampedpanda", "finetune",
     "ft_sac_PickCube_v1_dampedpanda_pd_joint_delta_pos_seed1_500000steps_DAMPED_damping15_finetune"),
    ("PickCube", "weakdampedpanda", "finetune",
     "ft_sac_PickCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_500000steps_WEAKDAMPED_force075_damping15_finetune"),
    ("StackCube", "weakpanda", "finetune",
     "ft_sac_StackCube_v1_weakpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKGRIPPER_force075_finetune"),
    ("StackCube", "dampedpanda", "finetune",
     "ft_sac_StackCube_v1_dampedpanda_pd_joint_delta_pos_seed1_1000000steps_DAMPED_damping15_finetune"),
    ("StackCube", "weakdampedpanda", "finetune",
     "ft_sac_StackCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_1000000steps_WEAKDAMPED_force075_damping15_finetune"),
]

THRESHOLDS = [0.5, 0.75, 0.9]


def first_step_above(scalars, threshold):
    """Return the step at which the metric first crosses `threshold`, or None."""
    for s in scalars:
        if s.value >= threshold:
            return int(s.step)
    return None


@app.function(image=image, timeout=10 * 60, volumes={"/output": volume})
def summarize() -> list[dict]:
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    results = []
    for env, robot, mode, run_name in RUN_NAMES:
        run_dir = f"/output/runs/{run_name}"
        tb_files = sorted(glob.glob(f"{run_dir}/events.out.tfevents.*"))
        entry = {"env": env, "robot": robot, "mode": mode}
        if not tb_files:
            entry["status"] = "NO_TB"
            results.append(entry)
            continue
        ea = EventAccumulator(tb_files[-1])
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        success_tag = None
        for cand in ["eval/success_once", "eval/success_once_mean", "charts/eval_success_once"]:
            if cand in tags:
                success_tag = cand
                break
        if success_tag is None:
            for t in tags:
                if "success_once" in t.lower() and "eval" in t.lower():
                    success_tag = t
                    break
        if success_tag is None:
            entry["status"] = "NO_TAG"
            results.append(entry)
            continue
        vals = ea.Scalars(success_tag)
        if not vals:
            entry["status"] = "EMPTY"
            results.append(entry)
            continue
        entry["status"] = "OK"
        entry["final"] = float(vals[-1].value)
        entry["max"] = float(max(v.value for v in vals))
        entry["final_step"] = int(vals[-1].step)
        for thresh in THRESHOLDS:
            entry[f"first_step_to_{thresh}"] = first_step_above(vals, thresh)
        # Also store the area under the curve (proxy for cumulative sample efficiency)
        # Simple trapezoidal rule across eval points.
        steps = [v.step for v in vals]
        vs = [v.value for v in vals]
        auc = 0.0
        for i in range(1, len(steps)):
            auc += (vs[i] + vs[i - 1]) / 2.0 * (steps[i] - steps[i - 1])
        entry["auc"] = auc
        entry["max_step"] = max(steps) if steps else None
        # Normalize AUC by total steps so it's a "mean success_once over training"
        if entry["max_step"]:
            entry["auc_normalized"] = auc / entry["max_step"]
        results.append(entry)
    return results


@app.local_entrypoint()
def main():
    results = summarize.remote()
    by_key = {(r["env"], r["robot"], r["mode"]): r for r in results}

    def fmt_step(s):
        if s is None:
            return "never"
        if s >= 1_000_000:
            return f"{s/1_000_000:.2f}M"
        if s >= 1_000:
            return f"{s/1_000:.0f}k"
        return str(s)

    def fmt_val(v):
        return f"{v:.4f}" if v is not None else "n/a"

    def fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "n/a"

    print("\n" + "=" * 130)
    print("SAMPLE EFFICIENCY: FIRST STEP TO REACH SUCCESS_ONCE THRESHOLD")
    print("=" * 130)
    print(f"{'env':<10}  {'robot':<18}  {'mode':<12}  "
          f"{'→0.5':>10}  {'→0.75':>10}  {'→0.9':>10}  "
          f"{'final':>8}  {'AUC_norm':>10}")
    print("-" * 130)
    for env in ["PickCube", "StackCube"]:
        for robot in ["weakpanda", "dampedpanda", "weakdampedpanda"]:
            for mode in ["fromscratch", "finetune"]:
                r = by_key.get((env, robot, mode))
                if not r or r.get("status") != "OK":
                    continue
                print(
                    f"{env:<10}  {robot:<18}  {mode:<12}  "
                    f"{fmt_step(r.get('first_step_to_0.5')):>10}  "
                    f"{fmt_step(r.get('first_step_to_0.75')):>10}  "
                    f"{fmt_step(r.get('first_step_to_0.9')):>10}  "
                    f"{fmt_val(r.get('final')):>8}  "
                    f"{fmt_pct(r.get('auc_normalized')):>10}"
                )
        print("-" * 130)

    print("\n\nSAMPLE-EFFICIENCY DELTAS (fine-tune - scratch; NEGATIVE = fine-tune is FASTER)")
    print("=" * 130)
    print(f"{'env':<10}  {'robot':<18}  "
          f"{'Δsteps→0.5':>14}  {'Δsteps→0.75':>14}  {'Δsteps→0.9':>14}  "
          f"{'ΔAUC%':>10}")
    print("-" * 130)
    for env in ["PickCube", "StackCube"]:
        for robot in ["weakpanda", "dampedpanda", "weakdampedpanda"]:
            sc = by_key.get((env, robot, "fromscratch"))
            ft = by_key.get((env, robot, "finetune"))
            if not sc or not ft or sc.get("status") != "OK" or ft.get("status") != "OK":
                continue
            def delta(s_key):
                sv = sc.get(s_key)
                fv = ft.get(s_key)
                if sv is None and fv is None:
                    return "both never"
                if sv is None:
                    return f"FT only ({fmt_step(fv)})"
                if fv is None:
                    return f"SC only ({fmt_step(sv)})"
                d = fv - sv
                sign = "+" if d >= 0 else ""
                return f"{sign}{fmt_step(abs(d)) if d >= 0 else '-' + fmt_step(abs(d))}"
            auc_sc = sc.get("auc_normalized") or 0
            auc_ft = ft.get("auc_normalized") or 0
            print(f"{env:<10}  {robot:<18}  "
                  f"{delta('first_step_to_0.5'):>14}  "
                  f"{delta('first_step_to_0.75'):>14}  "
                  f"{delta('first_step_to_0.9'):>14}  "
                  f"{(auc_ft-auc_sc)*100:+.1f}pp"[:10].rjust(10))
    print("=" * 130)
