from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_plots"
REWARD_THRESHOLDS = (0.5, 0.8)

# Stanford-ish grouped palette:
# weak        -> reds
# damped      -> blues
# weakdamped  -> greens
# panda       -> neutral accent
ROBOT_STYLES = {
    "weak": {
        "scratch": {"color": "#820000", "linestyle": "-"},
        "finetune": {"color": "#E50808", "linestyle": "--"},
    },
    "damped": {
        "scratch": {"color": "#00548f", "linestyle": "-"},
        "finetune": {"color": "#6FC3FF", "linestyle": "--"},
    },
    "weakdamped": {
        "scratch": {"color": "#006F54", "linestyle": "-"},
        "finetune": {"color": "#1AECBA", "linestyle": "--"},
    },
    "panda": {
        "baseline": {"color": "#544948", "linestyle": "-"},
    },
}


@dataclass
class Series:
    label: str
    steps: list[int]
    values: list[float]
    robot: str
    variant: str


def iter_mean_columns(header: list[str]) -> Iterable[tuple[int, str]]:
    for idx, name in enumerate(header):
        if idx == 0:
            continue
        if name.endswith("__MIN") or name.endswith("__MAX"):
            continue
        yield idx, name


def shorten_label(raw: str) -> str:
    name = raw.replace(" - eval/reward", "")
    if name.startswith("ft_sac_"):
        name = name[len("ft_sac_") :]
    elif name.startswith("sac_"):
        name = name[len("sac_") :]

    robot = "panda"
    if "weakdampedpanda" in name:
        robot = "weakdamped"
    elif "dampedpanda" in name:
        robot = "damped"
    elif "weakpanda" in name:
        robot = "weak"

    if "finetune" in name:
        suffix = "finetune"
    elif "fromscratch" in name:
        suffix = "scratch"
    else:
        suffix = "baseline"

    return f"{robot} | {suffix}"


def classify_series(raw: str) -> tuple[str, str]:
    name = raw.replace(" - eval/reward", "")
    if "weakdampedpanda" in name:
        robot = "weakdamped"
    elif "dampedpanda" in name:
        robot = "damped"
    elif "weakpanda" in name:
        robot = "weak"
    else:
        robot = "panda"

    if "finetune" in name:
        variant = "finetune"
    elif "fromscratch" in name:
        variant = "scratch"
    else:
        variant = "baseline"
    return robot, variant


def load_series(path: Path) -> list[Series]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    header, data_rows = rows[0], rows[1:]

    series_list: list[Series] = []
    for idx, raw_name in iter_mean_columns(header):
        steps: list[int] = []
        values: list[float] = []
        for row in data_rows:
            if row[idx] == "":
                continue
            steps.append(int(float(row[0])))
            values.append(float(row[idx]))
        robot, variant = classify_series(raw_name)
        series_list.append(Series(shorten_label(raw_name), steps, values, robot, variant))
    return series_list


def first_step_at_or_above(steps: list[int], values: list[float], threshold: float) -> int | None:
    for step, value in zip(steps, values):
        if value >= threshold:
            return step
    return None


def auc_normalized(steps: list[int], values: list[float]) -> float | None:
    if len(steps) < 2:
        return None
    auc = 0.0
    for i in range(1, len(steps)):
        auc += (values[i] + values[i - 1]) / 2.0 * (steps[i] - steps[i - 1])
    total_span = steps[-1] - steps[0]
    if total_span <= 0:
        return None
    return auc / total_span


def write_summary(csv_path: Path, series_list: list[Series]) -> None:
    out_path = OUTPUT_DIR / f"{csv_path.stem}_reward_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "first_step_to_reward_0.5",
                "first_step_to_reward_0.8",
                "reward_auc_normalized",
                "final_reward",
                "final_step",
            ]
        )
        for series in series_list:
            writer.writerow(
                [
                    series.label,
                    first_step_at_or_above(series.steps, series.values, 0.5),
                    first_step_at_or_above(series.steps, series.values, 0.8),
                    auc_normalized(series.steps, series.values),
                    series.values[-1] if series.values else None,
                    series.steps[-1] if series.steps else None,
                ]
            )


def plot_csv(csv_path: Path) -> Path:
    series_list = load_series(csv_path)
    order = {"panda": 0, "weak": 1, "damped": 2, "weakdamped": 3}
    variant_order = {"baseline": 0, "scratch": 1, "finetune": 2}
    series_list.sort(key=lambda s: (order.get(s.robot, 99), variant_order.get(s.variant, 99), s.label))
    fig, ax = plt.subplots(figsize=(12, 7))
    for series in series_list:
        style = ROBOT_STYLES.get(series.robot, {}).get(series.variant, {})
        ax.plot(
            series.steps,
            series.values,
            linewidth=2.5,
            label=series.label,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
        )
    env_name = "PickCube" if "pickcube" in csv_path.stem.lower() else "StackCube"
    ax.set_title(f"{env_name} Baseline Reward vs Env Steps")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Eval reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{csv_path.stem}_reward_vs_steps.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    write_summary(csv_path, series_list)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name in ("pickcube_baseline.csv", "stackcube_baseline.csv"):
        out_path = plot_csv(ROOT / name)
        print(out_path)


if __name__ == "__main__":
    main()
