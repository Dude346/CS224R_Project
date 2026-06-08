"""
Generate the paper's 8 core figures.
  - Figures 1-5: pulled from logged TB metrics (experiments/figures/figdata.json,
    produced by modal_pull_figdata.py — reads the Modal volume, no retraining).
  - Figures 6-8: from the transfer/Fetch runs (series captured in run logs).
Run:  .venv/bin/python experiments/figures/make_paper_figs.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figdata.json")))
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 11, "figure.dpi": 170})

def ser(run, tag):
    s = D.get(run, {}).get(tag)
    if not s:
        return None, None
    return [k/1000 for k, _ in s], [v for _, v in s]   # x in thousands of steps

V3   = "hiro_pickcube_panda_joint_v3_tefix_250k"
V6   = "hiro_pickcube_panda_v6_hybrid_w50_250k"
P5   = "hiro_p5_cubeCentric_b00_w50_g08_seed1_200k"
M2   = "hiro_M2_noOPC_cubeCentric_w50_seed1_300k"
ORA  = "hiro_e3a_oracle_cubeCentric_w50_seed1_300k"
E2   = "hiro_E2_annealW_noOPC_cubeCentric_seed1_300k"

# ---------- FIGURE 1: V1 residual reward poisoning ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x, y = ser(V3, "train/reward"); ax[0].plot(x, y, color="#d62728", lw=2)
ax[0].set_title("Worker task reward (V1 residual)"); ax[0].set_xlabel("steps (k)"); ax[0].set_ylabel("train reward")
ax[0].annotate("peaks ~126k\nthen degrades", xy=(126, max(y)), xytext=(150, max(y)*0.6),
               arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=11)
ax[0].grid(alpha=0.3)
x, y = ser(V3, "low/actor_loss"); ax[1].plot(x, y, color="#9467bd", lw=2)
ax[1].set_title("Worker actor loss (climbs = destabilizing)"); ax[1].set_xlabel("steps (k)"); ax[1].set_ylabel("actor loss"); ax[1].grid(alpha=0.3)
fig.suptitle("FIG 1 — V1 true-HIRO residual reward poisons the worker\n"
             "(cube term active pre-grasp → uncontrollable penalty; eval success stays 0, intrinsic stuck at ~-0.18 floor)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure1_reward_poisoning.png")); plt.close(fig)

# ---------- FIGURE 2: hover-hack vs cube-centric fix ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x, y = ser(V6, "data/grasp_rate"); ax[0].plot(x, y, "-o", color="#d62728", ms=3, label="tcp+cube subgoal")
x, y = ser(P5, "data/grasp_rate"); ax[0].plot(x, y, "-s", color="#2ca02c", ms=3, label="cube-centric subgoal")
ax[0].set_title("Grasp rate"); ax[0].set_xlabel("steps (k)"); ax[0].set_ylabel("grasp rate"); ax[0].set_ylim(-0.03, 0.7); ax[0].grid(alpha=0.3); ax[0].legend()
x, y = ser(V6, "data/mean_intrinsic_reward"); ax[1].plot(x, y, "-o", color="#d62728", ms=3, label="tcp+cube (near-max, no grasp = hover-hack)")
x, y = ser(P5, "data/mean_intrinsic_reward"); ax[1].plot(x, y, "-s", color="#2ca02c", ms=3, label="cube-centric (rises WITH grasp)")
ax[1].set_title("Mean intrinsic reward  (note: reward defs differ)"); ax[1].set_xlabel("steps (k)"); ax[1].set_ylabel("intrinsic reward"); ax[1].grid(alpha=0.3); ax[1].legend()
fig.suptitle("FIG 2 — The hover-hack and its fix: object-centric subgoal forces real grasping (~0.4% → ~57%)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure2_hover_hack.png")); plt.close(fig)

# ---------- FIGURE 3: off-policy correction ON vs OFF ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x, y = ser(P5, "eval/success_once"); ax[0].plot(x, y, "-o", color="#d62728", ms=3, label="OPC ON (stuck)")
x, y = ser(M2, "eval/success_once"); ax[0].plot(x, y, "-s", color="#2ca02c", ms=3, label="OPC OFF (solves)")
ax[0].set_title("Success rate"); ax[0].set_xlabel("steps (k)"); ax[0].set_ylabel("success"); ax[0].set_ylim(-0.05, 1.05); ax[0].grid(alpha=0.3); ax[0].legend()
x, y = ser(P5, "eval/subgoal_goal_align"); ax[1].plot(x, y, "-o", color="#d62728", ms=3, label="OPC ON")
x, y = ser(M2, "eval/subgoal_goal_align"); ax[1].plot(x, y, "-s", color="#2ca02c", ms=3, label="OPC OFF")
ax[1].axhline(0, color="gray", lw=0.8, ls=":")
ax[1].set_title("sg_align (manager → cube→goal)"); ax[1].set_xlabel("steps (k)"); ax[1].set_ylabel("sg_align"); ax[1].grid(alpha=0.3); ax[1].legend()
fig.suptitle("FIG 3 — Off-policy subgoal correction is a silent killer (cube-centric, w=0.5): ON never solves, OFF reaches 1.0", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure3_opc_on_off.png")); plt.close(fig)

# ---------- FIGURE 4: sg_align — manager not load-bearing ----------
fig, ax = plt.subplots(figsize=(8, 4.6))
x, y = ser(M2, "eval/subgoal_goal_align"); ax.plot(x, y, "-s", color="#1f77b4", lw=2, label="learned manager (M2)")
x, y = ser(ORA, "eval/subgoal_goal_align"); ax.plot(x, y, "-^", color="#2ca02c", lw=2, label="oracle manager (~0.84)")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_title("FIG 4 — sg_align: the learned manager is NOT load-bearing\n(its subgoals barely point at the goal vs the oracle's ~0.84)")
ax.set_xlabel("steps (k)"); ax.set_ylabel("sg_align = cos(subgoal, cube→goal)"); ax.set_ylim(-0.6, 1.0); ax.grid(alpha=0.3); ax.legend(loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure4_sg_align_loadbearing.png")); plt.close(fig)

# ---------- FIGURE 5: anneal w->0 collapse ----------
fig, ax = plt.subplots(figsize=(8.4, 4.6))
x, y = ser(M2, "eval/success_once"); ax.plot(x, y, "-s", color="#2ca02c", lw=2, label="fixed w=0.5 (M2) → solves")
x, y = ser(E2, "eval/success_once"); ax.plot(x, y, "-o", color="#d62728", lw=2, label="anneal w→0 (E2) → never solves")
ax.set_xlabel("steps (k)"); ax.set_ylabel("success"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
ax2 = ax.twinx()
x, y = ser(E2, "data/worker_w"); ax2.plot(x, y, "--", color="#7f7f7f", lw=1.6, label="w schedule (E2)")
ax2.set_ylabel("worker env-reward weight w", color="#7f7f7f"); ax2.set_ylim(-0.02, 0.55)
ax.set_title("FIG 5 — Annealing the env reward to 0 collapses performance\n(forcing reliance on the manager → fails; same recipe with w=0.5 solves)")
l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, loc="center left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure5_anneal_collapse.png")); plt.close(fig)

# ---------- FIGURE 6: catastrophic forgetting vs graceful adaptation ----------
sac = ([0,50,100,150,200,250,300,350,400,450],[1.00,0.00,0.812,0.938,1.00,1.00,0.938,0.938,1.00,1.00])
hu  = ([0,25,50,75,100,125,150,175,200],[0.980,0.730,0.777,0.859,0.875,0.926,0.934,0.945,0.957])
hf  = ([0,25,50,75,100,125,150,175,200],[0.980,0.695,0.832,0.828,0.848,0.906,0.949,0.922,0.961])
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.plot(*sac, "-o", color="#d62728", lw=2.2, label="Vanilla SAC")
ax.plot(*hu, "-s", color="#1f77b4", lw=2.2, label="HRL, manager unfrozen")
ax.plot(*hf, "-^", color="#2ca02c", lw=2.2, label="HRL, manager frozen")
ax.annotate("SAC: catastrophic\nforgetting (→0%)", xy=(50,0), xytext=(75,0.22), arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=11)
ax.set_title("FIG 6 — Fine-tuning onto a new embodiment (weak+damped)\nSAC collapses to 0%; HRL holds ~0.70 floor (frozen ≈ unfrozen)")
ax.set_xlabel("fine-tune steps (k)"); ax.set_ylabel("success"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure6_catastrophic_forgetting.png")); plt.close(fig)

# ---------- FIGURE 7: zero-shot transfer ----------
fig, ax = plt.subplots(figsize=(7, 4.6))
robots = ["panda\n(native)", "weak+damped", "scaled action"]; succ = [0.988, 0.980, 0.977]
b = ax.bar(robots, succ, color=["#7f7f7f","#1f77b4","#2ca02c"], width=0.6)
ax.axhline(0.988, color="#7f7f7f", lw=1, ls="--", label="native ceiling")
for bb, s in zip(b, succ): ax.text(bb.get_x()+bb.get_width()/2, s+0.005, f"{s:.3f}", ha="center", fontsize=12)
ax.set_title("FIG 7 — Whole-policy zero-shot transfer (256 episodes)"); ax.set_ylabel("success"); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=0.3); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure7_zeroshot_transfer.png")); plt.close(fig)

# ---------- FIGURE 8: Fetch feasibility ----------
xfer = ([0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
        [0,0,0,0,0,0,0.156,0.969,1,1,1,0.969,0.969,1,0.969,1,1,1,1,1,1])
scr  = ([0,25,50,75,100,125,150,175,200,225,250,275],[0,0,0,0,0,0,0,0,0,0,0,0])
sacf = ([50,100,150,200,250,300,350,400,450],[0.0,0.06,0.19,0.88,0.93,0.94,1.0,1.0,1.0])  # digitized from user's W&B screenshot
fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.plot(*xfer, "-^", color="#2ca02c", lw=2.2, label="HRL oracle object-centric mgr + fresh worker")
ax.plot(*sacf, "-d", color="#1f77b4", lw=2.0, label="Vanilla SAC from scratch (W&B, approx)")
ax.plot(*scr, "-o", color="#d62728", lw=2.2, label="HRL from scratch (learned mgr) — stuck")
ax.set_title("FIG 8 — Fetch (13-DOF mobile manipulator): FEASIBILITY, not efficiency\nobject-centric high level + fresh worker solves; SAC also solves; learned-mgr-from-scratch stuck")
ax.set_xlabel("training steps (k)"); ax.set_ylabel("success"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure8_fetch_feasibility.png")); plt.close(fig)

print("wrote:", sorted(f for f in os.listdir(HERE) if f.startswith("figure") and f.endswith(".png")))
