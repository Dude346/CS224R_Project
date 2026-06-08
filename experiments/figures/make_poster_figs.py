"""
Poster figures from existing single-seed logs (no training).
    .venv/bin/python experiments/figures/make_poster_figs.py
Outputs PNGs (poster resolution) next to this script.

Data sources (tracked runs, seed 1, weakdampedpanda 0.75 force / 1.5 damping):
  - SAC : ft_sac_PickCube_v1_weakdampedpanda_pd_joint_delta_pos_seed1_500000steps_WEAKDAMPED_force075_damping15_finetune
  - HIRO unfrozen : hiro_xfer_WHOLEarm_ft_weakdamped_seed1_eval256_200k
  - HIRO frozen   : hiro_xfer_FROZENmgr_ft_weakdamped_seed1_eval256_200k  (P1; fill when done)
  - Zero-shot 256-ep: hiro_eval_CONTROL_panda / _scaledpanda / WHOLEarm step-0
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "legend.fontsize": 13, "figure.dpi": 200})

# --- Fig 1: fine-tune success vs steps (catastrophic forgetting) ---
sac   = ([0,50,100,150,200,250,300,350,400,450],
         [1.00,0.00,0.812,0.938,1.00,1.00,0.938,0.938,1.00,1.00])
hiro_u= ([0,25,50,75,100,125,150,175,200],
         [0.980,0.730,0.777,0.859,0.875,0.926,0.934,0.945,0.957])
# HIRO frozen object-centric manager (P1), 200k:
hiro_f= ([0,25,50,75,100,125,150,175,200],
         [0.980,0.695,0.832,0.828,0.848,0.906,0.949,0.922,0.961])

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(sac[0], sac[1], "-o", color="#d62728", lw=2.4, ms=6, label="Vanilla SAC (whole policy)")
ax.plot(hiro_u[0], hiro_u[1], "-s", color="#1f77b4", lw=2.4, ms=6, label="HIRO unfrozen (manager+worker)")
if hiro_f[0]:
    ax.plot(hiro_f[0], hiro_f[1], "-^", color="#2ca02c", lw=2.4, ms=6, label="HIRO frozen object-centric manager")
ax.axhline(0.0, color="gray", lw=0.8, ls=":")
ax.annotate("SAC: catastrophic\nforgetting (→0%)", xy=(50, 0.0), xytext=(70, 0.22),
            arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=12)
ax.annotate("HIRO: graceful\ndip (floor ~0.73)", xy=(25, 0.73), xytext=(120, 0.55),
            arrowprops=dict(arrowstyle="->", color="#1f77b4"), color="#1f77b4", fontsize=12)
ax.set_xlabel("fine-tuning steps on new embodiment (k)")
ax.set_ylabel("success rate")
ax.set_title("Fine-tuning a transferred policy onto a weak+damped arm\n(PickCube, seed 1)")
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_finetune_forgetting.png")); plt.close(fig)

# --- Fig 2: zero-shot transfer (256-episode eval) ---
robots = ["panda\n(native)", "weak+damped\n(0.75 / 1.5)", "scaled action\n(0.75)"]
succ   = [0.988, 0.980, 0.977]
fig, ax = plt.subplots(figsize=(7.0, 5.0))
bars = ax.bar(robots, succ, color=["#7f7f7f", "#1f77b4", "#2ca02c"], width=0.6)
ax.axhline(0.988, color="#7f7f7f", lw=1.0, ls="--", label="native ceiling (0.988)")
for b, s in zip(bars, succ):
    ax.text(b.get_x()+b.get_width()/2, s+0.005, f"{s:.3f}", ha="center", fontsize=13)
ax.set_ylabel("zero-shot success (256 episodes)")
ax.set_title("Whole-policy zero-shot embodiment transfer\n(no fine-tuning; degradation within noise)")
ax.set_ylim(0.0, 1.05); ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_zeroshot_transfer.png")); plt.close(fig)

# --- Fig 3: cross-embodiment transfer to a DIFFERENT robot (Fetch) ---
# Oracle (object-centric) manager + fresh worker SOLVES; learned-from-scratch
# manager leaves the worker stuck at 0% (it grasps but never transports).
xfer_s = ([0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
          [0,0,0,0,0,0,0.156,0.969,1.00,1.00,1.00,0.969,0.969,1.00,0.969,1.00,1.00,1.00,1.00,1.00,1.00])
scratch_s = ([0,25,50,75,100,125,150,175,200,225,250,275],
             [0,0,0,0,0,0,0,0,0,0,0,0])  # paused at 275k, still 0%
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(xfer_s[0], xfer_s[1], "-^", color="#2ca02c", lw=2.6, ms=6,
        label="object-centric manager + fresh worker  (sg_align≈0.78)")
ax.plot(scratch_s[0], scratch_s[1], "-o", color="#d62728", lw=2.6, ms=6,
        label="learned manager from scratch  (sg_align≈0)")
ax.annotate("transferred high level\n→ fresh worker SOLVES", xy=(200, 1.0), xytext=(210, 0.66),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c", fontsize=12)
ax.annotate("learn high level from scratch\n→ grasps but stuck at 0%", xy=(250, 0.0), xytext=(60, 0.18),
            arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=12)
ax.set_xlabel("training steps on Fetch (k)")
ax.set_ylabel("success rate")
ax.set_title("Cross-embodiment transfer to a DIFFERENT robot\n(Fetch: 13-DOF mobile manipulator, PickCube, seed 1)")
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_fetch_crossembodiment.png")); plt.close(fig)

# --- Fig 4: key-numbers table (poster panel) ---
rows = [
    ["PickCube M2 (panda, 256-ep)", "0.988"],
    ["Zero-shot → weak+damp / scaled (256-ep)", "0.980 / 0.977"],
    ["Grasp rate after object-centric subgoal fix", "~2% → ~53%"],
    ["M2 seeds solved @ w=0.5 / w=0.7", "1/3 → 3/3"],
    ["Fine-tune floor: SAC vs HIRO", "0.00 vs ~0.70"],
    ["Weakdamped: from-scratch / zero-shot", "0.688 / 0.98"],
    ["Fetch oracle-transfer (object-centric mgr)", "1.000 SOLVED"],
    ["Fetch from-scratch (learned mgr)", "0.000 (stuck)"],
    ["Fetch obs / action dims (vs 42/8)", "54 / 13"],
]
fig, ax = plt.subplots(figsize=(9.5, 4.2)); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=["Result", "Value"], loc="center",
               cellLoc="left", colLoc="left", colWidths=[0.70, 0.26])
tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 1.7)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1f77b4"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f0f0")
    cell.set_edgecolor("#cccccc")
ax.set_title("Key results (single seed; multi-seed in the report)", fontsize=16, pad=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_key_numbers.png")); plt.close(fig)

print("wrote:", sorted(f for f in os.listdir(HERE) if f.endswith(".png")))
