"""Cheating plot (FIXED): gray wall now visible (z-order), short local breakaway arrow,
softened bottom-panel claim. Data from experiments/figures/figdata.json."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"legend.fontsize":10,
                     "figure.dpi":220,"figure.facecolor":"white","axes.facecolor":"white"})
def ser(run,tag):
    s=D.get(run,{}).get(tag); return ([k/1000 for k,_ in s],[v for _,v in s]) if s else ([],[])
WALL=[
 "hiro_pA_w0_cubeCentric_b00_g08_seed1_200k","hiro_pA_w0_cubeCentric_pPBRS_b05_g08_seed1_200k",
 "hiro_pD_w0_reach10_cubeCentric_b00_seed1_200k","hiro_pD_w0_reach10_cubeCentric_pPBRS_b05_seed1_200k",
 "hiro_pB_w0_ORACLE_cubeCentric_g08_seed1_200k","hiro_pD_w0_reach10_ORACLE_cubeCentric_seed1_200k",
]
M2="hiro_M2_noOPC_cubeCentric_w50_seed1_300k"
XMAX=300
fig=plt.figure(figsize=(9.6,8.6))
gs=GridSpec(2,1,height_ratios=[3,2],hspace=0.10)
ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1],sharex=ax0)

# --- TOP: success ---
# M2 first (lower zorder) then the gray wall ON TOP so the wall is visible at y=0
x,y=ser(M2,"eval/success_once"); ax0.plot(x,y,"-o",color="#1f77b4",lw=3.4,ms=5,zorder=4,
            label="M2: object-centric + environment reward (w=0.5)")
for i,run in enumerate(WALL):
    x,y=ser(run,"eval/success_once")
    ax0.plot(x,y,"-",color="#AAAAAA",lw=1.5,alpha=0.5,zorder=6,
             label=("Hierarchy-side attempts (no env reward)" if i==0 else None))
ax0.axhline(1.0,color="#2ca02c",ls="--",lw=1.6,zorder=3,label="Oracle ceiling (perfect subgoals)")
ax0.annotate("env reward added (w=0.5) —\nthe only change that\nbreaks the wall",
             xy=(158,0.13), xytext=(70,0.52),
             bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#1f77b4",lw=1.5),
             arrowprops=dict(arrowstyle="-|>",color="#1f77b4",lw=2.2,connectionstyle="arc3,rad=-0.3"),
             color="#1f77b4",fontsize=11,fontweight="bold")
ax0.set_ylabel("Eval Success Rate"); ax0.set_ylim(-0.04,1.10); ax0.grid(alpha=0.3,color="#cccccc")
ax0.legend(loc="upper left",fontsize=10.5,bbox_to_anchor=(0.005,0.99))
ax0.set_title("Success comes from the env reward, not the hierarchy", fontsize=13)
plt.setp(ax0.get_xticklabels(), visible=False)

# --- BOTTOM: sg_align (softened claim) ---
x,y=ser(M2,"eval/subgoal_goal_align"); ax1.plot(x,y,"-o",color="#1f77b4",lw=2.6,ms=5,zorder=4,
            label="M2 manager subgoal alignment")
ax1.axhline(0.84,color="#2ca02c",ls="--",lw=1.6,label="Oracle (manager steers, ~0.84)")
ax1.axhline(0,color="gray",ls=":",lw=0.9)
ax1.annotate("stays far below the oracle\nwhile success climbs above", xy=(150,0.05), xytext=(165,-0.18),
             arrowprops=dict(arrowstyle="-|>",color="#1f77b4",lw=1.8,connectionstyle="arc3,rad=0.2"),
             color="#1f77b4",fontsize=10.5)
ax1.set_ylabel("sg_align = cos(subgoal, cube→goal)"); ax1.set_xlabel("Training Steps (K)")
ax1.set_ylim(-0.30,1.0); ax1.set_xlim(0,XMAX); ax1.grid(alpha=0.3,color="#cccccc")
ax1.legend(loc="upper right",fontsize=10.5)
ax1.set_title("...yet the manager stays far below the oracle", fontsize=13)

fig.suptitle("Performance rises only with the environment reward;\nthe manager stays non-load-bearing",
             fontsize=15, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0,0,1,0.92])
out=os.path.join(HERE,"cheating_plot_fixed.png"); fig.savefig(out,dpi=220,facecolor="white"); plt.close(fig)
print("wrote",out)
