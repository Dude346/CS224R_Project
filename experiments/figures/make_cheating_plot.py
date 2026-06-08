"""The 'cheating plot': performance rises only with env reward; manager stays non-load-bearing.
Two stacked panels (shared x). Data from experiments/figures/figdata.json (pulled from the volume)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"legend.fontsize":10,"figure.dpi":180})
def ser(run,tag):
    s=D.get(run,{}).get(tag); return ([k/1000 for k,_ in s],[v for _,v in s]) if s else ([],[])
CLUSTER_A=[
 "hiro_pA_w0_cubeCentric_b00_g08_seed1_200k",
 "hiro_pA_w0_cubeCentric_pPBRS_b05_g08_seed1_200k",
 "hiro_pD_w0_reach10_cubeCentric_b00_seed1_200k",
 "hiro_pD_w0_reach10_cubeCentric_pPBRS_b05_seed1_200k",
 "hiro_pB_w0_ORACLE_cubeCentric_g08_seed1_200k",
 "hiro_pD_w0_reach10_ORACLE_cubeCentric_seed1_200k",
]
M2="hiro_M2_noOPC_cubeCentric_w50_seed1_300k"; ORA="hiro_e3a_oracle_cubeCentric_w50_seed1_300k"
XMAX=300
fig=plt.figure(figsize=(9.5,8.4))
gs=GridSpec(2,1,height_ratios=[3,2],hspace=0.10)
ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1],sharex=ax0)

# --- TOP: success ---
for i,run in enumerate(CLUSTER_A):
    x,y=ser(run,"eval/success_once")
    ax0.plot(x,y,"-",color="#8a8a8a",lw=1.8,alpha=0.65,
             label=("Hierarchy-side attempts (no env reward, w=0)" if i==0 else None))
x,y=ser(M2,"eval/success_once"); ax0.plot(x,y,"-o",color="#1f77b4",lw=3.2,ms=5,
             label="M2: object-centric + environment reward (w=0.5)")
ax0.axhline(1.0,color="#2ca02c",ls="--",lw=1.6,label="Oracle ceiling (perfect subgoals)")
ax0.annotate("environment reward\nadded (w=0.5) —\nthe only change that\nbreaks the wall",
             xy=(205,0.19),xytext=(30,0.50),
             arrowprops=dict(arrowstyle="->",color="#1f77b4",lw=2),color="#1f77b4",fontsize=12,fontweight="bold")
ax0.set_ylabel("Eval Success Rate"); ax0.set_ylim(-0.05,1.10); ax0.grid(alpha=0.3)
ax0.legend(loc="upper center",fontsize=10.5)
ax0.set_title("Success comes from the env reward, not the hierarchy", fontsize=13)
plt.setp(ax0.get_xticklabels(), visible=False)

# --- BOTTOM: sg_align ---
x,y=ser(M2,"eval/subgoal_goal_align"); ax1.plot(x,y,"-o",color="#1f77b4",lw=2.6,ms=5,
             label="M2 manager subgoal alignment")
ax1.axhline(0.84,color="#2ca02c",ls="--",lw=1.6,label="Oracle (manager steers, ~0.84)")
ax1.axhline(0,color="gray",ls=":",lw=0.9)
ax1.annotate("stays ≈ 0 while success climbs above", xy=(150,0.0), xytext=(70,-0.27),
             arrowprops=dict(arrowstyle="->",color="#1f77b4"),color="#1f77b4",fontsize=11)
ax1.set_ylabel("sg_align = cos(subgoal, cube→goal)"); ax1.set_xlabel("Training Steps (K)")
ax1.set_ylim(-0.42,1.0); ax1.set_xlim(0,XMAX); ax1.grid(alpha=0.3); ax1.legend(loc="upper right",fontsize=10.5)
ax1.set_title("...yet the manager never steers", fontsize=13)

fig.suptitle("Performance rises only with environment reward;\nthe manager stays non-load-bearing",
             fontsize=15, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0,0,1,0.92])
out=os.path.join(HERE,"cheating_plot_two_panel.png"); fig.savefig(out); plt.close(fig)
print("wrote",out)
