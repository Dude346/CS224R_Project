"""Panel 5 — Finding 2: The Manager Is Decorative.
Main: sg_align (learned manager ~0 vs oracle ~0.84). Inset: anneal w->0 -> reward collapses.
Data from experiments/figures/figdata.json (pulled via modal_pull_figdata.py)."""
import json, os, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,"legend.fontsize":12,"figure.dpi":180})
def ser(run,tag):
    s=D.get(run,{}).get(tag);
    return ([k/1000 for k,_ in s],[v for _,v in s]) if s else ([],[])
M2="hiro_M2_noOPC_cubeCentric_w50_seed1_300k"; ORA="hiro_e3a_oracle_cubeCentric_w50_seed1_300k"; E2="hiro_E2_annealW_noOPC_cubeCentric_seed1_300k"

fig,ax=plt.subplots(figsize=(10,6.8))
# --- MAIN: sg_align ---
x,y=ser(ORA,"eval/subgoal_goal_align"); ax.plot(x,y,"-^",color="#2ca02c",lw=2.6,ms=5,label="Oracle high level (reference)")
x,y=ser(M2,"eval/subgoal_goal_align"); ax.plot(x,y,"-o",color="#d62728",lw=2.6,ms=5,label="Learned manager (the solving run)")
ax.axhline(0,color="gray",lw=1,ls=":")
ax.axhspan(-0.4,0.2,color="#d62728",alpha=0.05)
ax.annotate("oracle ≈ 0.84\n(subgoals point at the goal)",xy=(150,0.84),xytext=(95,0.55),
            arrowprops=dict(arrowstyle="->",color="#2ca02c"),color="#2ca02c",fontsize=12)
ax.annotate("learned ≈ 0\n(not steering)",xy=(175,-0.10),xytext=(190,-0.30),
            arrowprops=dict(arrowstyle="->",color="#d62728"),color="#d62728",fontsize=12)
ax.set_xlabel("training steps (k)"); ax.set_ylabel("sg_align  =  cos(subgoal, cube→goal)")
ax.set_ylim(-0.42,1.0); ax.set_xlim(0,310); ax.grid(alpha=0.3); ax.legend(loc="upper center",fontsize=11)
ax.set_title("MEASURED: the manager's subgoals don't point where the cube must go", fontsize=13)

# --- INSET: anneal w->0 -> reward collapses (causal) ---
ins=ax.inset_axes([0.55,0.30,0.42,0.40])
xr,yr=ser(E2,"eval/reward"); ins.plot(xr,yr,"-",color="#d62728",lw=2.2,label="reward")
xw,yw=ser(E2,"data/worker_w"); ins.plot(xw,yw,"--",color="#7f7f7f",lw=1.8,label="env-reward weight w")
imax=max(range(len(yr)),key=lambda i:yr[i]) if yr else 0
ins.annotate("w→0 ⇒\nreward collapses",xy=(xr[-1],yr[-1]),xytext=(150,0.05),
             arrowprops=dict(arrowstyle="->",color="#d62728"),color="#d62728",fontsize=10)
ins.set_ylim(0,0.55); ins.set_xlim(0,310); ins.grid(alpha=0.3)
ins.set_title("CAUSAL: anneal env reward → 0",fontsize=11); ins.set_xlabel("steps (k)",fontsize=10); ins.set_ylabel("reward / w",fontsize=10)
ins.legend(fontsize=8,loc="upper right")

fig.suptitle("Finding 2 — The Manager Is Decorative", fontsize=20, fontweight="bold", y=0.985)
fig.text(0.5,0.935,"The hierarchy solves the task — but the worker does the work, not the manager.",
         ha="center",fontsize=12.5,style="italic")
fig.tight_layout(rect=[0,0,1,0.90]); out=os.path.join(HERE,"panel5_manager_decorative.png"); fig.savefig(out); plt.close(fig)
print("wrote",out)
