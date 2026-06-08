"""Plot 1 (poster): HIRO HRL Reward Shaping Trajectory.
Loads experiments/figures/figdata.json (pulled via modal_pull_figdata.py).
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.labelsize":13,"legend.fontsize":11,"figure.dpi":170})
def ser(run,tag):
    s=D.get(run,{}).get(tag);
    return ([k/1000 for k,_ in s],[v for _,v in s]) if s else (None,None)
SAC="sac_PickCube_v1_panda_seed1_500000steps"
H1="hiro_PickCube_v1_panda_pd_joint_delta_pos_c10_seed1_500000steps"
HP="hiro_FULL_PRETRAINEDworker_w50_g08_300k"
M2="hiro_M2_noOPC_cubeCentric_w50_seed1_300k"
lines=[(SAC,"Vanilla SAC (baseline)","#000000","-o"),
       (H1,"First HIRO (naive) — what we improved upon","#d62728","-o"),
       (HP,"HIRO + pretrained worker (tcp+cube)","#ff7f0e","-s"),
       (M2,"Final HIRO = M2 (object-centric)","#2ca02c","-^")]
fig,ax=plt.subplots(1,2,figsize=(13,5))
for run,lab,c,m in lines:
    x,y=ser(run,"eval/success_once")
    if x: ax[0].plot(x,y,m,color=c,lw=2,ms=4,label=lab)
    x,y=ser(run,"eval/reward")
    if x: ax[1].plot(x,y,m,color=c,lw=2,ms=4,label=lab)
ax[0].set_title("Success rate"); ax[0].set_xlabel("training steps (k)"); ax[0].set_ylabel("success"); ax[0].set_ylim(-0.05,1.05); ax[0].grid(alpha=0.3); ax[0].legend(loc="center right",fontsize=10)
ax[1].axhline(0.4,color="gray",ls=":",lw=0.9); ax[1].text(5,0.41,"0.4 plateau",color="gray",fontsize=10)
ax[1].set_title("Normalized eval reward"); ax[1].set_xlabel("training steps (k)"); ax[1].set_ylabel("reward"); ax[1].set_ylim(0,1.0); ax[1].grid(alpha=0.3)
fig.suptitle("Plot 1 — HIRO HRL Reward Shaping Trajectory  (PickCube, Panda, seed 1)", fontsize=16)
fig.tight_layout(); fig.savefig(os.path.join(HERE,"plot1_reward_shaping_trajectory.png")); plt.close(fig)
print("wrote plot1_reward_shaping_trajectory.png")
