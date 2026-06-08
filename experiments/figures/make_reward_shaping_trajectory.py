"""Subgoal State and Reward HRL Shaping (6 lines), pulled live from W&B."""
import os, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wandb
HERE=os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.labelsize":13,"legend.fontsize":10,"figure.dpi":170})
api=wandb.Api(timeout=90)
RUNS=[
 ("1. SAC Baseline",                            "cs224r-project","2xijocw3","#000000","-o"),
 ("2. HIRO Original",                           "cs224r-hiro","iegstnds","#d62728","-o"),
 ("3. HIRO + Reward Shaping",                   "cs224r-hiro","0mo7hp7g","#ff7f0e","-s"),
 ("4. Object-Centric",                          "cs224r-hiro","ahfz6s07","#9467bd","-s"),
 ("5. Latent Object-Centric (Baseline)",        "cs224r-hiro","vmny4t9a","#1f77b4","-^"),
 ("6. Latent Object-Centric + Reward Shaping",  "cs224r-hiro","i8ntgy5b","#2ca02c","-^"),
]
XMAX=500  # cut off at 500k timesteps
def series(h, col):
    pts=[]
    for row in h:
        v=row.get(col); s=row.get("_step")
        if v is None or s is None: continue
        if isinstance(v,float) and math.isnan(v): continue
        if s/1000.0 > XMAX: continue
        pts.append((s,v))
    pts.sort()
    return [p[0]/1000 for p in pts], [p[1] for p in pts]
fig,ax=plt.subplots(1,2,figsize=(14,5.2))
for lab,proj,rid,c,m in RUNS:
    r=api.run(f"mashwin-stanford-university/{proj}/{rid}")
    h=r.history(keys=["eval/success_once","eval/reward"], samples=10000)
    x,y=series(h,"eval/success_once");  ax[0].plot(x,y,m,color=c,lw=2,ms=3,label=lab)
    x,y=series(h,"eval/reward");        ax[1].plot(x,y,m,color=c,lw=2,ms=3,label=lab)
ax[0].set_title("Eval Success Once Rate"); ax[0].set_xlabel("Training Steps (K)"); ax[0].set_ylabel("Success Rate")
ax[0].set_ylim(-0.05,1.05); ax[0].set_xlim(0,XMAX); ax[0].grid(alpha=0.3); ax[0].legend(loc="center right",fontsize=9)
ax[1].set_title("Eval Reward"); ax[1].set_xlabel("Training Steps (K)"); ax[1].set_ylabel("Reward")
ax[1].set_ylim(0,1.0); ax[1].set_xlim(0,XMAX); ax[1].grid(alpha=0.3)
fig.suptitle("Subgoal State and Reward HRL Shaping  (PickCube, Panda, Seed 1)", fontsize=17)
fig.tight_layout(); out=os.path.join(HERE,"reward_shaping_trajectory.png"); fig.savefig(out); plt.close(fig)
print("wrote", out)
