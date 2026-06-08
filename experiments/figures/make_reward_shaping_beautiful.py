"""Subgoal State and Reward HRL Shaping — poster-quality, bold, high-contrast red palette.
Dense raw eval points from experiments/figures/rsdata.json (pulled from W&B)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size":16,"font.weight":"bold","axes.titleweight":"bold",
                     "axes.labelweight":"bold","figure.dpi":220,
                     "figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#333333",
                     "axes.linewidth":1.4})
D=json.load(open(os.path.join(HERE,"rsdata.json"))); XMAX=500
def series(rows,idx):
    pts=[(r[0]/1000.0,r[idx]) for r in rows if r[0] is not None and r[idx] is not None and r[0]/1000.0<=XMAX]
    pts.sort(); return [p[0] for p in pts],[p[1] for p in pts]
# (label, color, linestyle, marker)  -- strong solid reds for the 3 dynamic lines, light dashed for the 3 failed
STYLE=[
 ("1. SAC Baseline",                            "#CB181D","-","o"),
 ("2. HIRO Original",                           "#FCBBA1","--","s"),
 ("3. HIRO + Reward Shaping",                   "#FC9272","--","^"),
 ("4. Object-Centric",                          "#FB6A4A","--","D"),
 ("5. Latent Object-Centric (Baseline)",        "#A50F15","-","v"),
 ("6. Latent Object-Centric + Reward Shaping",  "#67000D","-","*"),
]
DEEPRED="#8B0000"
fig,ax=plt.subplots(1,2,figsize=(13.5,6.2))
for lab,c,ls,mk in STYLE:
    rows=D[lab]; me=max(1,len([r for r in rows if r[0]/1000<=XMAX])//14)
    for col,a in [(1,ax[0]),(2,ax[1])]:
        x,y=series(rows,col)
        a.plot(x,y,color=c,ls=ls,marker=mk,lw=2.8,ms=8,markevery=me,
               markeredgecolor="white",markeredgewidth=0.7,label=lab,solid_capstyle="round")
for a,(ttl,ylab,ymax) in zip(ax,[("Eval Success Once Rate","Success Rate",1.05),("Eval Reward","Reward",1.0)]):
    a.set_title(ttl,fontsize=19,pad=10,color=DEEPRED); a.set_xlabel("Training Steps (K)",fontsize=17); a.set_ylabel(ylab,fontsize=17)
    a.set_xlim(0,XMAX); a.set_ylim(-0.03 if ymax>1 else 0,ymax)
    a.grid(axis="y",alpha=0.4,color="#dddddd"); a.set_axisbelow(True)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.tick_params(labelsize=14)
    for t in a.get_xticklabels()+a.get_yticklabels(): t.set_fontweight("bold")
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=3,fontsize=13,frameon=False,
           prop={"weight":"bold","size":13},bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Subgoal State and Reward HRL Shaping   (PickCube, Panda, Seed 1)",fontsize=21,fontweight="bold",y=0.99,color=DEEPRED)
fig.tight_layout(rect=[0,0.11,1,0.95])
out=os.path.join(HERE,"reward_shaping_trajectory.png"); fig.savefig(out,dpi=220,facecolor="white",bbox_inches="tight"); plt.close(fig)
print("wrote",out)
