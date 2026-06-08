"""FIG 5 (clean, 6x4): without the env reward, the policy never solves.
Single axis, two red lines, no secondary w-axis / dashed line. Data from figdata.json."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":11,"figure.dpi":220,"figure.facecolor":"white","axes.facecolor":"white",
                     "axes.edgecolor":"#333333","axes.linewidth":1.1})
def ser(run,tag):
    s=D.get(run,{}).get(tag); return ([k/1000 for k,_ in s],[v for _,v in s]) if s else ([],[])
M2="hiro_M2_noOPC_cubeCentric_w50_seed1_300k"; E2="hiro_E2_annealW_noOPC_cubeCentric_seed1_300k"
SOLVE="#B30000"; FAIL="#F0918B"

fig,ax=plt.subplots(figsize=(6,4))
x,y=ser(M2,"eval/success_once"); ax.plot(x,y,"-s",color=SOLVE,lw=2.6,ms=5,markeredgecolor="white",
        markeredgewidth=0.6,label="Fixed $w$ = 0.5  →  Solves",zorder=4)
x,y=ser(E2,"eval/success_once"); ax.plot(x,y,"-o",color=FAIL,lw=2.4,ms=5,markeredgecolor="white",
        markeredgewidth=0.6,label="Anneal $w$: 0.5 → 0  →  Never Solves",zorder=3)
# one short annotation at the divergence
ax.annotate("Only the run that keeps\nthe env reward solves", xy=(214,0.42), xytext=(40,0.46),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="-|>",color="#555555",lw=1.4,connectionstyle="arc3,rad=-0.25"))
ax.set_xlim(0,300); ax.set_ylim(-0.03,1.05)
ax.set_xlabel("Training Steps (K)",fontsize=11.5); ax.set_ylabel("Success",fontsize=11.5)
ax.grid(axis="y",alpha=0.35,color="#dddddd"); ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper left",fontsize=9.5,frameon=True,edgecolor="#CCCCCC",framealpha=0.95)
fig.suptitle("Without the environment reward, the policy never solves",fontsize=12.5,fontweight="bold",y=0.99,color=SOLVE)
fig.text(0.5,0.915,"Two identical runs; only the env-reward weight $w$ differs.",ha="center",fontsize=9,color="#666666")
fig.tight_layout(rect=[0,0,1,0.90])
out=os.path.join(HERE,"figure5_anneal_polished.png"); fig.savefig(out,dpi=220,facecolor="white"); plt.close(fig)
print("wrote",out)
