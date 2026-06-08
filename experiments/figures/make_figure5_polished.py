"""FIG 5 (polished): annealing the env reward to 0 -> never solves. Styling + honesty pass.
Same data/structure (success left, w-schedule right). Data from figdata.json."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":13,"figure.dpi":220,"figure.facecolor":"white","axes.facecolor":"white"})
def ser(run,tag):
    s=D.get(run,{}).get(tag); return ([k/1000 for k,_ in s],[v for _,v in s]) if s else ([],[])
M2="hiro_M2_noOPC_cubeCentric_w50_seed1_300k"; E2="hiro_E2_annealW_noOPC_cubeCentric_seed1_300k"
GREEN="#2E8B57"; RED="#D64545"; GRAY="#999999"

fig,ax=plt.subplots(figsize=(9.6,6.1)); ax2=ax.twinx()
# faint band over the divergence region (where w is annealed away)
ax.axvspan(190,300,color=GRAY,alpha=0.07,zorder=0)
# left axis: success
x,y=ser(M2,"eval/success_once"); ax.plot(x,y,"-s",color=GREEN,lw=3,ms=7,solid_capstyle="round",
        zorder=4,label="fixed w=0.5 → solves")
x,y=ser(E2,"eval/success_once"); ax.plot(x,y,"-o",color=RED,lw=3,ms=7,solid_capstyle="round",
        zorder=4,label="anneal w→0 → never solves")
# right axis: w schedule (context, gray)
xw,yw=ser(E2,"data/worker_w"); ax2.plot(xw,yw,"--",color=GRAY,lw=2.4,solid_capstyle="round",
        zorder=2,label="w schedule")

# divergence annotation
ax.annotate("green keeps env reward → solves;\nred loses it → stays flat",
            xy=(216,0.44), xytext=(108,0.62),
            arrowprops=dict(arrowstyle="-|>",color="#444444",lw=1.8,connectionstyle="arc3,rad=-0.2"),
            fontsize=10.8, color="#333333")

ax.set_xlabel("training steps (k)",fontsize=14); ax.set_ylabel("success",fontsize=14)
ax.set_ylim(-0.03,1.05); ax.set_xlim(0,300); ax2.set_ylim(0,0.55)
ax2.set_ylabel("env-reward weight w (annealed)",fontsize=13,color=GRAY)
ax2.tick_params(axis="y",colors=GRAY); ax2.spines["right"].set_color(GRAY)
ax.grid(axis="y",alpha=0.4,color="#dddddd"); ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,loc="upper left",fontsize=11,edgecolor="#CCCCCC",framealpha=0.95)

fig.suptitle("Without the environment reward, the policy never solves",fontsize=15.5,fontweight="bold",y=0.985)
fig.text(0.5,0.905,"Two identical runs; the only difference is the env-reward weight w.\nKeep it (green) → solves.  Anneal it to 0 (red) → never solves.",
         ha="center",fontsize=10.8,color="#666666")
fig.tight_layout(rect=[0,0,1,0.85])
out=os.path.join(HERE,"figure5_anneal_polished.png"); fig.savefig(out,dpi=220,facecolor="white"); plt.close(fig)
print("wrote",out)
