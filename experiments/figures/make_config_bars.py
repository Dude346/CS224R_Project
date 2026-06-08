"""Bar chart: only adding the env reward solves the task. Final eval success per config.
Volume runs from figdata.json; W&B runs (ariannac/anjalisr) pulled live."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wandb
HERE=os.path.dirname(os.path.abspath(__file__)); D=json.load(open(os.path.join(HERE,"figdata.json")))
plt.rcParams.update({"font.size":13,"axes.titlesize":16,"figure.dpi":220,
                     "figure.facecolor":"white","axes.facecolor":"white"})
api=wandb.Api(timeout=60)
def vol_final(run):
    s=D.get(run,{}).get("eval/success_once"); return s[-1][1] if s else None
def wb_final(proj,rid):
    return api.run(f"mashwin-stanford-university/{proj}/{rid}").summary.get("eval/success_once")
# (label, value, source-run, w-source)
BARS=[
 ("Pure intrinsic\n(HIRO residual)", wb_final("cs224r-hiro","iegstnds"),               "iegstnds (anjalisr)","w=0*"),
 ("Reward shaping\n(w=0)",           vol_final("hiro_pD_w0_reach10_cubeCentric_pPBRS_b05_seed1_200k"),"pD_w0_reach10_pPBRS","w=0"),
 ("Cube-centric\n(w=0)",             vol_final("hiro_pA_w0_cubeCentric_b00_g08_seed1_200k"),"pA_w0_cubeCentric_b00","w=0"),
 ("Object-centric\n(w=0)",           wb_final("cs224r-hiro","ahfz6s07"),                "ahfz6s07 (ariannac)","w=0*"),
 ("Latent object-centric\n(w=0)",    wb_final("cs224r-hiro","vmny4t9a"),                "vmny4t9a (ariannac)","w=0*"),
 ("+ Env reward\n(M2)",              vol_final("hiro_M2_noOPC_cubeCentric_w50_seed1_300k"),"M2_noOPC_cubeCentric_w50","w=0.5"),
]
print(f"{'config':26} {'value':6} {'run':28} {'w':6}")
for lab,v,run,w in BARS:
    print(f"{lab.replace(chr(10),' '):26} {('%.3f'%v) if v is not None else 'MISSING':6} {run:28} {w}")

labels=[b[0] for b in BARS]; vals=[b[1] for b in BARS]
colors=["#AAAAAA"]*5+["#2C7FB8"]
fig,ax=plt.subplots(figsize=(10,6))
bars=ax.bar(range(len(vals)),vals,color=colors,width=0.66,edgecolor="white",zorder=3)
ax.axhline(1.0,color="#2ca02c",ls="--",lw=1.8,zorder=2,label="Oracle (perfect subgoals)")
for i,(b,v) in enumerate(zip(bars,vals)):
    ax.text(b.get_x()+b.get_width()/2, v+0.025, f"{v:.2f}", ha="center", fontsize=13,
            fontweight=("bold" if i==len(vals)-1 else "normal"),
            color=("#2C7FB8" if i==len(vals)-1 else "#555555"))
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
ax.set_ylabel("Final Eval Success Rate"); ax.set_ylim(0,1.10)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.grid(axis="y",alpha=0.35,color="#cccccc",zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper left",fontsize=11,frameon=False)
fig.suptitle("Only the environment reward solves the task", fontsize=17, fontweight="bold", y=0.985)
fig.text(0.5,0.918,"Every hierarchy-side attempt stays near 0%; adding the env reward — which lets the\nworker bypass the manager — is the only change that works.",
         ha="center",fontsize=10.5,color="#666666")
fig.tight_layout(rect=[0,0,1,0.87])
out=os.path.join(HERE,"config_comparison_bars.png"); fig.savefig(out,dpi=220,facecolor="white"); plt.close(fig)
print("wrote",out)
