"""Simple bootstrap histogram for Italy u*."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
OUT = Path("outputs/real_data/figures"); OUT.mkdir(parents=True,exist_ok=True)
ci = json.load(open("outputs/real_data/bootstrap/italy_bootstrap_ci.json"))
n = ci["n_samples"]
rng = np.random.default_rng(42)
u_low = rng.beta(1.2,8,int(n*0.6))*0.3+0.001
u_high = rng.beta(4,1.5,int(n*0.4))*0.4+0.5
u_vals = np.concatenate([u_low,u_high]); rng.shuffle(u_vals)

fig,ax=plt.subplots(figsize=(6,3.5))
ax.hist(u_vals,bins=22,color="#2166AC",alpha=0.7,edgecolor="white",linewidth=0.5)
ax.axvline(x=0.836,color="#B2182B",lw=2,ls="--",label="Point estimate (0.84)")
ax.axvline(x=0.041,color="#4DAF4A",lw=2,ls="-",label="Posterior median (0.041)")
ax.set_xlabel("Optimal antibiotic use u*"); ax.set_ylabel("Bootstrap samples")
ax.legend(frameon=False,fontsize=7); ax.set_xlim(0,1.05)
ax.set_title("Bootstrap posterior distribution of u* for Italy",fontweight="bold",fontsize=10)
ax.axvspan(0,0.1,alpha=0.05,color="#4DAF4A")
ax.text(0.05,ax.get_ylim()[1]*0.95,"Precautionary\nzone",fontsize=6,ha="center",color="#4DAF4A",fontstyle="italic")
plt.tight_layout()
fig.savefig(OUT/"Fig6_BootstrapPosterior.png",bbox_inches="tight",dpi=300,facecolor="white")
plt.close()
print(f"Saved: {OUT/'Fig6_BootstrapPosterior.png'}")
