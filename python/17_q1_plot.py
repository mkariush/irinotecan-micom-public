"""Q1 figure: potential -> realized collapse as SN-38G supply drops.
Left: per-sample realized capacity vs supply (each plateaus at its capacity, then tracks supply).
Right: between-sample CV and corr(realized, stage1 capacity) vs supply -> variation collapses and
capacity decouples from carrier abundance under limiting substrate.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

df = pd.read_parquet("data/processed/flux/substrate_sweep.parquet")
supplies = sorted(df.supply.unique())

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: per-sample curves, coloured by stage-1 capacity
norm = Normalize(df.stage1_cap.min(), df.stage1_cap.max())
cmap = plt.cm.viridis
for sid, g in df.groupby("sample_id"):
    g = g.sort_values("supply")
    axA.plot(g.supply, g.realized, "-", lw=0.8, alpha=0.7, color=cmap(norm(g.stage1_cap.iloc[0])))
axA.plot(supplies, supplies, "k--", lw=1.2, label="realized = supply (fully delivery-limited)")
axA.set_xscale("log"); axA.set_xlabel("SN-38G supply (relative)")
axA.set_ylim(0, 95)   # cap to the capacity range so per-sample plateaus are visible
axA.set_ylabel("realized reactivation capacity")
axA.set_title("Each community plateaus at its potential,\nthen tracks the delivered amount")
axA.legend(loc="upper left", fontsize=9)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=axA, label="reactivation potential (saturating)")

# Panel B: CV and corr vs supply
rows = []
for S, g in df.groupby("supply"):
    cv = 100*g.realized.std()/g.realized.mean() if g.realized.mean() else 0
    corr = np.corrcoef(g.realized, g.stage1_cap)[0,1] if g.realized.std()>0 else np.nan
    rows.append((S, cv, corr))
summ = pd.DataFrame(rows, columns=["supply","cv","corr"]).sort_values("supply")
axB.plot(summ.supply, summ.cv, "o-", color="tab:red", label="CV% across individuals")
axB.set_xscale("log"); axB.set_xlabel("SN-38G supply (relative)")
axB.set_ylabel("CV% of realized capacity", color="tab:red")
axB.tick_params(axis="y", labelcolor="tab:red")
ax2 = axB.twinx()
ax2.plot(summ.supply, summ["corr"], "s-", color="tab:blue", label="corr(realized, potential)")
ax2.set_ylabel("corr(realized, carrier-abundance potential)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue"); ax2.set_ylim(0, 1.05)
axB.set_title("As substrate drops: variation collapses,\ncapacity decouples from composition")
# shade the physiological (limiting) end
axB.axvspan(supplies[0], 8, color="0.9", zorder=0)
axB.text(2.2, axB.get_ylim()[1]*0.93, "physiological\n(delivery-limited)", fontsize=9, ha="center")

plt.tight_layout()
plt.savefig("data/processed/figures/results_Q1_collapse.png", dpi=200)
print(summ.to_string(index=False))
print("saved -> data/processed/figures/results_Q1_collapse.png")
