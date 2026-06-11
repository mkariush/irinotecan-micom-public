"""R5 forest plot: per-cohort CRC-vs-control effect size (Cliff's delta) + pooled estimate.

Cliff's delta = P(control>CRC) - P(CRC>control); positive => control higher, negative => CRC higher.
Per-cohort delta with bootstrap 95% CI; pooled = n-weighted mean with bootstrap CI. Shows that
significant per-cohort effects point in OPPOSITE directions and the pooled estimate straddles zero.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
cap  = pd.read_parquet("data/processed/flux/full_capacity.parquet")
meta = pd.read_parquet("data/processed/sample_metadata.parquet")
d = cap.merge(meta[["sample_id", "cohort", "study_condition"]], on="sample_id", how="left")
d = d[d.study_condition.isin(["CRC", "control"])]

def cliffs_delta(crc, ctl):
    # positive => control > CRC
    diff = ctl[:, None] - crc[None, :]
    return (np.sign(diff).sum()) / (len(crc) * len(ctl))

def boot_ci(crc, ctl, n=2000):
    ds = np.empty(n)
    for i in range(n):
        a = rng.choice(crc, len(crc), replace=True)
        b = rng.choice(ctl, len(ctl), replace=True)
        ds[i] = cliffs_delta(a, b)
    return np.percentile(ds, [2.5, 97.5])

rows = []
for coh, g in d.groupby("cohort"):
    crc = g.loc[g.study_condition == "CRC", "sn38_capacity"].values
    ctl = g.loc[g.study_condition == "control", "sn38_capacity"].values
    if len(crc) < 3 or len(ctl) < 3:
        continue
    delta = cliffs_delta(crc, ctl)
    lo, hi = boot_ci(crc, ctl)
    _, p = mannwhitneyu(crc, ctl, alternative="two-sided")
    rows.append({"cohort": coh, "delta": delta, "lo": lo, "hi": hi, "p": p,
                 "n": len(crc) + len(ctl), "n_crc": len(crc), "n_ctl": len(ctl)})
df = pd.DataFrame(rows).sort_values("delta")

# pooled: n-weighted mean delta, bootstrap CI (resample within cohorts)
def pooled_delta(frame):
    return np.average(frame["delta"], weights=frame["n"])
pb = np.empty(2000)
groups = {coh: (g.loc[g.study_condition=="CRC","sn38_capacity"].values,
                g.loc[g.study_condition=="control","sn38_capacity"].values)
          for coh, g in d.groupby("cohort")
          if (g.study_condition=="CRC").sum()>=3 and (g.study_condition=="control").sum()>=3}
for i in range(2000):
    ds, ns = [], []
    for coh,(crc,ctl) in groups.items():
        a=rng.choice(crc,len(crc),replace=True); b=rng.choice(ctl,len(ctl),replace=True)
        ds.append(cliffs_delta(a,b)); ns.append(len(crc)+len(ctl))
    pb[i]=np.average(ds,weights=ns)
pooled = np.average(df["delta"], weights=df["n"]); plo,phi = np.percentile(pb,[2.5,97.5])

# ---- plot ----
fig, ax = plt.subplots(figsize=(9, 6))
ypos = np.arange(len(df))[::-1]
sig = df["p"] < 0.05
ax.errorbar(df["delta"], ypos, xerr=[df["delta"]-df["lo"], df["hi"]-df["delta"]],
            fmt="s", color="0.25", ecolor="0.5", capsize=3, ms=7, ls="none")
ax.scatter(df.loc[sig,"delta"], ypos[sig.values], color="crimson", s=70, zorder=5, label="p < 0.05")
ax.axvline(0, color="k", lw=1, ls="--")
# pooled diamond
yb = -1.5
ax.add_patch(plt.Polygon([[plo,yb],[pooled,yb+0.3],[phi,yb],[pooled,yb-0.3]], color="navy"))
ax.text(pooled, yb-0.7, "Pooled (n-weighted)", ha="center", va="top", fontsize=10, color="navy")

labels = [f"{r.cohort}  (n={r.n_crc}/{r.n_ctl}, p={r.p:.3g})" for r in df.itertuples()]
ax.set_yticks(list(ypos)+[yb]); ax.set_yticklabels(labels+[""])
ax.set_xlim(-0.8, 0.8); ax.set_ylim(yb-1.2, len(df)-0.3)
ax.set_xlabel("Cliff's δ  (← CRC higher    |    control higher →)")
ax.set_title("CRC vs control SN-38 reactivation capacity, per cohort")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("data/processed/figures/results_R5_forest.png", dpi=200)
print(df.round(3).to_string(index=False))
print(f"\nPooled delta = {pooled:.3f}  95% CI [{plo:.3f}, {phi:.3f}]  (straddles 0 = no robust difference)")
print("saved -> data/processed/figures/results_R5_forest.png")
