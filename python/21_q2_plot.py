"""Q2 figure: is growth-coupled, carbon-limited reactivation additive or non-additive?
Left : incidental SN-38 vs stage-1 capacity (carrier abundance)  -> tight line = ADDITIVE (Q2 null)
Right: incidental SN-38 vs community growth                      -> tight line = GROWTH-COUPLED (Q2 go)
Annotates Spearman rho on each panel and the frac-of-ceiling distribution.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

d = pd.read_csv("data/processed/flux/q2_full.csv")
ok = d[d.incidental_sn38.notna() & d.status.astype(str).str.startswith("optimal")].copy()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

rs_cap, p_cap = spearmanr(ok.incidental_sn38, ok.capacity)
axA.scatter(ok.capacity, ok.incidental_sn38, c="tab:purple", s=40, alpha=0.8, edgecolor="k", lw=0.4)
axA.set_xlabel("reactivation potential / carrier abundance (stage-1 capacity)")
axA.set_ylabel("incidental SN-38 in growth solution")
axA.set_title(f"vs capacity:  Spearman rho = {rs_cap:.2f}  (p={p_cap:.1g})\n"
              f"tight line => ADDITIVE (Q2 null)")

rs_gr, p_gr = spearmanr(ok.incidental_sn38, ok.growth)
axB.scatter(ok.growth, ok.incidental_sn38, c="tab:green", s=40, alpha=0.8, edgecolor="k", lw=0.4)
axB.set_xlabel("community growth rate (carbon-limited)")
axB.set_ylabel("incidental SN-38 in growth solution")
axB.set_title(f"vs growth:  Spearman rho = {rs_gr:.2f}  (p={p_gr:.1g})\n"
              f"tight line => GROWTH-COUPLED / NON-ADDITIVE (Q2 go)")

fig.suptitle(f"Q2: realized reactivation in the growing community (n={len(ok)}, "
             f"median frac of ceiling = {ok.frac_of_ceiling.median():.3f})", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/figures/results_Q2_nonadditive.png", dpi=200, bbox_inches="tight")
print(f"n={len(ok)}  rho(capacity)={rs_cap:.3f}  rho(growth)={rs_gr:.3f}  "
      f"frac median={ok.frac_of_ceiling.median():.4f}")
print("saved -> data/processed/figures/results_Q2_nonadditive.png")
