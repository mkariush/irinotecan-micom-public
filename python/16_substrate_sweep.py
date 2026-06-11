"""Stage-2 pilot, Q1: substrate sweep. Does abundance-driven capacity collapse to a delivery-
limited plateau as SN-38G supply drops from saturating toward physiological?

For ~40 samples (sequential, load once, sweep supply), recompute max EX_sn38_m at each supply S.
If reactivation is complete up to a community's capacity, realized = min(S, capacity_i), so as S
falls below the capacity range, between-sample variation should COLLAPSE (everyone converts the
delivered amount) and capacity should DECOUPLE from carrier abundance -> the "potential vs realized"
reframe (composition matters only below a low-carriage threshold).
"""

import glob, os, gc
import numpy as np
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR = "data/processed/models"
MEDIUM = "data/media/western_diet_gut_agora.qza"
SN38, SN38G = "EX_sn38_m", "EX_sn38g_m"
SUPPLY = [1000, 90, 60, 40, 25, 15, 8, 4, 2, 1]      # saturating -> limiting
N_PER_COHORT = 6

base_med = dict(zip(*[load_qiime_medium(MEDIUM)[c] for c in ("reaction", "flux")]))
meta = pd.read_parquet("data/processed/sample_metadata.parquet")
cap1 = pd.read_parquet("data/processed/flux/full_capacity.parquet").set_index("sample_id")["sn38_capacity"]
built = {os.path.basename(p).replace(".pickle","") for p in glob.glob(f"{MODELS_DIR}/*.pickle")}
samples = (meta[meta.sample_id.isin(built)].sort_values("sample_id")
           .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())
print(f"sweeping {len(samples)} samples x {len(SUPPLY)} supply levels", flush=True)

rows = []
for i, s in enumerate(samples, 1):
    com = load_pickle(f"{MODELS_DIR}/{s}.pickle")
    have = {r.id for r in com.exchanges}
    if SN38 not in com.reactions:
        del com; gc.collect(); continue
    com.objective = com.reactions.get_by_id(SN38)
    com.variables.community_objective.lb = 0.0
    for S in SUPPLY:
        med = dict(base_med); med[SN38G] = S
        com.medium = {r: f for r, f in med.items() if r in have}
        sol = com.optimize()
        rows.append({"sample_id": s, "supply": S,
                     "realized": float(sol.objective_value) if sol else 0.0})
    del com; gc.collect()
    if i % 6 == 0: print(f"  {i}/{len(samples)}", flush=True)

df = pd.DataFrame(rows)
df["stage1_cap"] = df["sample_id"].map(cap1)
df["expected_min"] = np.minimum(df["supply"], df["stage1_cap"])
df.to_parquet("data/processed/flux/substrate_sweep.parquet")

print("\n=== per supply level: variation across samples + decoupling from capacity ===", flush=True)
print(f"{'supply':>7} {'median':>8} {'CV%':>7} {'frac_supply_limited':>20} {'corr_realized_vs_cap':>22}")
for S, g in df.groupby("supply"):
    cv = 100 * g.realized.std() / g.realized.mean() if g.realized.mean() else 0
    frac_lim = (g.realized < 0.99 * g.stage1_cap).mean()      # capped by supply, not own capacity
    corr = np.corrcoef(g.realized, g.stage1_cap)[0, 1] if g.realized.std() > 0 else np.nan
    print(f"{S:>7} {g.realized.median():>8.1f} {cv:>7.1f} {frac_lim:>20.2f} {corr:>22.3f}")
# how well does min(S,cap) match MICOM?
err = (df.realized - df.expected_min).abs().max()
print(f"\nmax|realized - min(supply,capacity)| = {err:.3f}  (small => behaves as threshold model)")
print("saved -> data/processed/flux/substrate_sweep.parquet")
