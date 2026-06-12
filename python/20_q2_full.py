"""Q2 FULL: is growth-coupled, carbon-limited SN-38 reactivation NON-ADDITIVE?

Probe (script 18/19) showed: under carbon limitation the community GROWS MORE with SN-38G available
(+116%), i.e. it harvests SN-38G's glucuronate as carbon. Flux exploration (b9y2f335c) showed the
INCIDENTAL EX_sn38_m in that growth solution is TINY (0.177) vs the max-secretion ceiling (49.8):
realized reactivation in the growing community is governed by growth dynamics, not capacity/supply.

This script measures incidental SN-38 across a compositional spread of communities and asks:
  - Is incidental reactivation predicted by carrier abundance / capacity?  (corr ~1  => ADDITIVE, Q2 null)
  - Or does it deviate, tracking community GROWTH/structure instead?       (corr low => NON-ADDITIVE, Q2 GO)

Method per sample (stable regime from the probe: moderate carbon limitation, NOT starvation):
  medium = western_diet_gut * SCALE,  EX_sn38g_m = SUPPLY (finite),
  sol = cooperative_tradeoff(fraction=0.5, fluxes=True, pfba=True)   # growth solution + flux distn
  incidental_sn38 = sol.fluxes.loc["medium", "EX_sn38_m"]            # community readout
  ceiling         = min(SUPPLY, stage1_capacity)                     # additive/threshold prediction

RESUMABLE: every sample is appended to the checkpoint CSV immediately. If the run is interrupted
(machine closed/slept/killed), just re-run the SAME command -- it skips samples already in the CSV
and continues. Sequential + gc to stay memory-safe (no OOM like the 4-worker run).
"""
import os, gc, glob, sys
import numpy as np
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR = "data/processed/models"
MEDIUM     = "data/media/western_diet_gut_agora.qza"
OUT        = "data/processed/flux/q2_full.csv"
SN38, SN38G = "EX_sn38_m", "EX_sn38g_m"
SCALE   = 0.10          # moderate carbon limitation -> growth solvable (~0.01-0.05); 0.05 hung the QP
SUPPLY  = 50.0          # finite SN-38G supply (>= typical capacities, so ceiling = capacity)
N_TARGET = 40           # samples, stratified across the capacity range for compositional spread

base = dict(zip(*[load_qiime_medium(MEDIUM)[c] for c in ("reaction", "flux")]))
cap1 = pd.read_parquet("data/processed/flux/full_capacity.parquet").set_index("sample_id")["sn38_capacity"]

# --- pick a stratified, compositionally diverse sample set among models present on disk ---
have_models = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{MODELS_DIR}/*.pickle")}
pool = cap1[(cap1.index.isin(have_models)) & (cap1 > 0)].sort_values()
if len(pool) > N_TARGET:                       # even spacing across the capacity range
    idx = np.linspace(0, len(pool) - 1, N_TARGET).round().astype(int)
    pool = pool.iloc[np.unique(idx)]
samples = list(pool.index)

# --- resume: skip samples already checkpointed ---
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT)["sample"].astype(str))
todo = [s for s in samples if s not in done]
print(f"{len(samples)} target samples | {len(done)} done | {len(todo)} to run "
      f"(SCALE={SCALE}, SUPPLY={SUPPLY})", flush=True)

def append_row(row):
    df = pd.DataFrame([row])
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)

for i, sid in enumerate(todo, 1):
    p = f"{MODELS_DIR}/{sid}.pickle"
    growth = incidental = float("nan"); status = "err"
    try:
        com = load_pickle(p)
        have = {r.id for r in com.exchanges}
        with com:
            med = {r: f * SCALE for r, f in base.items()}
            med[SN38G] = SUPPLY                          # SN-38G NOT scaled (finite supply)
            com.medium = {r: f for r, f in med.items() if r in have}
            sol = com.cooperative_tradeoff(fraction=0.5, fluxes=True, pfba=True)
            growth = float(sol.growth_rate)
            incidental = float(sol.fluxes.loc["medium", SN38])
            status = str(sol.status)
        del com
    except Exception as e:
        status = f"ERR:{repr(e)[:70]}"
    gc.collect()
    capacity = float(cap1.get(sid, float("nan")))
    ceiling  = min(SUPPLY, capacity)
    frac     = incidental / ceiling if ceiling else float("nan")
    append_row({"sample": sid, "scale": SCALE, "supply": SUPPLY, "capacity": round(capacity, 3),
                "ceiling": round(ceiling, 3), "growth": round(growth, 5),
                "incidental_sn38": round(incidental, 5), "frac_of_ceiling": round(frac, 5),
                "status": status})
    print(f"[{i}/{len(todo)}] {sid}: growth={growth:.4f}  incidental={incidental:.4f}  "
          f"ceiling={ceiling:.2f}  frac={frac:.4f}  ({status})", flush=True)

# --- summary (only meaningful once all/most are done) ---
if os.path.exists(OUT):
    d = pd.read_csv(OUT)
    ok = d[d["incidental_sn38"].notna() & d["status"].astype(str).str.startswith("optimal")]
    print(f"\n=== Q2 full summary: {len(ok)}/{len(d)} optimal ===", flush=True)
    if len(ok) >= 5:
        from scipy.stats import spearmanr, pearsonr
        rs_cap, p_cap = spearmanr(ok.incidental_sn38, ok.capacity)
        rs_gr,  p_gr  = spearmanr(ok.incidental_sn38, ok.growth)
        print(f"incidental vs CAPACITY (carrier abundance): Spearman rho={rs_cap:.3f} (p={p_cap:.2g})")
        print(f"incidental vs GROWTH:                        Spearman rho={rs_gr:.3f} (p={p_gr:.2g})")
        print(f"incidental/ceiling: median={ok.frac_of_ceiling.median():.4f} "
              f"range {ok.frac_of_ceiling.min():.4f}-{ok.frac_of_ceiling.max():.4f}")
        print("\nINTERPRET: incidental tracks capacity (rho~1) => additive/threshold (Q2 null).")
        print("           incidental tracks growth, NOT capacity => NON-ADDITIVE (Q2 GO).")
