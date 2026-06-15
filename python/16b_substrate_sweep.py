"""Substrate sweep on the SAME stratified 45-sample subset used for A==B (09_validate_AeqB.py).

Supersedes 16_substrate_sweep.py, which selected its 42 communities from the STALE pre-expansion
sample_metadata.parquet (six cohorts only). This version:
  - derives the subset from taxonomy_micom.parquet + the nine PRIMARY cohorts (the fixed pattern), and
  - reuses the identical selection rule as 09 (five built samples per primary cohort, by sample_id),
so the substrate sweep and the Arm A vs Arm B validation are provably the same 45 samples.

Cheap: pure LP re-optimization of EX_sn38_m at ten supply bounds (no QP). Sequential, load once.
Resumable via a per-sample checkpoint CSV.

Run AFTER models are built and full_capacity.parquet exists.
"""

import os
import glob
import gc
import numpy as np
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR  = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
CAP_PATH    = "data/processed/flux/full_capacity.parquet"
OUT_PATH    = "data/processed/flux/substrate_sweep_45.parquet"
CKPT_CSV    = OUT_PATH.replace(".parquet", "_checkpoint.csv")

SN38, SN38G = "EX_sn38_m", "EX_sn38g_m"
SUPPLY      = [1000, 90, 60, 40, 25, 15, 8, 4, 2, 1]      # saturating -> limiting
N_PER_COHORT = 5                                          # MATCHES 09 -> same 45 samples

# Same nine primary cohorts and selection rule as 09_validate_AeqB.py:
PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]


def _meta():
    """Per-sample cohort from the 11-cohort taxonomy (sample_metadata.parquet is the stale 812)."""
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    return tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")


def pick_subset() -> list:
    """First N_PER_COHORT built samples per PRIMARY cohort -- identical rule to 09.pick_subset()."""
    meta = _meta()
    meta = meta[meta.cohort.isin(PRIMARY)]
    built = {os.path.basename(p).replace(".pickle", "") for p in glob.glob(os.path.join(MODELS_DIR, "*.pickle"))}
    meta = meta[meta["sample_id"].isin(built)]
    return (meta.sort_values("sample_id")
            .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    base_med = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))
    cap1 = pd.read_parquet(CAP_PATH).set_index("sample_id")["sn38_capacity"]

    samples = pick_subset()
    done = set(pd.read_csv(CKPT_CSV)["sample_id"]) if os.path.exists(CKPT_CSV) else set()
    todo = [s for s in samples if s not in done]
    print(f"Sweeping {len(samples)} samples x {len(SUPPLY)} supply levels "
          f"| {len(done)} done | {len(todo)} to do", flush=True)

    for i, s in enumerate(todo, 1):
        path = os.path.join(MODELS_DIR, s + ".pickle")
        if not os.path.exists(path):
            continue
        com = load_pickle(path)
        have = {r.id for r in com.exchanges}
        if SN38 not in com.reactions:
            del com; gc.collect(); continue
        com.objective = com.reactions.get_by_id(SN38)
        com.variables.community_objective.lb = 0.0
        for S in SUPPLY:
            med = dict(base_med); med[SN38G] = S
            com.medium = {r: f for r, f in med.items() if r in have}
            sol = com.optimize()
            realized = float(sol.objective_value) if sol is not None else 0.0
            pd.DataFrame([{"sample_id": s, "supply": S, "realized": realized}]).to_csv(
                CKPT_CSV, mode="a", header=not os.path.exists(CKPT_CSV), index=False)
        print(f"  [{i}/{len(todo)}] {s} swept", flush=True)
        del com; gc.collect()

    df = pd.read_csv(CKPT_CSV).drop_duplicates(["sample_id", "supply"])
    df = df.merge(_meta(), on="sample_id", how="left")
    df["stage1_cap"]   = df["sample_id"].map(cap1)
    df["expected_min"] = np.minimum(df["supply"], df["stage1_cap"])
    df.to_parquet(OUT_PATH)

    print(f"\n=== {df.sample_id.nunique()} samples across {df.cohort.nunique()} cohorts ===", flush=True)
    print(f"{'supply':>7} {'median':>8} {'CV%':>7} {'frac_supply_limited':>20} {'corr_realized_vs_cap':>22}")
    for S, g in df.groupby("supply"):
        cv   = 100 * g.realized.std() / g.realized.mean() if g.realized.mean() else 0
        frac = (g.realized < 0.99 * g.stage1_cap).mean()
        corr = np.corrcoef(g.realized, g.stage1_cap)[0, 1] if g.realized.std() > 0 else np.nan
        print(f"{S:>7} {g.realized.median():>8.1f} {cv:>7.1f} {frac:>20.2f} {corr:>22.3f}")

    err = (df.realized - df.expected_min).abs().max()
    print(f"\nmax|realized - min(supply,capacity)| = {err:.4f}  (small => threshold model holds)")
    # Spearman of realized-vs-capacity at the limiting end (for R6 decoupling number):
    from scipy.stats import spearmanr
    sat = df[df.supply == 1000]; lim = df[df.supply == 1]
    print(f"Spearman(realized, capacity): saturating={spearmanr(sat.realized, sat.stage1_cap).correlation:.3f}  "
          f"limiting={spearmanr(lim.realized, lim.stage1_cap).correlation:.3f}")
    print(f"CV%: saturating={100*sat.realized.std()/sat.realized.mean():.1f}  "
          f"limiting={100*lim.realized.std()/lim.realized.mean() if lim.realized.mean() else 0:.1f}")
    print(f"saved -> {OUT_PATH}")
