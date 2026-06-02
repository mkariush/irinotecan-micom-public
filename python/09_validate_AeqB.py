"""Validate A == B across all cohorts before trusting the bulk Arm-A capacity (08).

For a stratified subset (~N_PER_COHORT samples x 6 cohorts), compute BOTH:
  Arm A: unconstrained max EX_sn38_m (single LP)          -- what 08 uses for all samples
  Arm B: max EX_sn38_m s.t. cooperative-tradeoff growth   -- the growth-constrained value
and check whether they agree within tolerance. If they do across all cohorts, the bulk
Arm-A run is justified ("realized == theoretical"). If any diverge, that community is a
finding (growth limits reactivation) and needs Arm B.

Run AFTER models are built (uses existing pickles in MODELS_DIR).
"""

import os
import glob
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR  = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
META_PATH   = "data/processed/sample_metadata.parquet"
OUT_PATH    = "data/processed/flux/AeqB_validation.parquet"

SN38_EXCHANGE  = "EX_sn38_m"
SN38G_EXCHANGE = "EX_sn38g_m"
UNLIMITED      = 1000.0
TRADEOFF       = 0.5
GROWTH_FRACTION = 0.95
N_PER_COHORT   = 5
REL_TOL        = 1e-3        # |A-B| / max(A, eps) below this == "equal"


def _set_medium(com):
    med = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))
    med[SN38G_EXCHANGE] = UNLIMITED
    have = {r.id for r in com.exchanges}
    com.medium = {r: f for r, f in med.items() if r in have}


def arm_a(com) -> float:
    with com:
        _set_medium(com)
        com.variables.community_objective.lb = 0.0
        com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
        s = com.optimize()
        com.variables.community_objective.lb = 0.0
        return float(s.objective_value) if s is not None else 0.0


def arm_b(com):
    with com:
        _set_medium(com)
        sol = com.cooperative_tradeoff(fraction=TRADEOFF)
        gr = sol.growth_rate
        com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
        com.variables.community_objective.lb = GROWTH_FRACTION * gr
        s = com.optimize()
        com.variables.community_objective.lb = 0.0
        return (float(s.objective_value) if s is not None else 0.0), gr


def pick_subset() -> list:
    """First N_PER_COHORT built samples per cohort."""
    meta = pd.read_parquet(META_PATH)
    built = {os.path.basename(p).replace(".pickle", "") for p in glob.glob(os.path.join(MODELS_DIR, "*.pickle"))}
    meta = meta[meta["sample_id"].isin(built)]
    return (meta.sort_values("sample_id")
            .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    samples = pick_subset()
    print(f"Validating A==B on {len(samples)} samples across cohorts")

    rows = []
    for i, s in enumerate(samples, 1):
        path = os.path.join(MODELS_DIR, s + ".pickle")
        if not os.path.exists(path):
            continue
        com = load_pickle(path)
        a = arm_a(com)
        b, gr = arm_b(com)
        reldiff = abs(a - b) / max(a, 1e-9)
        equal = reldiff < REL_TOL
        rows.append({"sample_id": s, "arm_a": a, "arm_b": b, "growth": gr,
                     "rel_diff": reldiff, "A_eq_B": equal})
        print(f"  [{i}/{len(samples)}] {s}: A={a:.3f} B={b:.3f} reldiff={reldiff:.2e} {'OK' if equal else '*** DIVERGES ***'}")

    df = pd.DataFrame(rows)
    df = df.merge(pd.read_parquet(META_PATH)[["sample_id", "cohort", "study_condition"]],
                  on="sample_id", how="left")
    df.to_parquet(OUT_PATH)

    n_eq = int(df["A_eq_B"].sum())
    print(f"\n=== A==B validation: {n_eq}/{len(df)} samples agree (rel_tol={REL_TOL}) ===")
    if n_eq == len(df):
        print("ALL agree -> bulk Arm-A capacity (08) is justified across cohorts.")
    else:
        print("DIVERGENCE found -> inspect these communities; use Arm B where A != B:")
        print(df[~df["A_eq_B"]][["sample_id", "cohort", "arm_a", "arm_b", "rel_diff"]].to_string(index=False))
    print(f"Saved -> {OUT_PATH}")
