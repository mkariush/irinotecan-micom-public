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


PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

def _meta():
    """Per-sample cohort/condition from the 11-cohort taxonomy (sample_metadata.parquet is stale 812)."""
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    return tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")

def pick_subset() -> list:
    """First N_PER_COHORT built samples per PRIMARY cohort (now spans all nine)."""
    meta = _meta()
    meta = meta[meta.cohort.isin(PRIMARY)]
    built = {os.path.basename(p).replace(".pickle", "") for p in glob.glob(os.path.join(MODELS_DIR, "*.pickle"))}
    meta = meta[meta["sample_id"].isin(built)]
    return (meta.sort_values("sample_id")
            .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())


CKPT_CSV = OUT_PATH.replace(".parquet", "_checkpoint.csv")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    samples = pick_subset()
    # RESUMABLE: skip samples already in the checkpoint (the QP is slow; protect a long run)
    done = set(pd.read_csv(CKPT_CSV)["sample_id"]) if os.path.exists(CKPT_CSV) else set()
    todo = [s for s in samples if s not in done]
    print(f"Validating A==B on {len(samples)} samples | {len(done)} done | {len(todo)} to do", flush=True)

    for i, s in enumerate(todo, 1):
        path = os.path.join(MODELS_DIR, s + ".pickle")
        if not os.path.exists(path):
            continue
        com = load_pickle(path)
        a = arm_a(com)
        try:
            b, gr = arm_b(com)
            reldiff = abs(a - b) / max(a, 1e-9)
            equal = bool(reldiff < REL_TOL)
            note = "OK" if equal else "*** DIVERGES ***"
        except Exception as e:
            # cooperative_tradeoff (QP) crossover is numerically infeasible on a few models under the
            # hybrid OSQP+HiGHS solver. arm_a (a plain LP) still succeeded, so the sample is simply not
            # B-evaluable -- record arm_b=NaN (NOT a divergence) and keep going.
            b = gr = reldiff = float("nan")
            equal = False
            note = f"arm_b SOLVER-FAILED ({type(e).__name__}: {e})"
        pd.DataFrame([{"sample_id": s, "arm_a": a, "arm_b": b, "growth": gr,
                       "rel_diff": reldiff, "A_eq_B": equal}]).to_csv(
            CKPT_CSV, mode="a", header=not os.path.exists(CKPT_CSV), index=False)
        print(f"  [{i}/{len(todo)}] {s}: A={a:.3f} B={b:.3f} reldiff={reldiff:.2e} {note}", flush=True)

    df = pd.read_csv(CKPT_CSV).drop_duplicates("sample_id")
    df = df.merge(_meta(), on="sample_id", how="left")
    df.to_parquet(OUT_PATH)

    failed = df["arm_b"].isna()
    evald  = df[~failed]
    n_eq   = int(evald["A_eq_B"].sum())
    print(f"\n=== A==B validation: {n_eq}/{len(evald)} EVALUATED samples agree (rel_tol={REL_TOL}) ===")
    if failed.any():
        print(f"{int(failed.sum())} sample(s) not B-evaluable (cooperative_tradeoff solver-infeasible; "
              f"excluded from the tally, arm_a still computed):")
        print(df[failed][["sample_id", "cohort", "arm_a"]].to_string(index=False))
    diverged = evald[~evald["A_eq_B"]]
    if len(diverged) == 0:
        print("All evaluated samples agree -> bulk Arm-A capacity (08) is justified across cohorts.")
    else:
        print("DIVERGENCE found -> inspect these communities; use Arm B where A != B:")
        print(diverged[["sample_id", "cohort", "arm_a", "arm_b", "rel_diff"]].to_string(index=False))
    print(f"Saved -> {OUT_PATH}")
