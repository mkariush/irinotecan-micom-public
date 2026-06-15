"""Oxygen-independence of SN-38 reactivation capacity, at scale (R3 oxygen claim).

The 45-sample A==B validation (09) tests the GROWTH constraint but not oxygen (both arms anaerobic).
This script tests the oxygen claim on the SAME stratified 45 samples: for each community, max EX_sn38_m
(Arm-A style, no growth requirement, saturating SN-38G) under two oxygen regimes:
  ANAEROBIC : EX_o2_m uptake = 0      (no oxygen)
  O2-FLOOD  : EX_o2_m uptake = 1000   (unlimited oxygen)
If the two are identical, reactivation is neither oxygen-limited nor oxygen-suppressed -- consistent with
a hydrolysis that neither consumes nor produces O2. Pure LP; sequential; resumable.

Run AFTER models are built. Output: data/processed/flux/o2_check.parquet (+ checkpoint).
"""
import os
import glob
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR  = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
OUT_PATH    = "data/processed/flux/o2_check.parquet"
CKPT_CSV    = OUT_PATH.replace(".parquet", "_checkpoint.csv")

SN38_EXCHANGE  = "EX_sn38_m"
SN38G_EXCHANGE = "EX_sn38g_m"
O2_EXCHANGE    = "EX_o2_m"
UNLIMITED      = 1000.0
N_PER_COHORT   = 5
REL_TOL        = 1e-3

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]


def _meta():
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    return tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")


def pick_subset() -> list:
    """Identical selection rule to 09_validate_AeqB.py -> the same 45 samples."""
    meta = _meta()
    meta = meta[meta.cohort.isin(PRIMARY)]
    built = {os.path.basename(p).replace(".pickle", "") for p in glob.glob(os.path.join(MODELS_DIR, "*.pickle"))}
    meta = meta[meta["sample_id"].isin(built)]
    return (meta.sort_values("sample_id")
            .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())


_BASE_MED = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))


def capacity(com, o2_level: float) -> float:
    """Max EX_sn38_m (no growth requirement), saturating SN-38G, with EX_o2_m set to o2_level."""
    with com:
        med = dict(_BASE_MED)
        med[SN38G_EXCHANGE] = UNLIMITED
        med[O2_EXCHANGE] = o2_level
        have = {r.id for r in com.exchanges}
        com.medium = {r: f for r, f in med.items() if r in have}
        com.variables.community_objective.lb = 0.0
        com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
        s = com.optimize()
        com.variables.community_objective.lb = 0.0
        return float(s.objective_value) if s is not None else 0.0


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    samples = pick_subset()
    done = set(pd.read_csv(CKPT_CSV)["sample_id"]) if os.path.exists(CKPT_CSV) else set()
    todo = [s for s in samples if s not in done]
    print(f"O2 check on {len(samples)} samples | {len(done)} done | {len(todo)} to do", flush=True)

    for i, s in enumerate(todo, 1):
        path = os.path.join(MODELS_DIR, s + ".pickle")
        if not os.path.exists(path):
            continue
        com = load_pickle(path)
        if SN38_EXCHANGE not in com.reactions:        # zero-carrier community
            anae = flood = 0.0
        else:
            anae = capacity(com, 0.0)
            flood = capacity(com, UNLIMITED)
        reldiff = abs(anae - flood) / max(anae, 1e-9)
        equal = bool(reldiff < REL_TOL)
        pd.DataFrame([{"sample_id": s, "cap_anaerobic": anae, "cap_o2flood": flood,
                       "rel_diff": reldiff, "O2_invariant": equal}]).to_csv(
            CKPT_CSV, mode="a", header=not os.path.exists(CKPT_CSV), index=False)
        print(f"  [{i}/{len(todo)}] {s}: anaerobic={anae:.3f} O2flood={flood:.3f} "
              f"reldiff={reldiff:.2e} {'OK' if equal else '*** DIFFERS ***'}", flush=True)

    df = pd.read_csv(CKPT_CSV).drop_duplicates("sample_id").merge(_meta(), on="sample_id", how="left")
    df.to_parquet(OUT_PATH)
    n_eq = int(df["O2_invariant"].sum())
    print(f"\n=== O2 invariance: {n_eq}/{len(df)} samples identical (rel_tol={REL_TOL}) ===")
    print("max rel_diff = %.3g" % df["rel_diff"].max())
    if n_eq == len(df):
        print("ALL identical -> reactivation capacity is oxygen-independent (anaerobic == O2-flood).")
    else:
        print(df[~df["O2_invariant"]][["sample_id", "cohort", "cap_anaerobic", "cap_o2flood", "rel_diff"]]
              .to_string(index=False))
    print(f"Saved -> {OUT_PATH}")
