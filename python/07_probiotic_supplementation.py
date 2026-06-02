"""Secondary analysis: in silico probiotic supplementation.

For each sample, add a probiotic taxon at a target relative abundance, rebuild the
community, and recompute SN-38 reactivation capacity (Arm B: growth-constrained
max EX_sn38_m). Compare to baseline (no probiotic).

Contrast arms:
  - Lactobacillus plantarum : GUS-NEGATIVE (verified no SN38G_GLCAASE in AGORA2) ->
    expected to DECREASE reactivation by competing for resources / diluting carriers.
  - Bifidobacterium longum  : GUS-POSITIVE (known producer) ->
    may INCREASE reactivation by adding a reactivating taxon.

This demonstrates that not all probiotics are protective. Secondary / proof-of-concept.
"""

import os
import shutil
import pandas as pd
from micom.workflows import build
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

TAXONOMY_PATH   = "data/processed/taxonomy_micom.parquet"
META_PATH       = "data/processed/sample_metadata.parquet"
AGORA2_MANIFEST = r"databases\AGORA2_json"
MEDIUM_PATH     = "data/media/western_diet_gut_agora.qza"
WORK_DIR        = "data/processed/probiotic_work"
OUT_PATH        = "data/processed/flux/probiotic_supplementation.parquet"

SN38_EXCHANGE   = "EX_sn38_m"
SN38G_EXCHANGE  = "EX_sn38g_m"
UNLIMITED       = 1000.0
TRADEOFF        = 0.5
GROWTH_FRACTION = 0.95
CUTOFF          = 0.001
SOLVER          = "hybrid"

# Probiotic contrast: display name -> AGORA2 "genus species"
PROBIOTICS = {
    "L. plantarum (GUS-)": "Lactobacillus plantarum",
    "B. longum (GUS+)":    "Bifidobacterium longum",
}
ENRICH_LEVELS = [0.05]          # POC: 5%. Extend to [0.01, 0.05, 0.10] for full run.

# POC scope
TEST_COHORT     = "ZellerG_2014"
TEST_N_SAMPLES  = 2


def make_taxonomy(base: pd.DataFrame, probiotic_species=None, frac=0.0) -> pd.DataFrame:
    """Return a per-sample taxonomy; optionally add a probiotic at relative abundance `frac`."""
    df = base[["sample_id", "id", "abundance"]].copy()
    if probiotic_species and frac > 0:
        df["abundance"] = df["abundance"] * (1.0 - frac)          # scale existing down
        add = pd.DataFrame([{"sample_id": s, "id": probiotic_species, "abundance": frac}
                            for s in df["sample_id"].unique()])
        df = pd.concat([df, add], ignore_index=True)
    df["genus"]   = df["id"].str.split().str[0]
    df["species"] = df["id"].str.split().str[1]
    return df


def capacity(com) -> float:
    """Arm B: max SN-38 secretion while holding cooperative-tradeoff growth (anaerobic)."""
    med = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))
    med[SN38G_EXCHANGE] = UNLIMITED
    have = {r.id for r in com.exchanges}
    com.medium = {r: f for r, f in med.items() if r in have}
    sol = com.cooperative_tradeoff(fraction=TRADEOFF)
    com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
    com.variables.community_objective.lb = GROWTH_FRACTION * sol.growth_rate
    s = com.optimize()
    com.variables.community_objective.lb = 0.0
    return float(s.objective_value) if s is not None else 0.0


def build_and_score(tax: pd.DataFrame, tag: str) -> dict:
    """Build community models for `tax` and return {sample_id: capacity}."""
    outdir = os.path.join(WORK_DIR, tag)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    manifest = build(tax, model_db=AGORA2_MANIFEST, out_folder=outdir,
                     cutoff=CUTOFF, threads=2, solver=SOLVER)
    out = {}
    for _, row in manifest.iterrows():
        com = load_pickle(os.path.join(outdir, row["file"]))
        out[row["sample_id"]] = capacity(com)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tax = pd.read_parquet(TAXONOMY_PATH)
    meta = pd.read_parquet(META_PATH)
    if TEST_COHORT:
        tax = tax[tax["cohort"] == TEST_COHORT]
    if TEST_N_SAMPLES:
        keep = tax["sample_id"].unique()[:TEST_N_SAMPLES]
        tax = tax[tax["sample_id"].isin(keep)]
    cond = dict(zip(meta["sample_id"], meta["study_condition"]))

    rows = []
    # baseline (no probiotic)
    print("=== baseline ===")
    base_cap = build_and_score(make_taxonomy(tax), "baseline")
    for s, c in base_cap.items():
        rows.append({"sample_id": s, "condition": cond.get(s), "probiotic": "none",
                     "enrich": 0.0, "capacity": c, "delta": 0.0, "pct_change": 0.0})

    # each probiotic x enrichment
    for label, species in PROBIOTICS.items():
        for f in ENRICH_LEVELS:
            tag = f"{label.split()[0]}_{int(f*100)}pct".replace(".", "")
            print(f"=== {label} @ {f:.0%} ===")
            cap = build_and_score(make_taxonomy(tax, species, f), tag)
            for s, c in cap.items():
                b = base_cap.get(s, float("nan"))
                rows.append({"sample_id": s, "condition": cond.get(s), "probiotic": label,
                             "enrich": f, "capacity": c, "delta": c - b,
                             "pct_change": 100 * (c - b) / b if b else float("nan")})

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH)
    print("\n=== Probiotic supplementation (SN-38 reactivation capacity) ===")
    print(df.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")
