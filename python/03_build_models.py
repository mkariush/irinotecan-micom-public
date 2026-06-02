"""Build per-sample MICOM community models from AGORA2."""

import os
import pandas as pd
from micom.workflows import build

TAXONOMY_PATH  = "data/processed/taxonomy_micom.parquet"
AGORA2_MANIFEST = r"databases\AGORA2_json"
MODELS_DIR     = "data/processed/models"
CUTOFF         = 0.001
THREADS        = 4
SOLVER         = "hybrid"   # QP-capable open-source solver (HiGHS + OSQP).
                            # Required for MICOM cooperative tradeoff; GLPK is LP-only.
                            # Switch to "gurobi" for the full production run once licensed.

# For testing, restrict to a small subset before full run.
# Overridable by env vars so the full overnight run doesn't edit this file:
#   MICOM_TEST_COHORT=None  MICOM_TEST_N=None   -> full 6-cohort run
TEST_COHORT    = os.environ.get("MICOM_TEST_COHORT", "YuJ_2015")
TEST_COHORT    = None if TEST_COHORT in ("", "None") else TEST_COHORT
_n             = os.environ.get("MICOM_TEST_N", "10")
TEST_N_SAMPLES = None if _n in ("", "None") else int(_n)
TEST_STRATIFY  = True         # balance TEST_N_SAMPLES across study_condition when subsetting

if __name__ == "__main__":
    if not os.path.exists(AGORA2_MANIFEST):
        raise FileNotFoundError(
            f"AGORA2 manifest not found at {AGORA2_MANIFEST}.\n"
            "Download AGORA2 models from Zenodo (see MICOM documentation) and "
            "update AGORA2_MANIFEST path."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)

    taxonomy = pd.read_parquet(TAXONOMY_PATH)

    if TEST_COHORT:
        taxonomy = taxonomy[taxonomy["cohort"] == TEST_COHORT]
    if TEST_N_SAMPLES:
        if TEST_STRATIFY and "study_condition" in taxonomy.columns:
            # balance N across conditions (e.g. 5 CRC + 5 control)
            samp = taxonomy[["sample_id", "study_condition"]].drop_duplicates()
            conds = samp["study_condition"].unique()
            per = max(1, TEST_N_SAMPLES // len(conds))
            keep = (samp.groupby("study_condition")["sample_id"]
                    .apply(lambda s: s.head(per)).tolist())
        else:
            keep = list(taxonomy["sample_id"].unique()[:TEST_N_SAMPLES])
        taxonomy = taxonomy[taxonomy["sample_id"].isin(keep)]
        print(f"Selected {len(keep)} samples"
              + (f" stratified by condition" if TEST_STRATIFY else ""))

    micom_tax = taxonomy[["sample_id", "id", "abundance"]].copy()
    micom_tax["genus"]   = micom_tax["id"].str.split().str[0]
    micom_tax["species"] = micom_tax["id"].str.split().str[1]

    print(f"Building models for {micom_tax['sample_id'].nunique()} samples, "
          f"{micom_tax['id'].nunique()} unique taxa")

    manifest = build(
        micom_tax,
        model_db=AGORA2_MANIFEST,
        out_folder=MODELS_DIR,
        cutoff=CUTOFF,
        threads=THREADS,
        solver=SOLVER,
    )

    manifest.to_csv("data/processed/model_manifest.csv", index=False)
    print(f"Built {len(manifest)} community models -> {MODELS_DIR}")
