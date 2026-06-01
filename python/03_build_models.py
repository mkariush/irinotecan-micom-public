"""Build per-sample MICOM community models from AGORA2."""

import os
import pandas as pd
from micom.workflows import build

TAXONOMY_PATH  = "data/processed/taxonomy_micom.parquet"
AGORA2_MANIFEST = r"databases\AGORA2_SBML\manifest.csv"
MODELS_DIR     = "data/processed/models"
CUTOFF         = 0.001
THREADS        = 4
SOLVER         = "glpk"

# For testing, restrict to a small subset before full run
TEST_COHORT    = None   # set to e.g. "ZellerG_2014" to limit scope
TEST_N_SAMPLES = None   # set to e.g. 10 to limit to first N samples

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
    keep = taxonomy["sample_id"].unique()[:TEST_N_SAMPLES]
    taxonomy = taxonomy[taxonomy["sample_id"].isin(keep)]

# MICOM build expects: sample_id, id (taxon), abundance
micom_tax = taxonomy[["sample_id", "id", "abundance"]].copy()

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
