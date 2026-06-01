"""Simulate community growth under gut medium conditions."""

import os
import pandas as pd
from micom.workflows import grow
from micom.qiime_formats import load_qiime_medium

MANIFEST_PATH = "data/processed/model_manifest.csv"
MODELS_DIR    = "data/processed/models"
GROWTH_DIR    = "data/processed/growth"
TRADEOFF      = 0.5
THREADS       = 4
SOLVER        = "glpk"

if __name__ == "__main__":
    os.makedirs(GROWTH_DIR, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Growing {len(manifest)} community models")

    # Load MICOM built-in western diet gut medium
    medium = load_qiime_medium("western_diet_gut")

    # Add SN-38G to medium so bacteria can deconjugate it.
    # Flux = 1.0 mmol/gDW/h (saturating) measures community reactivation CAPACITY,
    # not absolute flux at a specific dose.
    medium["EX_sn38g(e)"] = 1.0

    results = grow(
        manifest,
        model_folder=MODELS_DIR,
        medium=medium,
        tradeoff=TRADEOFF,
        threads=THREADS,
        solver=SOLVER,
    )

    results.growth_rates.to_parquet(os.path.join(GROWTH_DIR, "growth_rates.parquet"))
    results.exchange_fluxes.to_parquet(os.path.join(GROWTH_DIR, "exchange_fluxes.parquet"))

    print(f"Growth simulation complete. Results written to {GROWTH_DIR}")
    print(f"  Samples: {results.growth_rates['sample_id'].nunique()}")
    print(f"  Median community growth: {results.growth_rates['growth_rate'].median():.4f}")
