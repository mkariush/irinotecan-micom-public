"""Simulate community growth under gut medium conditions."""

import os
import pandas as pd
import micom
from micom.workflows import grow
from micom.qiime_formats import load_qiime_medium

MANIFEST_PATH = "data/processed/model_manifest.csv"
MODELS_DIR    = "data/processed/models"
GROWTH_DIR    = "data/processed/growth"
TRADEOFF      = 0.5
THREADS       = 4

if __name__ == "__main__":
    os.makedirs(GROWTH_DIR, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Growing {len(manifest)} community models")

    # Load MICOM built-in gut medium (bundled with package)
    medium_path = os.path.join(os.path.dirname(micom.__file__), "data", "artifacts", "medium.qza")
    medium = load_qiime_medium(medium_path)

    # Add SN-38G at de facto unlimited rate (1,000 mmol/gDW/h) to measure
    # community reactivation CAPACITY — consistent with Heinken et al. 2023.
    # Medium uses _m suffix convention for exchange reactions.
    sn38g_row = pd.DataFrame(
        [{"reaction": "EX_sn38g_m", "flux": 1000.0, "metabolite": "sn38g_m"}],
        index=["EX_sn38g_m"]
    )
    medium = pd.concat([medium, sn38g_row])

    print(f"Medium reactions: {len(medium)} (includes SN-38G)")

    results = grow(
        manifest,
        model_folder=MODELS_DIR,
        medium=medium,
        tradeoff=TRADEOFF,
        threads=THREADS,
    )

    results.growth_rates.to_parquet(os.path.join(GROWTH_DIR, "growth_rates.parquet"))
    results.exchange_fluxes.to_parquet(os.path.join(GROWTH_DIR, "exchange_fluxes.parquet"))

    print(f"Growth simulation complete. Results written to {GROWTH_DIR}")
    print(f"  Samples: {results.growth_rates['sample_id'].nunique()}")
    print(f"  Median community growth: {results.growth_rates['growth_rate'].median():.4f}")
