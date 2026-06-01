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

    # Load AGORA-matched western diet gut medium (171 metabolites).
    # NOTE: the package-bundled medium.qza is a 4-metabolite demo and yields
    # zero growth — do not use it. This file is from github.com/micom-dev/media.
    medium_path = "data/media/western_diet_gut_agora.qza"
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
        presolve=True,   # recovers samples that fail cooperative tradeoff on hybrid/OSQP
    )

    # GrowthResults fields are: growth_rates, exchanges, annotations.
    # `exchanges` holds the per-taxon exchange fluxes incl. SN-38 reactivation.
    results.growth_rates.to_parquet(os.path.join(GROWTH_DIR, "growth_rates.parquet"))
    results.exchanges.to_parquet(os.path.join(GROWTH_DIR, "exchange_fluxes.parquet"))

    print(f"Growth simulation complete. Results written to {GROWTH_DIR}")
    print(f"  Samples: {results.growth_rates['sample_id'].nunique()}")
    print(f"  Median community growth: {results.growth_rates['growth_rate'].median():.4f}")
