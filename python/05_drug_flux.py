"""Extract SN-38 reactivation flux; add beta-glucuronidase reaction if absent from AGORA2."""

import os
import glob
import pandas as pd
import cobra

MODELS_DIR  = "data/processed/models"
GROWTH_DIR  = "data/processed/growth"
FLUX_DIR    = "data/processed/flux"

# Confirmed AGORA2 reaction and metabolite IDs (from JSON models)
BGUC_REACTIONS = {"SN38G_GLCAASE", "SN38G_GLCAASEe", "SN38G_GLCAASEepp"}
SN38_EXCHANGE  = "EX_sn38(e)"   # community SN-38 production flux
SN38G_EXCHANGE = "EX_sn38g(e)"  # SN-38G uptake flux

os.makedirs(FLUX_DIR, exist_ok=True)


def check_bguc_coverage(n_models: int = 20) -> pd.DataFrame:
    """Screen community models for SN38G_GLCAASE reaction coverage."""
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pickle"))
    records = []
    for path in model_files[:n_models]:
        from micom.util import load_pickle
        com = load_pickle(path)
        rxn_ids = {r.id.split("__")[0] for r in com.reactions}
        has_bguc = bool(rxn_ids & BGUC_REACTIONS)
        records.append({"model": os.path.basename(path), "has_bguc": has_bguc})
    return pd.DataFrame(records)


def extract_sn38_flux(exchange_fluxes: pd.DataFrame) -> pd.DataFrame:
    """Extract community SN-38 production flux (EX_sn38(e)) from grow() output."""
    if SN38_EXCHANGE not in exchange_fluxes.columns:
        available = [c for c in exchange_fluxes.columns if "sn38" in c.lower()]
        raise ValueError(
            f"Column '{SN38_EXCHANGE}' not found in exchange_fluxes.\n"
            f"Columns containing 'sn38': {available}\n"
            "Ensure EX_sn38g(e) was added to the medium in 04_grow.py."
        )
    return exchange_fluxes[["sample_id", SN38_EXCHANGE]].rename(
        columns={SN38_EXCHANGE: "sn38_flux"}
    )


if __name__ == "__main__":
    os.makedirs(FLUX_DIR, exist_ok=True)

    print("=== Extracting SN-38 reactivation flux ===")
    exchange_fluxes = pd.read_parquet(os.path.join(GROWTH_DIR, "exchange_fluxes.parquet"))

    print(f"Exchange flux columns containing 'sn38': "
          f"{[c for c in exchange_fluxes.columns if 'sn38' in c.lower()]}")

    sn38_flux = extract_sn38_flux(exchange_fluxes)
    sn38_flux.to_parquet(os.path.join(FLUX_DIR, "sn38_flux.parquet"))

    print(f"SN-38 flux extracted for {sn38_flux['sample_id'].nunique()} samples")
    print(f"Median flux: {sn38_flux['sn38_flux'].median():.6f} mmol/gDW/h")
    print(f"Samples with non-zero flux: {(sn38_flux['sn38_flux'] > 0).sum()}")
