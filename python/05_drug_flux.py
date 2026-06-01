"""Extract SN-38 reactivation flux; add beta-glucuronidase reaction if absent from AGORA2."""

import os
import glob
import pandas as pd
import cobra

MODELS_DIR  = "data/processed/models"
GROWTH_DIR  = "data/processed/growth"
FLUX_DIR    = "data/processed/flux"

# Beta-glucuronidase identifiers to search for in AGORA2 models
BGUC_KEYWORDS = ["GUR", "BGUC", "GLUCUR", "glucuronid"]
BGUC_EC       = "3.2.1.31"
SN38G_METABOLITE_PATTERNS = ["sn38g", "SN38G", "cpd17595"]  # adjust if AGORA2 uses different IDs

os.makedirs(FLUX_DIR, exist_ok=True)


def find_bguc_reactions(model: cobra.Model) -> list:
    hits = []
    for r in model.reactions:
        id_hit  = any(kw.lower() in r.id.lower() for kw in BGUC_KEYWORDS)
        ec_hit  = BGUC_EC in (r.annotation.get("ec-code", "") or "")
        met_hit = any(
            any(p.lower() in m.id.lower() for p in SN38G_METABOLITE_PATTERNS)
            for m in r.metabolites
        )
        if id_hit or ec_hit or met_hit:
            hits.append(r)
    return hits


def check_agora2_coverage(n_models: int = 10) -> pd.DataFrame:
    """Screen a sample of AGORA2 single-organism models for bGUC activity."""
    model_files = glob.glob(os.path.join(MODELS_DIR, "**/*.xml"), recursive=True)
    if not model_files:
        model_files = glob.glob(os.path.join(MODELS_DIR, "**/*.json"), recursive=True)

    records = []
    for path in model_files[:n_models]:
        model = cobra.io.read_sbml_model(path)
        rxns  = find_bguc_reactions(model)
        records.append({
            "model":    os.path.basename(path),
            "n_bguc":   len(rxns),
            "rxn_ids":  [r.id for r in rxns],
        })
    return pd.DataFrame(records)


def extract_sn38_flux(exchange_fluxes: pd.DataFrame) -> pd.DataFrame:
    """Pull SN-38 exchange flux from grow() output."""
    sn38_cols = [c for c in exchange_fluxes.columns
                 if any(p.lower() in c.lower() for p in SN38G_METABOLITE_PATTERNS)]
    if not sn38_cols:
        raise ValueError(
            "No SN-38G columns found in exchange_fluxes. "
            "Either AGORA2 lacks the reaction (run check_agora2_coverage) or "
            "the metabolite ID pattern needs updating."
        )
    return exchange_fluxes[["sample_id"] + sn38_cols]


if __name__ == "__main__":
    print("=== Checking AGORA2 beta-glucuronidase coverage ===")
    coverage = check_agora2_coverage(n_models=20)
    print(coverage.to_string())
    print(f"\nModels with bGUC activity: {(coverage['n_bguc'] > 0).sum()} / {len(coverage)}")
    coverage.to_csv(os.path.join(FLUX_DIR, "bguc_coverage.csv"), index=False)

    if (coverage["n_bguc"] > 0).sum() == 0:
        print("\nWARNING: No bGUC reactions found. Manual COBRApy addition required.")
        print("See: https://cobrapy.readthedocs.io — add EC 3.2.1.31 reaction to relevant models")
    else:
        print("\n=== Extracting SN-38 flux from growth results ===")
        exchange_fluxes = pd.read_parquet(os.path.join(GROWTH_DIR, "exchange_fluxes.parquet"))
        sn38_flux = extract_sn38_flux(exchange_fluxes)
        sn38_flux.to_parquet(os.path.join(FLUX_DIR, "sn38_flux.parquet"))
        print(f"SN-38 flux extracted for {sn38_flux['sample_id'].nunique()} samples")
