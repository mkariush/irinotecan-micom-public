"""AGORA2 abundance coverage per sample (mSystems R1 major comment 1).

Reviewer 1 asked whether renormalizing to 1 after dropping low-abundance taxa inflates the
survivors. Our pipeline renormalizes TWICE, and the second step is much the larger:

  1. R/02_prepare_taxonomy.R drops taxa below 0.001 and renormalizes. NOTE the cutoff is applied
     to MetaPhlAn values on the PERCENT scale, so the operative threshold is 0.001% (1e-5 as a
     fraction), not the 0.1% our submitted Methods wording implied. Discarded mass is a median of
     0.0018% per sample (max 0.0113%) -> inflation factor <= 1.000113.
  2. MICOM's build drops taxa with no AGORA2 reconstruction and renormalizes the rest. THIS is the
     consequential one, and the submitted manuscript never reported its magnitude.

This script quantifies (2): the fraction of each sample's (already cutoff-filtered, renormalized)
abundance that maps to a species-level AGORA2 reconstruction. Output feeds:
  - the Methods coverage sentence (median 92.8%, 5th pct 74.3% across the 1,509 primary samples)
  - the total-community sensitivity analysis, via SENS_COVERAGE=1 in 10_results.py

Species key: AGORA2 manifest `id` is "Genus species strain", so the species-level key is
genus + " " + species, matching the `id` column of taxonomy_micom.parquet ("Genus species").

Run:  python python/33_agora2_coverage.py
Out:  data/processed/agora2_coverage.csv  (gitignored; regenerate with this script)
"""

import os
import pandas as pd

# Where the converted AGORA2 reconstructions live. AGORA2 is not redistributed with this
# repository (see the data-availability statement), so point AGORA2_DIR at your own copy:
#     set AGORA2_DIR=D:\databases\AGORA2_json      (Windows)
#     export AGORA2_DIR=~/databases/AGORA2_json    (POSIX)
AGORA2_DIR = os.environ.get(
    "AGORA2_DIR", r"databases\AGORA2_json")
MANIFEST = os.path.join(AGORA2_DIR, "manifest.csv")
TAXONOMY = "data/processed/taxonomy_micom.parquet"
OUT_CSV = "data/processed/agora2_coverage.csv"

if __name__ == "__main__":
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"AGORA2 manifest not found at {MANIFEST}. See python/03_build_models.py.")

    man = pd.read_csv(MANIFEST)
    species = set((man["genus"].astype(str) + " " + man["species"].astype(str)).unique())
    print(f"AGORA2 species-level keys: {len(species)} (from {len(man)} strain reconstructions)")

    # ALL cohorts (primary + depth-sensitivity), so the sensitivity run has no NaN rows.
    tax = pd.read_parquet(TAXONOMY).copy()
    tax["matched"] = tax["id"].isin(species)

    cov = (tax.groupby(["sample_id", "cohort", "cohort_set"])
              .apply(lambda g: pd.Series({
                  "cov_abund": g.loc[g.matched, "abundance"].sum(),   # matched share of abundance
                  "cov_taxa": g.matched.mean(),                       # matched share of taxa
              }), include_groups=False)
              .reset_index())

    # Abundances are renormalized to 1 per sample upstream, so coverage is a proper fraction.
    assert ((cov.cov_abund >= 0) & (cov.cov_abund <= 1 + 1e-9)).all(), "coverage outside [0,1]"

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cov.to_csv(OUT_CSV, index=False)

    print(f"\nsamples: {len(cov)} | cohorts: {cov.cohort.nunique()}")
    print("\nabundance coverage by cohort set:")
    print(cov.groupby("cohort_set")["cov_abund"]
             .describe()[["count", "50%", "min", "max"]].round(4).to_string())
    prim = cov[cov.cohort_set == "primary"]["cov_abund"]
    print(f"\nPRIMARY (n={len(prim)}): median {prim.median():.4f}, "
          f"5th pct {prim.quantile(0.05):.4f}, min {prim.min():.4f}")
    print(f"=> median {100 * (1 - prim.median()):.1f}% of community mass renormalized away at build "
          f"(95th pct {100 * (1 - prim.quantile(0.05)):.1f}%)")
    print("\nper-cohort median abundance coverage:")
    print(cov.groupby(["cohort_set", "cohort"])["cov_abund"].median().round(4).to_string())
    print(f"\nSaved -> {OUT_CSV}")
