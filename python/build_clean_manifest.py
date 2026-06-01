"""
Screen AGORA2 models for cobra readability.
For each species, find the first strain that loads without error.
Writes a clean manifest usable by MICOM.
"""

import os
import sys
import pandas as pd
import cobra
from concurrent.futures import ThreadPoolExecutor, as_completed

AGORA2_DIR       = r"databases\AGORA2_SBML"
MANIFEST_ALL     = os.path.join(AGORA2_DIR, "manifest_all_strains.csv")
MANIFEST_OUT     = os.path.join(AGORA2_DIR, "manifest.csv")
THREADS          = 4


def can_load(filepath: str) -> bool:
    try:
        cobra.io.read_sbml_model(filepath)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    full = pd.read_csv(MANIFEST_ALL)
    full["filepath"] = full["file"].apply(lambda f: os.path.join(AGORA2_DIR, f))

    species_groups = list(full.groupby(["genus", "species"]))
    print(f"Screening {len(full)} strains across {len(species_groups)} species ...")

    clean_rows = []
    skipped    = []

    # Screen first strain per species; fall back to alternatives only if needed
    first_pass = full.drop_duplicates(subset=["genus", "species"], keep="first")
    print(f"Testing {len(first_pass)} representative strains ...")

    results = {}
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(can_load, row.filepath): row for _, row in first_pass.iterrows()}
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            results[(row.genus, row.species)] = (future.result(), row)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(first_pass)} tested ...")

    failed_species = [(g, s) for (g, s), (ok, _) in results.items() if not ok]
    print(f"First-choice failures: {len(failed_species)}")

    # For failed species, try remaining strains
    fallback_results = {}
    for genus, species in failed_species:
        candidates = full[(full.genus == genus) & (full.species == species)].iloc[1:]
        found = False
        for _, row in candidates.iterrows():
            if can_load(row.filepath):
                fallback_results[(genus, species)] = row
                found = True
                break
        if not found:
            skipped.append(f"{genus} {species}")

    # Build clean manifest
    for (genus, species), (ok, row) in results.items():
        if ok:
            clean_rows.append(row)
        elif (genus, species) in fallback_results:
            clean_rows.append(fallback_results[(genus, species)])

    clean = pd.DataFrame(clean_rows)[["id", "genus", "species", "strain", "file", "summary_rank"]]
    clean.to_csv(MANIFEST_OUT, index=False)

    print(f"\nClean manifest: {len(clean)} species with loadable models")
    print(f"Skipped (no valid strain): {len(skipped)}")
    if skipped:
        print("  " + "\n  ".join(skipped[:20]))
    print(f"Written to {MANIFEST_OUT}")
