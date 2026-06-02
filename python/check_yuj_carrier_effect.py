"""Option A: fast robustness check of CRC vs control SN-38 reactivation in ALL YuJ samples.

Relies on the validated additivity result (capacity ~= 104 * summed GUS-carrier abundance).
Instead of building/optimizing 128 community models, we sum the relative abundance of
GUS-carrier species (those whose AGORA2 representative encodes SN38G_GLCAASE) per sample
and compare CRC vs control with a Wilcoxon rank-sum test.

Carrier = the first-alphabetical AGORA2 strain for that species (matches build pipeline),
checked for an SN38G_GLCAASE* reaction by raw-text scan of the JSON (fast, no cobra parse).
"""

import os
import glob
import pandas as pd
from scipy.stats import mannwhitneyu

TAXONOMY_PATH = "data/processed/taxonomy_micom.parquet"
META_PATH     = "data/processed/sample_metadata.parquet"
AGORA2_DIR    = r"databases\AGORA2_json"
MANIFEST      = os.path.join(AGORA2_DIR, "manifest.csv")
COHORT        = "YuJ_2015"
SLOPE         = 104.0   # capacity ~= SLOPE * carrier abundance (from fig3)
OUT_PATH      = "data/processed/flux/yuj_carrier_effect.parquet"


def carrier_species(species_list) -> set:
    """Return the subset of species whose representative AGORA2 model has SN38G_GLCAASE."""
    man = pd.read_csv(MANIFEST)
    man["species_name"] = man["genus"] + " " + man["species"]
    carriers = set()
    checked = 0
    for sp in species_list:
        rows = man[man["species_name"] == sp].sort_values("id")
        if rows.empty:
            continue
        strain_id = rows["id"].iloc[0]                       # first-alphabetical strain
        fpath = os.path.join(AGORA2_DIR, strain_id.replace(" ", "_") + ".json")
        if not os.path.exists(fpath):
            # fallback: glob by genus_species prefix
            gpref = sp.replace(" ", "_")
            cand = sorted(glob.glob(os.path.join(AGORA2_DIR, gpref + "*.json")))
            if not cand:
                continue
            fpath = cand[0]
        try:
            with open(fpath, "rb") as fh:
                if b"SN38G_GLCAASE" in fh.read():
                    carriers.add(sp)
        except Exception:
            continue
        checked += 1
    print(f"Scanned {checked} representative models; {len(carriers)} carrier species")
    return carriers


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tax = pd.read_parquet(TAXONOMY_PATH)
    tax = tax[tax["cohort"] == COHORT]
    meta = pd.read_parquet(META_PATH)

    species = sorted(tax["id"].unique())
    print(f"YuJ: {tax['sample_id'].nunique()} samples, {len(species)} unique species")
    carriers = carrier_species(species)

    tax["is_carrier"] = tax["id"].isin(carriers)
    per_sample = (tax[tax["is_carrier"]]
                  .groupby("sample_id")["abundance"].sum()
                  .rename("carrier_abundance").reset_index())
    # samples with zero carriers won't appear above; add them back as 0
    allids = tax[["sample_id"]].drop_duplicates()
    per_sample = allids.merge(per_sample, on="sample_id", how="left").fillna({"carrier_abundance": 0.0})
    per_sample["predicted_capacity"] = SLOPE * per_sample["carrier_abundance"]
    per_sample = per_sample.merge(meta[["sample_id", "study_condition"]], on="sample_id", how="left")

    per_sample.to_parquet(OUT_PATH)

    print("\n=== Carrier abundance & predicted capacity by condition ===")
    g = per_sample.groupby("study_condition")["predicted_capacity"]
    print(g.agg(["count", "median", "mean", "min", "max"]).round(2))

    crc = per_sample.loc[per_sample.study_condition == "CRC", "predicted_capacity"]
    ctl = per_sample.loc[per_sample.study_condition == "control", "predicted_capacity"]
    if len(crc) and len(ctl):
        u, p = mannwhitneyu(crc, ctl, alternative="two-sided")
        print(f"\nWilcoxon rank-sum (CRC vs control): U={u:.1f}, p={p:.4g}")
        print(f"CRC median {crc.median():.2f}  vs  control median {ctl.median():.2f}")
        direction = "control > CRC" if ctl.median() > crc.median() else "CRC > control"
        print(f"Direction: {direction}")
    print(f"\nSaved -> {OUT_PATH}")
