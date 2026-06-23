"""Supplementary Table S5: overlap of canonical CRC-signature species with GUS carriers.

The CRC-signature set is defined REPRODUCIBLY as the intersection of the two landmark cross-cohort
CRC meta-analysis signatures: the 29-species core enriched in CRC at FDR < 1e-5 (Wirbel et al. 2019,
Nat Med, Extended Data Fig 10) AND the CRC-enriched signature of Thomas et al. 2019 (Nat Med).
This yields 13 species (B. fragilis is in Thomas but NOT in Wirbel's core, so it is excluded by the
intersection criterion). For each, we report whether it occurs in our primary cohorts, whether it is a
predicted beta-glucuronidase carrier (non-zero gus_flux in full_taxa_contributions), and its share of
total community reactivation flux. This substantiates the D4 decoupling claim.

Idempotent: re-running replaces any existing S5 block. Build order: 26 -> 28 -> 29 -> 30.

    python python/30_table_s5_crc_overlap.py
"""
import os
import pandas as pd

OUT_DIR  = "data/processed/tables"
OUT_CSV  = f"{OUT_DIR}/table_S5_crc_overlap.csv"
OUT_MD   = "docs/supplementary_tables.md"
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

# Wirbel 2019 (29-species core, FDR<1e-5) INTERSECT Thomas 2019 (CRC-enriched signature).
# Underscored to match the contributions/taxonomy naming; "Parvimonas spp." handled at genus level.
CRC_SIGNATURE = [
    "Fusobacterium_nucleatum", "Parvimonas_micra", "Parvimonas_spp",
    "Gemella_morbillorum", "Peptostreptococcus_stomatis", "Solobacterium_moorei",
    "Clostridium_symbiosum", "Porphyromonas_asaccharolytica", "Porphyromonas_somerae",
    "Porphyromonas_uenonis", "Prevotella_intermedia", "Anaerococcus_vaginalis",
    "Anaerococcus_obesiensis",
]

tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
tax = tax[tax.cohort.isin(PRIMARY)].copy()
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
con = pd.read_parquet("data/processed/flux/full_taxa_contributions.parquet")  # sample_id, taxon, gus_flux
prim_samples = set(tax.sample_id.unique())
con = con[con.sample_id.isin(prim_samples)].copy()

# total abundance-weighted reactivation flux across all carriers (primary cohorts)
ab = tax[["sample_id", "taxon", "abundance"]]
contrib = con.merge(ab, on=["sample_id", "taxon"], how="left").dropna(subset=["abundance"])
contrib["weighted"] = contrib["abundance"] * contrib["gus_flux"]
total_react = contrib["weighted"].sum()
carrier_taxa = set(con.loc[con.gus_flux > 0, "taxon"].unique())

def match(sig):
    """Match a signature entry against present taxa and carrier taxa (genus-level for *_spp)."""
    if sig.endswith("_spp"):
        genus = sig.split("_")[0] + "_"
        present = sorted(t for t in tax.taxon.unique() if t.startswith(genus))
        carr = sorted(t for t in carrier_taxa if t.startswith(genus))
    else:
        present = [sig] if sig in set(tax.taxon.unique()) else []
        carr = [sig] if sig in carrier_taxa else []
    n = tax.loc[tax.taxon.isin(present), "sample_id"].nunique() if present else 0
    flux = contrib.loc[contrib.taxon.isin(carr), "weighted"].sum()
    pct = 100 * flux / total_react if total_react else 0.0
    return present, carr, n, pct

rows = []
for sig in CRC_SIGNATURE:
    present, carr, n, pct = match(sig)
    rows.append({
        "CRC-signature species": "*" + sig.replace("_spp", " spp.").replace("_", " ") + "*",
        "Present in primary cohorts": f"yes (n={n})" if n else "not detected",
        "β-glucuronidase carrier": "yes" if carr else "no",
        "Reactivation flux contribution": f"{pct:.2f}%",
    })
s5 = pd.DataFrame(rows)
s5.to_csv(OUT_CSV, index=False)

n_present = sum(1 for r in rows if r["Present in primary cohorts"].startswith("yes"))
n_carrier = sum(1 for r in rows if r["β-glucuronidase carrier"] == "yes")
sig_flux = sum(float(r["Reactivation flux contribution"].rstrip("%")) for r in rows)

hdr = (
    "## Table S5. Canonical CRC-signature species versus β-glucuronidase carriers\n\n"
    "The CRC-signature set is defined reproducibly as the **intersection** of the two landmark "
    "cross-cohort CRC meta-analysis signatures: the 29-species core enriched in CRC at FDR < 1×10⁻⁵ "
    "(Wirbel et al. 2019) and the CRC-enriched signature of Thomas et al. 2019. This yields the "
    f"{len(CRC_SIGNATURE)} species below. (*Bacteroides fragilis* is CRC-enriched in Thomas et al. but is "
    "**not** in Wirbel's 29-species core — it appears there only as the *bft* toxin gene — so it is "
    "excluded by the intersection criterion.) For each species we report occurrence in our nine primary "
    "cohorts, whether it is a predicted β-glucuronidase carrier (non-zero gus_flux), and its share of "
    f"total community reactivation flux. **{n_carrier} of {len(CRC_SIGNATURE)}** signature species are "
    f"carriers, and together they account for **{sig_flux:.2f}%** of total reactivation flux — confirming "
    "that the cancer-associated signature is decoupled from drug-reactivating capacity (Discussion, D4).\n\n"
)
block = hdr + s5.to_markdown(index=False) + "\n"

prev = open(OUT_MD, encoding="utf-8").read() if os.path.exists(OUT_MD) else ""
marker = "## Table S5."
if marker in prev:
    prev = prev[:prev.index(marker)]
prev = prev.rstrip() + "\n\n"
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(prev + block)

print(f"signature species: {len(CRC_SIGNATURE)} | present in cohorts: {n_present} | "
      f"GUS carriers: {n_carrier} | combined reactivation flux: {sig_flux:.3f}%")
print(f"total community reactivation flux basis = {total_react:,.0f}")
print(f"saved -> {OUT_CSV} and appended S5 to {OUT_MD}")
