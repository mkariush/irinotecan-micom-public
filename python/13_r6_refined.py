"""R6 refined: re-weight carriers by REAL per-species GUS class (from the targeted lookup).

Sources: Biernat 2019 (structures/kinetics; L1 enzymes efficiently process SN-38-G), Pellock 2018
(No-Loop BuGUS-1 ~ E. coli on SN-38-G), Candeliere 2022 (metagenome-wide GUS class per taxon).
Rule: a species reactivates via its BEST GUS (multi-GUS per species is common), so species
efficiency = class efficiency of its highest-efficiency characterized GUS.

Class -> relative SN-38-G efficiency (anchored to Pellock Fig 8B / Biernat):
  L1 = 1.0, NL = 1.0  (drug-reactivating classes)
  mL1 = 0.4, L2 = 0.4 (medium)
  mL2 = 0.05, NC = 0.0 (poor/none)
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr

CLASS_EFF = {"L1": 1.0, "NL": 1.0, "mL1": 0.4, "L2": 0.4, "mL2": 0.05, "NC": 0.0}

# best characterized GUS class per species (underscored names)
SPECIES_CLASS = {
    "Faecalibacterium_prausnitzii": "L1",   # L1 + NL (Biernat, Candeliere)
    "Eubacterium_eligens":          "L1",   # L1, "most active" (Biernat, Candeliere)
    "Escherichia_coli":             "L1",
    "Clostridium_perfringens":      "L1",
    "Bacteroides_uniformis":        "NL",   # BuGUS-1 (Pellock); NL (Candeliere)
    "Bacteroides_ovatus":           "NL",
    "Bacteroides_dorei":            "NL",
    "Bacteroides_massiliensis":     "NL",
    "Parabacteroides_merdae":       "NL",   # NL + L2; one mL2 nearly inactive (Biernat)
    "Bacteroides_vulgatus":         "mL1",  # dominant mL1-176 (Candeliere)
    "Bacteroides_fragilis":         "mL1",  # mL1 (Biernat)
    "Ruminococcus_gnavus":          "mL1",
    "Bacteroides_cellulosilyticus": "L2",   # L2 (Candeliere)
    "Lactobacillus_rhamnosus":      "L1",
}
# genus default for carriers not individually looked up
GENUS_DEFAULT = {"Bacteroides": 0.7, "Parabacteroides": 0.7, "Escherichia": 1.0,
                 "Faecalibacterium": 1.0, "Eubacterium": 0.7, "Clostridium": 0.7,
                 "Roseburia": 0.5, "Ruminococcus": 0.4, "Prevotella": 0.5,
                 "Paraprevotella": 0.5}
DEFAULT = 0.5


def eff(taxon):
    if taxon in SPECIES_CLASS:
        return CLASS_EFF[SPECIES_CLASS[taxon]]
    return GENUS_DEFAULT.get(taxon.split("_")[0], DEFAULT)


con  = pd.read_parquet("data/processed/flux/full_taxa_contributions.parquet")
tax  = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = pd.read_parquet("data/processed/sample_metadata.parquet")
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
carr = (con[["sample_id", "taxon"]].drop_duplicates()
        .merge(tax[["sample_id", "taxon", "abundance"]], on=["sample_id", "taxon"], how="left")
        .dropna(subset=["abundance"]))
carr["eff"] = carr["taxon"].map(eff)

# capacities
uni = carr.groupby("sample_id")["abundance"].sum().mul(100).rename("uni")
carr["w"] = carr["abundance"] * carr["eff"]
ref = carr.groupby("sample_id")["w"].sum().mul(100).rename("ref")
cap = pd.concat([uni, ref], axis=1).reset_index()

rho = spearmanr(cap["uni"], cap["ref"]).statistic
print(f"capacity median: uniform {cap.uni.median():.1f} -> refined {cap.ref.median():.1f}")
print(f"Spearman(uniform, refined) = {rho:.4f}  (1.0 = ranking unchanged)")

# driver taxa: uniform vs refined
def drivers(weighted):
    c = carr.copy()
    c["contrib"] = c["abundance"] * (c["eff"] if weighted else 1.0) * 100
    d = c.groupby("taxon")["contrib"].sum().sort_values(ascending=False).head(12)
    return [(t.replace("_", " "), round(v)) for t, v in d.items()]

print("\n=== top drivers: UNIFORM vs REFINED ===")
u, r = drivers(False), drivers(True)
print(f"{'rank':>4} {'uniform':38s} {'refined (class-weighted)':38s}")
for i in range(12):
    print(f"{i+1:>4} {u[i][0]:30s}{u[i][1]:>7} {r[i][0]:30s}{r[i][1]:>7}")

# R5 meta on refined capacity
d = cap.merge(meta[["sample_id", "cohort", "study_condition"]], on="sample_id", how="left")
d = d[d.study_condition.isin(["CRC", "control"])]
zs, ws = [], []
for coh, g in d.groupby("cohort"):
    a = g.loc[g.study_condition == "CRC", "ref"]; b = g.loc[g.study_condition == "control", "ref"]
    if len(a) < 3 or len(b) < 3: continue
    uu, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - 2 * uu / (len(a) * len(b))
    zs.append(norm.isf(min(max(p/2, 1e-12), 0.5)) * np.sign(rbc or 1)); ws.append(np.sqrt(len(a)+len(b)))
zs, ws = np.array(zs), np.array(ws)
zmeta = (zs * ws).sum() / np.sqrt((ws**2).sum())
print(f"\nR5 (refined) Stouffer meta: z={zmeta:.2f}, p={2*norm.sf(abs(zmeta)):.3g}, "
      f"dir={'control>CRC' if zmeta>0 else 'CRC>control'}")
