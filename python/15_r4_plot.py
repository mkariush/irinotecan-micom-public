"""R4 figure: driver taxa under UNIFORM vs ENZYME-CLASS-WEIGHTED efficiency (grouped bars).

Shows the top drivers, their GUS structural class, and how class-weighting changes contribution.
F. prausnitzii (L1+NL) unchanged at #1; B. vulgatus (mL1) demoted. Reuses 13_r6_refined weighting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_EFF = {"L1": 1.0, "NL": 1.0, "mL1": 0.4, "L2": 0.4, "mL2": 0.05, "NC": 0.0}
SPECIES_CLASS = {
    "Faecalibacterium_prausnitzii": "L1", "Eubacterium_eligens": "L1", "Escherichia_coli": "L1",
    "Clostridium_perfringens": "L1", "Bacteroides_uniformis": "NL", "Bacteroides_ovatus": "NL",
    "Bacteroides_dorei": "NL", "Bacteroides_massiliensis": "NL", "Parabacteroides_merdae": "NL",
    "Bacteroides_vulgatus": "mL1", "Bacteroides_fragilis": "mL1", "Ruminococcus_gnavus": "mL1",
    "Bacteroides_cellulosilyticus": "L2", "Lactobacillus_rhamnosus": "L1",
    "Prevotella_copri": "L1",  # L1-associated (Candeliere clustering)
}
GENUS_DEFAULT = {"Bacteroides": 0.7, "Parabacteroides": 0.7, "Escherichia": 1.0,
                 "Faecalibacterium": 1.0, "Eubacterium": 0.7, "Clostridium": 0.7,
                 "Roseburia": 0.5, "Ruminococcus": 0.4, "Prevotella": 0.5, "Paraprevotella": 0.5}
def eff(t):  return CLASS_EFF[SPECIES_CLASS[t]] if t in SPECIES_CLASS else GENUS_DEFAULT.get(t.split("_")[0], 0.5)
def cls(t):  return SPECIES_CLASS.get(t, "?")

con = pd.read_parquet("data/processed/flux/full_taxa_contributions.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
carr = (con[["sample_id", "taxon"]].drop_duplicates()
        .merge(tax[["sample_id", "taxon", "abundance"]], on=["sample_id", "taxon"], how="left").dropna())
carr["uni"] = carr["abundance"] * 100
carr["ref"] = carr["abundance"] * carr["taxon"].map(eff) * 100
agg = carr.groupby("taxon")[["uni", "ref"]].sum().sort_values("uni", ascending=False).head(12)

labels = [f"{t.replace('_',' ')}  [{cls(t)}]" for t in agg.index]
y = np.arange(len(agg))[::-1]; h = 0.38
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(y + h/2, agg["uni"], height=h, color="0.6", label="uniform")
# colour refined bars: red if it dropped notably, else teal
drop = agg["ref"] < 0.7 * agg["uni"]
ax.barh(y - h/2, agg["ref"], height=h,
        color=["crimson" if d else "teal" for d in drop], label="class-weighted")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("summed contribution to community SN-38 secretion (relative)")
ax.set_title("R4: driver taxa, uniform vs enzyme-class-weighted (GUS class in brackets)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data/processed/figures/results_R4_uniform_vs_weighted.png", dpi=200)
print(agg.round(0).astype(int).to_string())
print("\nNotably down-weighted (refined < 70% of uniform):", [t.replace('_',' ') for t in agg.index[drop]])
print("saved -> data/processed/figures/results_R4_uniform_vs_weighted.png")
