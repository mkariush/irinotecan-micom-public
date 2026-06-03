"""R6 Phase-0 sensitivity: does enzyme-class weighting change the conclusions?

Additivity => weighted capacity = 100 * sum_over_carriers(abundance * efficiency). Pure dataframe
on the contributions + taxonomy tables (no model loading). Test plausible efficiency schemes and
ask whether R1 ranking (Spearman vs uniform), R4 driver taxa, and R5 CRC-vs-control verdict change.

Efficiency anchored to Pellock Fig 8B (SN-38-G kcat/Km): E. coli & Bacteroides-type ~high; the
dominant Firmicute drivers (Faecalibacterium, Roseburia, Eubacterium, Clostridium) are NOT
characterized -> their SN-38-G efficiency is the key uncertainty we probe.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr

con  = pd.read_parquet("data/processed/flux/full_taxa_contributions.parquet")
tax  = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = pd.read_parquet("data/processed/sample_metadata.parquet")
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)

# carriers per sample WITH abundance
carr = (con[["sample_id", "taxon"]].drop_duplicates()
        .merge(tax[["sample_id", "taxon", "abundance"]], on=["sample_id", "taxon"], how="left")
        .dropna(subset=["abundance"]))
carr["genus"] = carr["taxon"].str.split("_").str[0]

# --- efficiency schemes: genus -> eff in [0,1] (default applied if genus not listed) ---
HIGH = {"Bacteroides": 1.0, "Parabacteroides": 1.0, "Escherichia": 1.0}
FIRM = ["Faecalibacterium", "Roseburia", "Eubacterium", "Clostridium", "Enterocloster",
        "Erysipelatoclostridium", "Ruminococcus", "Subdoligranulum", "Anaerostipes"]

def eff_uniform(g):       return 1.0
def eff_bact_firm_mid(g): return HIGH.get(g, 0.3 if g in FIRM else 0.5)
def eff_firm_zero(g):     return HIGH.get(g, 0.0 if g in FIRM else 0.5)
def eff_ecoli_high(g):    return 1.0 if g == "Escherichia" else 0.3

SCHEMES = {"uniform": eff_uniform, "bact_high_firm_mid": eff_bact_firm_mid,
           "firm_zero": eff_firm_zero, "ecoli_high": eff_ecoli_high}


def weighted_capacity(efffn):
    c = carr.copy()
    c["w"] = c["abundance"] * c["genus"].map(efffn)
    cap = c.groupby("sample_id")["w"].sum().mul(100).rename("cap").reset_index()
    return cap


def r5_meta(cap):
    d = cap.merge(meta[["sample_id", "cohort", "study_condition"]], on="sample_id", how="left")
    d = d[d.study_condition.isin(["CRC", "control"])]
    zs, ws = [], []
    for coh, g in d.groupby("cohort"):
        a = g.loc[g.study_condition == "CRC", "cap"]; b = g.loc[g.study_condition == "control", "cap"]
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1 - 2 * u / (len(a) * len(b))
        zs.append(norm.isf(min(max(p/2, 1e-12), 0.5)) * np.sign(rbc or 1)); ws.append(np.sqrt(len(a)+len(b)))
    zs, ws = np.array(zs), np.array(ws)
    zmeta = (zs * ws).sum() / np.sqrt((ws**2).sum())
    return zmeta, 2 * norm.sf(abs(zmeta))


# uniform reference
uni = weighted_capacity(eff_uniform).set_index("sample_id")["cap"]

print(f"{'scheme':20s} {'median':>7s} {'rho_vs_uni':>10s} {'R5_meta_p':>10s} {'R5_dir':>14s}  top3_drivers")
for name, fn in SCHEMES.items():
    cap = weighted_capacity(fn)
    rho = spearmanr(uni.reindex(cap.sample_id).values, cap["cap"].values).statistic
    zmeta, pmeta = r5_meta(cap)
    direction = "control>CRC" if zmeta > 0 else "CRC>control"
    # driver taxa under this weighting
    c = carr.copy(); c["contrib"] = c["abundance"] * c["genus"].map(fn) * 100
    top = (c.groupby("taxon")["contrib"].sum().sort_values(ascending=False).head(3).index.tolist())
    top = [t.replace("_", " ") for t in top]
    print(f"{name:20s} {cap['cap'].median():>7.1f} {rho:>10.3f} {pmeta:>10.3g} {direction:>14s}  "
          + "; ".join(top))

print("\nInterpretation: rho_vs_uni near 1 = sample ranking barely changes (uniform suffices).")
print("R5_meta_p staying >0.05 across schemes = the CRC-vs-control NULL is robust to enzyme weighting.")
print("If firm_zero changes things a lot, results hinge on the (uncharacterized) Firmicute drivers.")
