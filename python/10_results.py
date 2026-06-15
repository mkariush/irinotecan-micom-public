"""R1-R5 results from the full 812-sample capacity run. Pure dataframe (no model loading).

Inputs (data/processed/flux/ and data/processed/):
  full_capacity.parquet            sample_id, sn38_capacity, n_gus_taxa
  full_taxa_contributions.parquet  sample_id, taxon, gus_flux
  sample_metadata.parquet          sample_id, cohort, study_condition
  taxonomy_micom.parquet           sample_id, id, abundance (for carrier abundance)

Outputs: console summary + data/processed/figures/results_*.png + data/processed/flux/results_summary.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FLUX = "data/processed/flux"
FIG  = "data/processed/figures"
sns.set_theme(style="whitegrid", context="talk")
os.makedirs(FIG, exist_ok=True)

# Primary cohorts for R1-R6 (expanded 2026-06-12: +YachidaS_2019, +ThomasAM_2019_c).
# Gupta/Hannigan are depth-sensitivity only (cohort_set=="sensitivity"); reported separately below.
PRIMARY_COHORTS = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
                   "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

cap_all = pd.read_parquet(f"{FLUX}/full_capacity.parquet")
con_all = pd.read_parquet(f"{FLUX}/full_taxa_contributions.parquet")

# per-sample metadata from the taxonomy table (authoritative for all 11 cohorts + cohort_set)
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
smeta = (tax[["sample_id", "cohort", "study_condition", "cohort_set"]]
         .drop_duplicates("sample_id"))
cap_all = cap_all.merge(smeta, on="sample_id", how="left")

# split: primary drives R1-R6; sensitivity (shallow cohorts) reported on its own
cap = cap_all[cap_all.cohort.isin(PRIMARY_COHORTS)].copy()
con = con_all[con_all.sample_id.isin(set(cap.sample_id))].copy()
cap_sens = cap_all[cap_all.cohort_set == "sensitivity"].copy()
print(f"PRIMARY: {len(cap)} samples / {cap.cohort.nunique()} cohorts | "
      f"SENSITIVITY: {len(cap_sens)} samples / {cap_sens.cohort.nunique()} cohorts\n")

# ---------- R1: distribution ----------
print("=" * 70)
print("R1 -- SN-38 reactivation capacity distribution (n=%d)" % len(cap))
print(cap["sn38_capacity"].describe().round(2).to_string())
nz = cap[cap.sn38_capacity > 0]
print(f"zero-capacity (no carriers): {(cap.sn38_capacity==0).sum()}")
print(f"fold-range among non-zero: {nz.sn38_capacity.max()/nz.sn38_capacity.min():.1f}x")
print("\nmedian capacity by cohort:")
print(cap.groupby("cohort")["sn38_capacity"].median().round(2).to_string())

# ---------- R2: capacity vs carrier abundance ----------
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
carriers = con[["sample_id", "taxon"]].drop_duplicates()
carr_ab = (carriers.merge(tax[["sample_id", "taxon", "abundance"]], on=["sample_id", "taxon"], how="left")
           .groupby("sample_id")["abundance"].sum().rename("carrier_abundance").reset_index())
r2 = cap.merge(carr_ab, on="sample_id", how="left").fillna({"carrier_abundance": 0.0})
m = r2[r2.carrier_abundance > 0]
slope, intercept = np.polyfit(m.carrier_abundance, m.sn38_capacity, 1)
corr = np.corrcoef(m.carrier_abundance, m.sn38_capacity)[0, 1]
print("\n" + "=" * 70)
print(f"R2 -- capacity ~ carrier abundance: slope={slope:.1f}, intercept={intercept:.2f}, r^2={corr**2:.4f}")

# ---------- R4: driver taxa ----------
drv = (con.groupby("taxon")
       .agg(total_flux=("gus_flux", "sum"), n_samples=("sample_id", "nunique"))
       .reset_index().sort_values("total_flux", ascending=False))
drv["pct_samples"] = (100 * drv.n_samples / con.sample_id.nunique()).round(1)
print("\n" + "=" * 70)
print("R4 -- top 15 driver taxa (by summed GUS flux):")
print(drv.head(15).to_string(index=False))

# ---------- R5: CRC vs control meta-analysis ----------
print("\n" + "=" * 70)
print("R5 -- CRC vs control (per cohort + pooled)")
sub = cap[cap.study_condition.isin(["CRC", "control"])].copy()
rows = []
for coh, g in sub.groupby("cohort"):
    crc = g.loc[g.study_condition == "CRC", "sn38_capacity"]
    ctl = g.loc[g.study_condition == "control", "sn38_capacity"]
    if len(crc) < 3 or len(ctl) < 3:
        continue
    u, p = mannwhitneyu(crc, ctl, alternative="two-sided")
    rbc = 1 - 2 * u / (len(crc) * len(ctl))            # rank-biserial (ctl>CRC positive)
    rows.append({"cohort": coh, "n_CRC": len(crc), "n_ctrl": len(ctl),
                 "med_CRC": round(crc.median(), 1), "med_ctrl": round(ctl.median(), 1),
                 "p": p, "rank_biserial": round(rbc, 3)})
meta_df = pd.DataFrame(rows)
print(meta_df.to_string(index=False))

# pooled: within-cohort z-score then MWU (cohort-adjusted)
sub["z"] = sub.groupby("cohort")["sn38_capacity"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1))
zc, zk = sub.loc[sub.study_condition == "CRC", "z"], sub.loc[sub.study_condition == "control", "z"]
u, p_pool = mannwhitneyu(zc, zk, alternative="two-sided")
# Stouffer combine of per-cohort p (two-sided, signed by rank-biserial)
if len(meta_df):
    zs = norm.isf(meta_df.p / 2) * np.sign(meta_df.rank_biserial)
    w = np.sqrt(meta_df.n_CRC + meta_df.n_ctrl)
    z_meta = (zs * w).sum() / np.sqrt((w**2).sum())
    p_meta = 2 * norm.sf(abs(z_meta))
    print(f"\nPooled (within-cohort z, MWU): p={p_pool:.4g}")
    print(f"Stouffer meta (n-weighted): z={z_meta:.2f}, p={p_meta:.4g}")
    print(f"direction: {'control > CRC' if z_meta>0 else 'CRC > control'} (signed by rank-biserial)")

# ---------- figures ----------
plt.figure(figsize=(12, 6))
order = cap.groupby("cohort")["sn38_capacity"].median().sort_values().index
cohort_palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(order))))
sns.violinplot(data=cap, x="cohort", y="sn38_capacity", order=order, hue="cohort",
               hue_order=order, palette=cohort_palette, saturation=0.35,
               cut=0, inner="quartile", legend=False)
sns.stripplot(data=cap, x="cohort", y="sn38_capacity", order=order, hue="cohort",
              hue_order=order, palette=cohort_palette, size=4, alpha=0.65,
              edgecolor="0.3", linewidth=0.2, legend=False)
plt.ylabel("SN-38 reactivation capacity (relative units)"); plt.xlabel(""); plt.xticks(rotation=30, ha="right")
plt.title("Predicted SN-38 reactivation capacity across cohorts (n=%d)" % len(cap)); plt.tight_layout()
plt.savefig(f"{FIG}/results_R1_by_cohort.png", dpi=200); plt.close()

plt.figure(figsize=(8, 7))
sns.regplot(data=m, x="carrier_abundance", y="sn38_capacity",
            scatter_kws=dict(s=15, alpha=0.4), line_kws=dict(color="red"), ci=None)
plt.xlabel("summed GUS-carrier abundance"); plt.ylabel("SN-38 capacity")
plt.title(f"R2: slope={slope:.0f}, r^2={corr**2:.3f}"); plt.tight_layout()
plt.savefig(f"{FIG}/results_R2_abundance.png", dpi=200); plt.close()

plt.figure(figsize=(10, 7))
top = drv.head(15)
sns.barplot(data=top, y="taxon", x="total_flux", color="tab:green")
plt.xlabel("total GUS flux (summed across samples)"); plt.ylabel("")
plt.title("R4: driver taxa"); plt.tight_layout()
plt.savefig(f"{FIG}/results_R4_driver_taxa.png", dpi=200); plt.close()

if len(sub):
    plt.figure(figsize=(11, 6))
    sns.violinplot(data=sub, x="cohort", y="sn38_capacity", hue="study_condition",
                   order=order, cut=0, split=False, inner="quartile")
    plt.ylabel("SN-38 capacity"); plt.xlabel(""); plt.xticks(rotation=30, ha="right")
    plt.title("R5: CRC vs control by cohort"); plt.legend(title="", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{FIG}/results_R5_crc_vs_control.png", dpi=200); plt.close()

# ---------- depth-sensitivity cohorts (Gupta, Hannigan): reported, NOT in primary ----------
if len(cap_sens):
    print("\n" + "=" * 70)
    print("DEPTH-SENSITIVITY cohorts (shallow; excluded from primary R1-R6)")
    print(cap_sens.groupby("cohort")["sn38_capacity"].describe()[["count", "50%", "min", "max"]]
          .round(2).to_string())
    print("compare medians to primary cohorts above; low capacity here may be depth-driven.")

# save summary
drv.to_csv(f"{FLUX}/results_driver_taxa.csv", index=False)
if len(meta_df): meta_df.to_csv(f"{FLUX}/results_crc_meta.csv", index=False)
cap_all[["sample_id", "cohort", "cohort_set", "study_condition", "sn38_capacity"]].to_csv(
    f"{FLUX}/results_capacity_by_sample.csv", index=False)
print(f"\nPRIMARY n={len(cap)} ({cap.cohort.nunique()} cohorts). "
      f"Figures -> {FIG}/results_*.png ; tables -> {FLUX}/results_*.csv")
