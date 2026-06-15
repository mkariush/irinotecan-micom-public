"""Supplementary Table S1: Arm A == Arm B reactivation capacity on the 45-sample validation subset.

Since A==B is exact, it gets no main figure (a plot would be trivially flat); instead we present the
per-sample Arm A / Arm B values as a supplementary table. Reads AeqB_validation.parquet (09).
Emits a CSV (supplementary data) and a markdown rendering (for the supplement draft).

    python python/26_table_s1_AeqB.py
"""
import os
import pandas as pd

SRC      = "data/processed/flux/AeqB_validation.parquet"
OUT_DIR  = "data/processed/tables"
OUT_CSV  = f"{OUT_DIR}/table_S1_AeqB.csv"
OUT_MD   = "docs/supplementary_tables.md"

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_parquet(SRC).copy()
df["abs_diff"] = (df["arm_a"] - df["arm_b"]).abs()
df["cohort"] = pd.Categorical(df["cohort"], categories=PRIMARY, ordered=True)
df = df.sort_values(["cohort", "sample_id"]).reset_index(drop=True)

tab = pd.DataFrame({
    "Sample ID":            df["sample_id"],
    "Cohort":               df["cohort"].astype(str),
    "Condition":            df["study_condition"],
    "Community growth (h-1)": df["growth"].round(3),
    "Arm A capacity":       df["arm_a"].round(2),
    "Arm B capacity":       df["arm_b"].round(2),
    "|A - B|":              df["abs_diff"].round(3),
    "Relative difference":  df["rel_diff"].round(6),
})
tab.to_csv(OUT_CSV, index=False)

# markdown rendering
n = len(tab)
n_agree = int(df["A_eq_B"].sum())
hdr = ("# Supplementary tables\n\n"
       "## Table S1. Growth-constrained (Arm B) versus unconstrained (Arm A) SN-38 reactivation capacity\n\n"
       f"Per-sample reactivation capacity (relative units) for the stratified {n}-sample validation "
       "subset (five per primary cohort). **Arm A** is the unconstrained maximum SN-38 secretion; "
       "**Arm B** is the same maximization subject to the community retaining ≥95% of its "
       "cooperative-tradeoff growth solution (community growth rate shown). The growth rate is non-zero "
       "in every sample, so the constraint is real, yet Arm A and Arm B are identical in all "
       f"{n_agree}/{n} samples (relative difference = 0; |A−B| = 0) — reactivation is neither "
       "growth- nor oxygen-limited under saturating substrate (Results, R3). Capacity is in relative "
       "units (uniform AGORA2 bound, not a measured Vmax; Methods).\n\n")

md = hdr + tab.to_markdown(index=False)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md + "\n")

print(f"n={n} | A==B agree: {n_agree}/{n} | max|A-B|={df['abs_diff'].max():.3g} | "
      f"max rel_diff={df['rel_diff'].max():.3g}")
print(f"saved -> {OUT_CSV}  and  {OUT_MD}")
