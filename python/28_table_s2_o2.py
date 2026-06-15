"""Supplementary Table S2: oxygen-independence of SN-38 reactivation capacity (45-sample subset).

Reads o2_check.parquet (27_o2_check.py) and appends Table S2 to docs/supplementary_tables.md
(idempotent: replaces any existing S2 section) + writes the CSV. Run AFTER 26 (S1) and 27 (o2 check).

    python python/28_table_s2_o2.py
"""
import os
import pandas as pd

SRC     = "data/processed/flux/o2_check.parquet"
OUT_DIR = "data/processed/tables"
OUT_CSV = f"{OUT_DIR}/table_S2_o2.csv"
OUT_MD  = "docs/supplementary_tables.md"
MARKER  = "## Table S2."

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_parquet(SRC).copy()
df["abs_diff"] = (df["cap_anaerobic"] - df["cap_o2flood"]).abs()
df["cohort"] = pd.Categorical(df["cohort"], categories=PRIMARY, ordered=True)
df = df.sort_values(["cohort", "sample_id"]).reset_index(drop=True)

tab = pd.DataFrame({
    "Sample ID":                 df["sample_id"],
    "Cohort":                    df["cohort"].astype(str),
    "Condition":                 df["study_condition"],
    "Capacity (anaerobic)":      df["cap_anaerobic"].round(2),
    "Capacity (O2-flooded)":     df["cap_o2flood"].round(2),
    "|anaerobic - O2|":          df["abs_diff"].round(3),
    "Relative difference":       df["rel_diff"].round(6),
})
tab.to_csv(OUT_CSV, index=False)

n = len(tab); n_eq = int(df["O2_invariant"].sum())
section = (f"{MARKER} Oxygen-independence of SN-38 reactivation capacity\n\n"
           f"Per-sample reactivation capacity (relative units) for the same stratified {n}-sample subset "
           "(five per primary cohort) under **anaerobic** (no oxygen uptake) versus **oxygen-flooded** "
           "(unlimited O2) conditions, both maximizing SN-38 secretion at saturating substrate without a "
           f"growth requirement (Arm A). Capacity is identical in {n_eq}/{n} samples (relative difference "
           f"= {df['rel_diff'].max():.3g}; |anaerobic − O2| = {df['abs_diff'].max():.3g}), so reactivation "
           "is neither oxygen-limited nor oxygen-suppressed — consistent with a hydrolysis that neither "
           "consumes nor produces oxygen (Results, R3).\n\n" + tab.to_markdown(index=False) + "\n")

prev = ""
if os.path.exists(OUT_MD):
    prev = open(OUT_MD, encoding="utf-8").read()
head = prev.split(MARKER)[0].rstrip() if MARKER in prev else prev.rstrip()
if not head:
    head = "# Supplementary tables"          # safety if S1 not present yet
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(head + "\n\n" + section)

print(f"n={n} | O2-invariant: {n_eq}/{n} | max|diff|={df['abs_diff'].max():.3g} | "
      f"max rel_diff={df['rel_diff'].max():.3g}")
print(f"saved -> {OUT_CSV}  and appended Table S2 to {OUT_MD}")
