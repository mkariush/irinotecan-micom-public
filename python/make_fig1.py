"""Fig 1 -- predicted SN-38 reactivation capacity across cohorts (Results R1).

Violin + strip plot of per-sample capacity by cohort (tab10 palette, reused across figures).
Loads full_capacity.parquet + cohort labels. Edit the STYLE block to adjust appearance.
Emits data/processed/figures/results_R1_by_cohort.svg.

    python python/make_fig1.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- STYLE (edit me) ----------------
FIGSIZE      = (9, 4.5)
PALETTE      = "tab10"      # try: "Set2", "husl", "colorblind", "tab20"
VIOLIN_SAT   = 1         # violin colour vividness (0=grey, 1=full)
VIOLIN_ALPHA   = 0.2         # violin colour vividness (0=grey, 1=full)
POINT_SIZE   = 6
POINT_ALPHA  = 0.65
POINT_EDGE   = "0.3"
POINT_LW     = 0.2
SHOW_TITLE   = False
TITLE        = "Predicted SN-38 reactivation capacity across cohorts"
YLABEL       = "SN-38 reactivation capacity (relative units)"
THEME_CTX    = "paper"       # "notebook" | "talk" | "paper" | "poster"
OUT          = "data/processed/figures/results_R1_by_cohort.svg"
DPI          = 600
# -------------------------------------------------

PRIMARY_COHORTS = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
                   "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

cap = pd.read_parquet("data/processed/flux/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
cap = cap.merge(meta, on="sample_id", how="left")
cap = cap[cap.cohort.isin(PRIMARY_COHORTS)].copy()

sns.set_theme(style="whitegrid", context=THEME_CTX)
order = cap.groupby("cohort")["sn38_capacity"].median().sort_values().index
palette = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))

plt.figure(figsize=FIGSIZE)
sns.violinplot(data=cap, x="cohort", y="sn38_capacity", order=order, hue="cohort",
               hue_order=order, palette=palette, saturation=VIOLIN_SAT, alpha =VIOLIN_ALPHA, 
               cut=0, inner="quartile", legend=False)
sns.stripplot(data=cap, x="cohort", y="sn38_capacity", order=order, hue="cohort",
              hue_order=order, palette=palette, size=POINT_SIZE, alpha=POINT_ALPHA,
              edgecolor=POINT_EDGE, linewidth=POINT_LW, legend=False)
plt.ylabel(YLABEL); plt.xlabel(""); plt.xticks(rotation=30, ha="right")
if SHOW_TITLE:
    plt.title(TITLE)
plt.tight_layout()
plt.savefig(OUT, dpi=DPI)
plt.close()
print(f"n={len(cap)} / {cap.cohort.nunique()} cohorts -> {OUT}")
