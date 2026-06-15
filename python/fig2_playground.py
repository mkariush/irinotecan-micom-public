"""Fast Fig 2 (capacity vs carrier abundance, R2) playground -- iterate on STYLE without re-running
10_results.py.

Loads full_capacity.parquet + full_taxa_contributions.parquet + taxonomy (for carrier abundance).
Scatter of summed GUS-carrier abundance (x) vs SN-38 capacity (y) with the linear fit; points coloured
by cohort (same tab10 as Fig 1) so the P. copri / E. eligens below-line deviation (Yachida) is visible.
Edit the STYLE block, run, look at the PNG. When you like it, copy the settings into 10_results.py.

    python python/fig2_playground.py
"""

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- STYLE (edit me) ----------------
FIGSIZE         = (6, 5.5)
PALETTE         = "tab10"        # MUST match Fig 1
POINT_BY_COHORT = True           # True: colour points by cohort; False: neutral grey
POINT_SIZE      = 12
POINT_ALPHA     = 0.45
POINT_EDGE      = "none"         # "none" or e.g. "0.3"
POINT_LW        = 0.0
POINT_NEUTRAL   = "0.35"
SHOW_FIT        = True
FIT_COLOR       = "red"
FIT_LW          = 1.6
ANNOTATE        = True           # in-plot slope / r^2 text (instead of a title)
SHOW_LEGEND     = True
LEGEND_LOC      = "lower right"
SHOW_TITLE      = False
TITLE           = "Capacity reduces to summed β-glucuronidase-carrier abundance"
XLABEL          = "summed GUS-carrier abundance"
YLABEL          = "SN-38 reactivation capacity (relative units)"
THEME_CTX       = "paper"
OUT             = "data/processed/figures/results_R2_abundance_TEST.png"   # _TEST, not the real one
DPI             = 600
# -------------------------------------------------

PRIMARY_COHORTS = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
                   "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

cap = pd.read_parquet("data/processed/flux/full_capacity.parquet")
con = pd.read_parquet("data/processed/flux/full_taxa_contributions.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
cap = cap.merge(meta, on="sample_id", how="left")
cap = cap[cap.cohort.isin(PRIMARY_COHORTS)].copy()

# carrier abundance = summed abundance of taxa that carry non-zero reactivation flux (same as 10_results)
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
carriers = con[["sample_id", "taxon"]].drop_duplicates()
carr_ab = (carriers.merge(tax[["sample_id", "taxon", "abundance"]], on=["sample_id", "taxon"], how="left")
           .groupby("sample_id")["abundance"].sum().rename("carrier_abundance").reset_index())
r2 = cap.merge(carr_ab, on="sample_id", how="left").fillna({"carrier_abundance": 0.0})
m = r2.copy()   # all 1,509 primary; the 4 zero-carrier sit at the origin, on the line (0 = 100x0)

slope, intercept = np.polyfit(m.carrier_abundance, m.sn38_capacity, 1)
corr = np.corrcoef(m.carrier_abundance, m.sn38_capacity)[0, 1]

sns.set_theme(style="whitegrid", context=THEME_CTX)
order = cap.groupby("cohort")["sn38_capacity"].median().sort_values().index.tolist()
cohort_palette = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))

fig, ax = plt.subplots(figsize=FIGSIZE)
if POINT_BY_COHORT:
    for coh in order:
        g = m[m.cohort == coh]
        ax.scatter(g.carrier_abundance, g.sn38_capacity, s=POINT_SIZE, alpha=POINT_ALPHA,
                   color=cohort_palette[coh], edgecolor=POINT_EDGE, linewidth=POINT_LW,
                   label=coh, zorder=3)
else:
    ax.scatter(m.carrier_abundance, m.sn38_capacity, s=POINT_SIZE, alpha=POINT_ALPHA,
               color=POINT_NEUTRAL, edgecolor=POINT_EDGE, linewidth=POINT_LW, zorder=3)

if SHOW_FIT:
    xs = np.array([m.carrier_abundance.min(), m.carrier_abundance.max()])
    ax.plot(xs, slope * xs + intercept, color=FIT_COLOR, lw=FIT_LW, zorder=5)

if ANNOTATE:
    ax.text(0.04, 0.96, f"capacity = {slope:.0f} × carrier abundance\n$r^2$ = {corr**2:.3f}  (n = {len(m)})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
if SHOW_TITLE:
    ax.set_title(TITLE)
if SHOW_LEGEND and POINT_BY_COHORT:
    ax.legend(fontsize=7, loc=LEGEND_LOC, frameon=True, markerscale=1.5)
plt.tight_layout()
plt.savefig(OUT, dpi=DPI)
plt.close()
print(f"n={len(m)} / {m.cohort.nunique()} cohorts | slope={slope:.1f} r^2={corr**2:.4f} -> {OUT}")
