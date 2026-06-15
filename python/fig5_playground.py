"""Fast Fig 5 (CRC-vs-control by cohort) playground -- iterate on STYLE without re-running 10_results.py.

Loads only full_capacity.parquet + cohort/condition labels (no contributions, no stats). ~4 s/run.
Encoding: cohort = fill colour (same tab10 as Fig 1); condition = texture (control solid, CRC hatched).
Edit the STYLE block, run, look at the PNG. When you like it, copy the settings into 10_results.py.

    python python/fig5_playground.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ---------------- STYLE (edit me) ----------------
FIGSIZE      = (12, 4.5)
PALETTE      = "tab10"       # MUST match Fig 1 for consistency
VIOLIN_ALPHA = 0.2         # fill opacity of the violins
VIOLIN_EDGE  = "0.25"       # outline + hatch colour
VIOLIN_LW    = 0.8          # violin outline width
HATCH        = "////"      # CRC texture; try "///", "\\\\", "xxxx", "...."
INNER        = "quartile"   # "quartile" | "box" | "point" | None
THEME_CTX    = "paper"       # "notebook" | "talk" | "paper" | "poster"
SHOW_TITLE   = False
TITLE        = "SN-38 reactivation capacity, CRC vs control by cohort"
YLABEL       = "SN-38 reactivation capacity (relative units)"
LEGEND_LOC   = "upper left"
OUT          = "data/processed/figures/results_R5_crc_vs_control_TEST.png"   # _TEST, not the real one
DPI          = 600
# points overlay
SHOW_POINTS     = True
POINT_BY_COHORT = True       # True: points coloured by cohort (like Fig 1); False: neutral grey
POINT_SIZE      = 2.5
POINT_ALPHA     = 0.55
POINT_EDGE      = "0.25"
POINT_LW        = 0.15
POINT_JITTER    = 0.08
POINT_NEUTRAL   = "0.3"
# -------------------------------------------------

PRIMARY_COHORTS = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
                   "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

cap = pd.read_parquet("data/processed/flux/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")
cap = cap.merge(meta, on="sample_id", how="left")
cap = cap[cap.cohort.isin(PRIMARY_COHORTS)].copy()
sub = cap[cap.study_condition.isin(["CRC", "control"])].copy()

sns.set_theme(style="whitegrid", context=THEME_CTX)
order = cap.groupby("cohort")["sn38_capacity"].median().sort_values().index
cohort_palette = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))

fig, ax = plt.subplots(figsize=FIGSIZE)
sns.violinplot(data=sub, x="cohort", y="sn38_capacity", hue="study_condition",
               hue_order=["control", "CRC"], order=order, split=True,
               cut=0, inner=INNER, linewidth=VIOLIN_LW, ax=ax,
               palette={"control": "0.8", "CRC": "0.8"})
# cohort = colour; condition = texture (control solid, CRC hatched)
for coll in ax.collections:
    paths = coll.get_paths()
    if not paths:
        continue
    xc = paths[0].vertices[:, 0].mean()
    ci = int(round(xc))
    if 0 <= ci < len(order):
        coll.set_facecolor(cohort_palette[order[ci]])
        coll.set_edgecolor(VIOLIN_EDGE)
        coll.set_alpha(VIOLIN_ALPHA)
        if xc > ci:                      # right half = CRC (hue_order control, CRC)
            coll.set_hatch(HATCH)

# points overlay -- dodged so control lands on the left half, CRC on the right half
n_violin = len(ax.collections)
if SHOW_POINTS:
    sns.stripplot(data=sub, x="cohort", y="sn38_capacity", hue="study_condition",
                  hue_order=["control", "CRC"], order=order, dodge=True, jitter=POINT_JITTER,
                  size=POINT_SIZE, linewidth=POINT_LW, edgecolor=POINT_EDGE, ax=ax,
                  legend=False, palette={"control": POINT_NEUTRAL, "CRC": POINT_NEUTRAL})
    if POINT_BY_COHORT:
        for coll in ax.collections[n_violin:]:        # only the point collections
            offs = coll.get_offsets()
            if len(offs) == 0:
                continue
            cols = []
            for x, _ in offs:
                ci = int(round(x))
                cols.append(cohort_palette[order[ci]] if 0 <= ci < len(order) else (0.3, 0.3, 0.3))
            coll.set_facecolor(cols)
            coll.set_alpha(POINT_ALPHA)

ax.set_ylabel(YLABEL); ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
if SHOW_TITLE:
    ax.set_title(TITLE)
legend_elems = [Patch(facecolor="0.8", edgecolor=VIOLIN_EDGE, label="control"),
                Patch(facecolor="0.8", edgecolor=VIOLIN_EDGE, hatch=HATCH, label="CRC")]
ax.legend(handles=legend_elems, title="", loc=LEGEND_LOC, fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig(OUT, dpi=DPI)
plt.close()
print(f"n={len(sub)} ({sub.study_condition.value_counts().to_dict()}) / "
      f"{sub.cohort.nunique()} cohorts -> {OUT}")
