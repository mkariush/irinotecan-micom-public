"""Fast Fig 3 (driver taxa, R4) playground -- iterate on STYLE without touching fig3_panels.py.

Renders Panel A (drivers stacked by cohort) and/or Panel B (uniform vs class-weighted) to *_TEST files.
Panel A supports NORMALIZE: absolute stacked flux (False) or proportional 100% composition (True, which
removes the cohort-sample-count effect). Edit the STYLE block, run, look at the PNGs.

    python python/fig3_playground.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ---------------- STYLE (edit me) ----------------
RENDER_A   = True
RENDER_B   = True
TOPN       = 12
NORMALIZE  = False          # Panel A: False = absolute stacked flux; True = proportional (100%) composition
PALETTE    = "tab10"        # MUST match Fig 1
THEME_CTX  = "paper"
A_SIZE     = (5.5, 4)
B_SIZE     = (5.5, 4)
BAR_EDGE   = "white"        # separator between cohort segments in Panel A
BAR_LW     = 0.3
DEMOTE     = 0.7            # Panel B: weighted < DEMOTE x uniform -> crimson
A_LEGEND_LOC = "lower right"
B_LEGEND_LOC = "lower right"
FMT        = "svg"          # playground default; fig3_panels.py emits svg
DPI        = 600
OUT_A      = "data/processed/figures/results_R4_A_drivers_TEST"
OUT_B      = "data/processed/figures/results_R4_B_reweight_TEST"
# -------------------------------------------------

FLUX = "data/processed/flux"
PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]
CLASS_EFF = {"L1": 1.0, "NL": 1.0, "mL1": 0.4, "L2": 0.4, "mL2": 0.05, "NC": 0.0}
SPECIES_CLASS = {
    "Faecalibacterium_prausnitzii": "L1", "Eubacterium_eligens": "L1", "Escherichia_coli": "L1",
    "Clostridium_perfringens": "L1", "Bacteroides_uniformis": "NL", "Bacteroides_ovatus": "NL",
    "Bacteroides_dorei": "NL", "Bacteroides_massiliensis": "NL", "Parabacteroides_merdae": "NL",
    "Bacteroides_vulgatus": "mL1", "Bacteroides_fragilis": "mL1", "Ruminococcus_gnavus": "mL1",
    "Bacteroides_cellulosilyticus": "L2", "Lactobacillus_rhamnosus": "L1", "Prevotella_copri": "L1",
}
GENUS_DEFAULT = {"Bacteroides": 0.7, "Parabacteroides": 0.7, "Escherichia": 1.0,
                 "Faecalibacterium": 1.0, "Eubacterium": 0.7, "Clostridium": 0.7,
                 "Roseburia": 0.5, "Ruminococcus": 0.4, "Prevotella": 0.5, "Paraprevotella": 0.5}
def eff(t): return CLASS_EFF[SPECIES_CLASS[t]] if t in SPECIES_CLASS else GENUS_DEFAULT.get(t.split("_")[0], 0.5)
def cls(t): return SPECIES_CLASS.get(t, "?")

con = pd.read_parquet(f"{FLUX}/full_taxa_contributions.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
smeta = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
con = con.merge(smeta, on="sample_id", how="left")
con = con[con.cohort.isin(PRIMARY)]
cap = pd.read_parquet(f"{FLUX}/full_capacity.parquet").merge(smeta, on="sample_id", how="left")
order = (cap[cap.cohort.isin(PRIMARY)].groupby("cohort")["sn38_capacity"].median()
         .sort_values().index.tolist())
palette = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))
sns.set_theme(style="whitegrid", context=THEME_CTX)

totals = con.groupby("taxon")["gus_flux"].sum().sort_values(ascending=False)
top = totals.head(TOPN).index.tolist()

if RENDER_A:
    piv = (con[con.taxon.isin(top)].groupby(["taxon", "cohort"])["gus_flux"].sum()
           .unstack(fill_value=0).reindex(index=top))
    if NORMALIZE:
        piv = piv.div(piv.sum(axis=1), axis=0) * 100
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=A_SIZE)
    left = np.zeros(len(top))
    for coh in order:
        vals = piv[coh].values if coh in piv.columns else np.zeros(len(top))
        ax.barh(y, vals, left=left, color=palette[coh], edgecolor=BAR_EDGE, linewidth=BAR_LW, label=coh)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([t.replace("_", " ") for t in top]); ax.invert_yaxis()
    ax.set_xlabel("cohort share of β-glucuronidase flux (%)" if NORMALIZE
                  else "summed β-glucuronidase flux across samples (relative units; ≈ carriage × uniform cap)")
    ax.legend(fontsize=7, loc=A_LEGEND_LOC, title="cohort", title_fontsize=7, frameon=True)
    plt.tight_layout(); plt.savefig(f"{OUT_A}.{FMT}", dpi=DPI); plt.close()
    print(f"Panel A -> {OUT_A}.{FMT}  (NORMALIZE={NORMALIZE}, TOPN={TOPN})")

if RENDER_B:
    uni = totals.reindex(top).values.astype(float)
    wt = np.array([totals[t] * eff(t) for t in top], dtype=float)
    drop = wt < DEMOTE * uni
    y = np.arange(len(top)); h = 0.38
    fig, ax = plt.subplots(figsize=B_SIZE)
    ax.barh(y + h/2, uni, height=h, color="0.6")
    ax.barh(y - h/2, wt, height=h, color=["red" if d else "green" for d in drop])
    ax.set_yticks(y); ax.set_yticklabels([f"{t.replace('_', ' ')}  [{cls(t)}]" for t in top])
    ax.invert_yaxis()
    ax.set_xlabel("summed β-glucuronidase flux across samples (relative units)")
    leg = [Patch(facecolor="0.6", label="uniform"), Patch(facecolor="green", label="class-weighted")]
    if drop.any():
        leg.append(Patch(facecolor="red", label="class-weighted, demoted (<0.7× uniform)"))
    ax.legend(handles=leg, loc=B_LEGEND_LOC, fontsize=7.5, frameon=True)
    plt.tight_layout(); plt.savefig(f"{OUT_B}.{FMT}", dpi=DPI); plt.close()
    print(f"Panel B -> {OUT_B}.{FMT}  demoted: {[top[i].replace('_',' ') for i in np.where(drop)[0]]}")
