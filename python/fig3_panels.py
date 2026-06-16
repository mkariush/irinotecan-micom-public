"""Fig 3 (driver taxa, R4) as two standalone SVG panels for external (Inkscape) composition.

  A  results_R4_A_drivers.svg   top driver taxa by summed SN-38 secretion, each bar STACKED by cohort
                                (Fig-1 tab10 palette) -> shows the drivers are pan-cohort commensals.
  B  results_R4_B_reweight.svg  the same taxa, uniform vs enzyme-class-weighted contribution
                                (GUS class in brackets; demoted taxa, e.g. B. vulgatus mL1, in crimson).

Both panels rank by actual summed flux (the realized-driver metric, as in R4); panel B's weighting is
that flux x GUS-class efficiency (reuses 13_r6_refined / 15_r4_plot classes). Primary cohorts only.
NOTE: stacked-bar segment size scales with cohort sample count (YachidaS_2019 n=616 is largest), so
segment area reflects total contribution, not per-sample rate.

    python python/fig3_panels.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ---------------- STYLE ----------------
FMT       = "svg"            # "svg" | "pdf" | "png"
PALETTE   = "tab10"          # MUST match Fig 1
THEME_CTX = "paper"
DPI       = 600
TOPN      = 12
A_SIZE    = (7.5, 6)
B_SIZE    = (7.5, 6)
# ---------------------------------------

FIGDIR = "data/processed/figures"
FLUX   = "data/processed/flux"
PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

# GUS structural class -> SN-38-G efficiency (13_r6_refined / 15_r4_plot)
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


def panel_A(fname):
    piv = (con[con.taxon.isin(top)].groupby(["taxon", "cohort"])["gus_flux"].sum()
           .unstack(fill_value=0).reindex(index=top))
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=A_SIZE)
    left = np.zeros(len(top))
    for coh in order:
        vals = piv[coh].values if coh in piv.columns else np.zeros(len(top))
        ax.barh(y, vals, left=left, color=palette[coh], edgecolor="white", linewidth=0.3, label=coh)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([t.replace("_", " ") for t in top]); ax.invert_yaxis()
    ax.set_xlabel("summed β-glucuronidase flux across samples (relative units; ≈ carriage × uniform cap)")
    ax.legend(fontsize=7, loc="lower right", title="cohort", title_fontsize=7, frameon=True)
    plt.tight_layout(); plt.savefig(f"{fname}.{FMT}", dpi=DPI); plt.close()
    print(f"  {fname}.{FMT}  (top {len(top)} drivers, stacked by cohort)")


def panel_B(fname):
    uni = totals.reindex(top).values.astype(float)
    wt = np.array([totals[t] * eff(t) for t in top], dtype=float)
    drop = wt < 0.7 * uni
    y = np.arange(len(top)); h = 0.38
    fig, ax = plt.subplots(figsize=B_SIZE)
    ax.barh(y + h/2, uni, height=h, color="0.6")
    ax.barh(y - h/2, wt, height=h, color=["red" if d else "green" for d in drop])
    ax.set_yticks(y); ax.set_yticklabels([f"{t.replace('_', ' ')}  [{cls(t)}]" for t in top])
    ax.invert_yaxis()
    ax.set_xlabel("summed β-glucuronidase flux across samples (relative units)")
    leg = [Patch(facecolor="0.6", label="uniform"),
           Patch(facecolor="green", label="class-weighted")]
    if drop.any():
        leg.append(Patch(facecolor="red", label="class-weighted, demoted (<0.7× uniform)"))
    ax.legend(handles=leg, loc="lower right", fontsize=7.5, frameon=True)
    plt.tight_layout(); plt.savefig(f"{fname}.{FMT}", dpi=DPI); plt.close()
    print(f"  {fname}.{FMT}  demoted (<0.7x): {[top[i].replace('_',' ') for i in np.where(drop)[0]]}")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    print("Panel A (drivers, stacked by cohort):"); panel_A(f"{FIGDIR}/results_R4_A_drivers")
    print("Panel B (uniform vs class-weighted):"); panel_B(f"{FIGDIR}/results_R4_B_reweight")
    print("done -> 2 panels in", FIGDIR)
