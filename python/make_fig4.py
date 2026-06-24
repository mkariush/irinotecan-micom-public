"""Fig 4 -- CRC vs control (Results R5): four standalone SVG panels for Inkscape composition.

  A  results_R5_A_violin              CRC vs control capacity by cohort (cohort colour + CRC hatch + points)
  B  results_R5_B_forest_primary      forest: Cliff's delta, 9 primary cohorts, uniform capacity
  C  results_R5_C_forest_reweighted   forest: same, efficiency-class-reweighted capacity
  D  results_R5_D_forest_sensitivity  forest: uniform + GuptaA_2019 & HanniganGD_2017 (grey)

Bootstrap stats are cached to data/processed/flux/r5_forest_*.parquet, so STYLE tweaks are instant;
set RECOMPUTE=True once if the underlying capacities change (slow: 2000x bootstrap x forests). Edit the
STYLE block to adjust. Emits the four results_R5_*.svg panels.

    python python/make_fig4.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ---------------- STYLE (edit me) ----------------
RENDER_A     = True            # CRC vs control violins
RENDER_B     = True            # forest, primary, uniform capacity
RENDER_C     = True            # forest, primary, efficiency-reweighted
RENDER_D     = True            # forest, + depth-sensitivity cohorts
FMT          = "svg"           # "png" quick view | "svg" Inkscape deliverable | "pdf"
DPI          = 600             # only matters for png
PALETTE      = "tab10"         # MUST match Fig 1
THEME_CTX    = "paper"
SHOW_TITLE   = False            # forest panel titles; set False for caption-only (publication) style
# panel A (violin)
A_SIZE       = (9, 4.5)
VIOLIN_ALPHA = 0.2
VIOLIN_EDGE  = "0.25"
VIOLIN_LW    = 0.8
HATCH        = "////"
POINT_SIZE   = 4
POINT_ALPHA  = 0.65
POINT_EDGE   = "0.3"
POINT_LW     = 0.2
POINT_JITTER = 0.08
# panels B/C/D (forest)
FOREST_SIZE  = (4.3, 4)
XLIM         = (-0.85, 0.85)
SENS_GREY    = {"GuptaA_2019": "0.55", "HanniganGD_2017": "0.72"}   # significance shown by asterisk on label only
POOL_COLOR   = "navy"
# bootstrap (leave RECOMPUTE False to reuse the cache; flip True only if capacities changed)
BOOT         = 2000
RECOMPUTE    = False
SEED         = 0
# -------------------------------------------------

FIGDIR = "data/processed/figures"
FLUX   = "data/processed/flux"
PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]
SENSITIVITY = ["GuptaA_2019", "HanniganGD_2017"]
rng = np.random.default_rng(SEED)

# ---- efficiency weighting (shared single source of truth) ----
from gus_efficiency import CLASS_EFF, SPECIES_CLASS, GENUS_DEFAULT, eff

# ---- load + build capacities ----
cap = pd.read_parquet(f"{FLUX}/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")
cap = cap.merge(meta, on="sample_id", how="left")

# refined (efficiency-weighted) per-sample capacity, primary cohorts (for panel C)
tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
con = pd.read_parquet(f"{FLUX}/full_taxa_contributions.parquet")
carr = (con[["sample_id", "taxon"]].drop_duplicates()
        .merge(tax[["sample_id", "taxon", "abundance"]], on=["sample_id", "taxon"], how="left")
        .dropna(subset=["abundance"]))
carr["w"] = carr["abundance"] * carr["taxon"].map(eff)
ref = carr.groupby("sample_id")["w"].sum().mul(100).rename("cap_ref").reset_index()
cap = cap.merge(ref, on="sample_id", how="left")

sns.set_theme(style="whitegrid", context=THEME_CTX)
order = (cap[cap.cohort.isin(PRIMARY)].groupby("cohort")["sn38_capacity"].median()
         .sort_values().index.tolist())
cohort_palette = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))

def panel_color(c):
    return cohort_palette.get(c, SENS_GREY.get(c, "0.5"))

# ---- forest stats (Cliff's delta + bootstrap), cached ----
def cliffs_delta(crc, ctl):           # positive => control > CRC
    return np.sign(ctl[:, None] - crc[None, :]).sum() / (len(crc) * len(ctl))

def boot_ci(crc, ctl, n):
    ds = np.empty(n)
    for i in range(n):
        ds[i] = cliffs_delta(rng.choice(crc, len(crc), True), rng.choice(ctl, len(ctl), True))
    return np.percentile(ds, [2.5, 97.5])

def forest_stats(value_col, cohorts, key):
    path = f"{FLUX}/r5_forest_{key}.parquet"
    if os.path.exists(path) and not RECOMPUTE:
        return pd.read_parquet(path)
    d = cap[cap.cohort.isin(cohorts) & cap.study_condition.isin(["CRC", "control"])]
    rows, groups = [], {}
    for coh, g in d.groupby("cohort"):
        crc = g.loc[g.study_condition == "CRC", value_col].dropna().values
        ctl = g.loc[g.study_condition == "control", value_col].dropna().values
        if len(crc) < 3 or len(ctl) < 3:
            continue
        delta = cliffs_delta(crc, ctl); lo, hi = boot_ci(crc, ctl, BOOT)
        _, p = mannwhitneyu(crc, ctl, alternative="two-sided")
        rows.append(dict(cohort=coh, delta=delta, lo=lo, hi=hi, p=p,
                         n=len(crc) + len(ctl), n_crc=len(crc), n_ctl=len(ctl)))
        groups[coh] = (crc, ctl)
    df = pd.DataFrame(rows)
    pb = np.empty(BOOT)
    for i in range(BOOT):
        ds, ns = [], []
        for crc, ctl in groups.values():
            ds.append(cliffs_delta(rng.choice(crc, len(crc), True), rng.choice(ctl, len(ctl), True)))
            ns.append(len(crc) + len(ctl))
        pb[i] = np.average(ds, weights=ns)
    pooled = np.average(df.delta, weights=df.n); plo, phi = np.percentile(pb, [2.5, 97.5])
    df = pd.concat([df, pd.DataFrame([dict(cohort="__pooled__", delta=pooled, lo=plo, hi=phi,
                    p=np.nan, n=int(df.n.sum()), n_crc=-1, n_ctl=-1)])], ignore_index=True)
    df.to_parquet(path)
    return df

def draw_forest(df, cohort_order, fname, title):
    rows = df[df.cohort != "__pooled__"].set_index("cohort")
    present = [c for c in cohort_order if c in rows.index]
    ypos = np.arange(len(present))[::-1]
    fig, ax = plt.subplots(figsize=FOREST_SIZE)
    for y, c in zip(ypos, present):
        r = rows.loc[c]
        ax.errorbar(r.delta, y, xerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="none",
                    ecolor="0.6", capsize=2, lw=1.0, zorder=2)
        ax.scatter(r.delta, y, s=55, color=panel_color(c), edgecolor="0.25", linewidth=0.5, zorder=5)
    yb = -1.4
    pr = df[df.cohort == "__pooled__"].iloc[0]
    ax.add_patch(plt.Polygon([[pr.lo, yb], [pr.delta, yb + 0.3], [pr.hi, yb], [pr.delta, yb - 0.3]],
                             color=POOL_COLOR, zorder=5))
    ax.axvline(0, color="k", lw=1, ls="--", zorder=1)
    labels = [c.replace("_", " ") + ("  *" if rows.loc[c].p < 0.05 else "") for c in present]
    ax.set_yticks(list(ypos) + [yb]); ax.set_yticklabels(labels + ["Pooled (n-weighted)"])
    ax.set_xlim(*XLIM); ax.set_ylim(yb - 0.9, len(present) - 0.3)
    ax.set_xlabel("Cliff's δ   (← CRC higher    |    control higher →)")
    if SHOW_TITLE:
        ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{fname}.{FMT}", dpi=DPI); plt.close()
    print(f"  {fname}.{FMT}  pooled δ={pr.delta:.3f} [{pr.lo:.3f},{pr.hi:.3f}]")

def draw_violin(fname):
    sub = cap[cap.cohort.isin(PRIMARY) & cap.study_condition.isin(["CRC", "control"])].copy()
    fig, ax = plt.subplots(figsize=A_SIZE)
    sns.violinplot(data=sub, x="cohort", y="sn38_capacity", hue="study_condition",
                   hue_order=["control", "CRC"], order=order, split=False, cut=0,
                   inner="quartile", linewidth=VIOLIN_LW, ax=ax, saturation = 1,
                   palette={"control": "0.8", "CRC": "0.8"})
    for coll in ax.collections:
        paths = coll.get_paths()
        if not paths:
            continue
        xc = paths[0].vertices[:, 0].mean(); ci = int(round(xc))
        if 0 <= ci < len(order):
            coll.set_facecolor(cohort_palette[order[ci]]); coll.set_edgecolor(VIOLIN_EDGE)
            coll.set_alpha(VIOLIN_ALPHA)
            if xc > ci:
                coll.set_hatch(HATCH)
    n_violin = len(ax.collections)
    sns.stripplot(data=sub, x="cohort", y="sn38_capacity", hue="study_condition",
                  hue_order=["control", "CRC"], order=order, dodge=True, jitter=POINT_JITTER,
                  size=POINT_SIZE, linewidth=POINT_LW, edgecolor=POINT_EDGE, ax=ax,
                  legend=False, palette={"control": "0.3", "CRC": "0.3"})
    for coll in ax.collections[n_violin:]:
        offs = coll.get_offsets()
        if len(offs) == 0:
            continue
        coll.set_facecolor([cohort_palette[order[int(round(x))]]
                            if 0 <= int(round(x)) < len(order) else (0.3, 0.3, 0.3) for x, _ in offs])
        coll.set_alpha(POINT_ALPHA)
    ax.set_ylabel("SN-38 reactivation capacity (relative units)"); ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    ax.legend(handles=[Patch(facecolor="0.8", edgecolor=VIOLIN_EDGE, label="control"),
                       Patch(facecolor="0.8", edgecolor=VIOLIN_EDGE, hatch=HATCH, label="CRC")],
              title="", loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout(); plt.savefig(f"{fname}.{FMT}", dpi=DPI); plt.close()
    print(f"  {fname}.{FMT}  (violin, n={len(sub)})")

if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    if RENDER_A:
        print("Panel A (CRC vs control violins):"); draw_violin(f"{FIGDIR}/results_R5_A_violin")
    if RENDER_B:
        print("Panel B (primary, uniform):")
        draw_forest(forest_stats("sn38_capacity", PRIMARY, "primary_uniform"),
                    order, f"{FIGDIR}/results_R5_B_forest_primary", "Uniform capacity (9 primary cohorts)")
    if RENDER_C:
        print("Panel C (primary, efficiency-reweighted):")
        draw_forest(forest_stats("cap_ref", PRIMARY, "primary_reweighted"),
                    order, f"{FIGDIR}/results_R5_C_forest_reweighted", "Efficiency-class-reweighted capacity")
    if RENDER_D:
        print("Panel D (uniform + depth-sensitivity cohorts):")
        draw_forest(forest_stats("sn38_capacity", PRIMARY + SENSITIVITY, "with_sensitivity"),
                    order + SENSITIVITY, f"{FIGDIR}/results_R5_D_forest_sensitivity",
                    "+ depth-sensitivity cohorts (grey)")
    print("done -> panels in", FIGDIR)
