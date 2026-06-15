"""Fig 6 (potential vs realized) as standalone SVG panels for external (Inkscape) composition.

  A  results_R6_A_collapse.svg   realized reactivation vs SN-38G supply, one line per community
                                 (cohort-coloured); lines fan out at saturating supply (each at its
                                 capacity) and collapse onto realized = min(supply, capacity) as supply
                                 falls -> the potential-vs-realized threshold.
  B  results_R6_B_metrics.svg    between-community CV% (left axis) and Spearman(realized, capacity)
                                 (right axis) vs supply -> variation collapse + decoupling.

Reads data/processed/flux/substrate_sweep_45.parquet (produced by 16b_substrate_sweep.py).
x-axis is supply on a log scale, INVERTED (saturating -> limiting, left -> right) to match the prose.
Same palette/order as Fig 1 so cohort colours are consistent across figures.

    python python/fig6_panels.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- STYLE ----------------
FMT        = "svg"            # "svg" | "pdf" | "png"
PALETTE    = "tab10"          # MUST match Fig 1
THEME_CTX  = "paper"
DPI        = 600
A_SIZE     = (6.5, 4.5)
B_SIZE     = (6.5, 4.5)
LINE_LW    = 1.1
LINE_ALPHA = 0.55
MARK_SIZE  = 4
CV_COLOR   = "0.15"           # CV% line/axis
RHO_COLOR  = "#C44E52"        # Spearman line/axis
REF_SHOW   = True             # faint "realized = supply" reference in panel A
# ---------------------------------------

FIGDIR = "data/processed/figures"
FLUX   = "data/processed/flux"
PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

sw = pd.read_parquet(f"{FLUX}/substrate_sweep_45.parquet")

# cohort palette/order identical to Fig 1 (primary medians of full capacity)
cap = pd.read_parquet(f"{FLUX}/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
meta = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
cap = cap.merge(meta, on="sample_id", how="left")
order = (cap[cap.cohort.isin(PRIMARY)].groupby("cohort")["sn38_capacity"].median()
         .sort_values().index.tolist())
cohort_palette = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))

sns.set_theme(style="whitegrid", context=THEME_CTX)
supplies = sorted(sw["supply"].unique())          # 1 .. 1000


def _invert_logx(ax):
    ax.set_xscale("log")
    ax.set_xlim(max(supplies) * 1.15, min(supplies) * 0.85)   # saturating (left) -> limiting (right)
    ax.set_xlabel("SN-38-glucuronide supply (relative units;  saturating → limiting)")


def panel_A(fname):
    fig, ax = plt.subplots(figsize=A_SIZE)
    if REF_SHOW:
        xs = np.array(supplies, float)
        ax.plot(xs, xs, ls="--", lw=1.0, color="0.6", zorder=1,
                label="realized = supply (delivery-limited)")
    for sid, g in sw.groupby("sample_id"):
        g = g.sort_values("supply")
        coh = g["cohort"].iloc[0]
        ax.plot(g["supply"], g["realized"], "-o", ms=MARK_SIZE, lw=LINE_LW, alpha=LINE_ALPHA,
                color=cohort_palette.get(coh, "0.5"), zorder=3)
    _invert_logx(ax)
    ax.set_ylim(-2, sw["realized"].max() * 1.08)   # cap to the capacity range (the ref line clips above)
    ax.set_ylabel("realized reactivation (relative units)")
    # cohort legend (one handle per cohort) + the reference line
    handles = [plt.Line2D([0], [0], color=cohort_palette[c], lw=2, label=c) for c in order]
    if REF_SHOW:
        handles = [plt.Line2D([0], [0], color="0.6", ls="--", lw=1, label="realized = supply")] + handles
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=1, frameon=True)
    plt.tight_layout(); plt.savefig(f"{fname}.{FMT}", dpi=DPI); plt.close()
    print(f"  {fname}.{FMT}  (45 communities)")


def panel_B(fname):
    rows = []
    for S, g in sw.groupby("supply"):
        cv = 100 * g["realized"].std() / g["realized"].mean() if g["realized"].mean() else 0.0
        rho = spearmanr(g["realized"], g["stage1_cap"]).correlation if g["realized"].std() > 0 else np.nan
        rows.append({"supply": S, "cv": cv, "rho": rho})
    m = pd.DataFrame(rows).sort_values("supply")

    fig, axL = plt.subplots(figsize=B_SIZE)
    axL.plot(m["supply"], m["cv"], "-o", ms=5, lw=1.6, color=CV_COLOR, label="CV%")
    axL.set_ylabel("between-community CV (%)", color=CV_COLOR)
    axL.tick_params(axis="y", labelcolor=CV_COLOR)
    axL.set_ylim(-3, max(m["cv"]) * 1.15)

    axR = axL.twinx()
    axR.grid(False)
    axR.plot(m["supply"], m["rho"], "-s", ms=5, lw=1.6, color=RHO_COLOR, label="Spearman ρ")
    axR.set_ylabel("Spearman ρ (realized vs capacity)", color=RHO_COLOR)
    axR.tick_params(axis="y", labelcolor=RHO_COLOR)
    axR.set_ylim(-0.05, 1.05)

    _invert_logx(axL)
    # combined legend
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left", frameon=True)
    plt.tight_layout(); plt.savefig(f"{fname}.{FMT}", dpi=DPI); plt.close()
    sat = m[m.supply == max(supplies)].iloc[0]; lim = m[m.supply == min(supplies)].iloc[0]
    print(f"  {fname}.{FMT}  CV {sat.cv:.1f}%->{lim.cv:.1f}%  rho {sat.rho:.2f}->{lim.rho:.2f}")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    print("Panel A:"); panel_A(f"{FIGDIR}/results_R6_A_collapse")
    print("Panel B:"); panel_B(f"{FIGDIR}/results_R6_B_metrics")
    print("done -> 2 panels in", FIGDIR)
