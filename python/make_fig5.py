"""Fast Fig 6 (potential vs realized, R6) playground -- TWO SEPARATE panels, viridis 17b style.

Renders Panel A (per-community collapse, coloured by saturating capacity) and Panel B (CV% +
correlation vs supply) to SEPARATE * files, for external (Inkscape) composition. Same design as
17b_q1_plot.py (the combined version), just split. Edit the STYLE block, run, view the PNGs.

Note on Panel B: at the limiting (low-supply) end, realized reactivation is ~constant across samples,
so its correlation with capacity is ~0 (and noisy: it dips to about -0.07 at supply=4). That is the
"touches zero then rises" you saw -- it is decoupling noise, not a real anticorrelation. CORR_YLIM
controls whether the small negative is shown honestly (default) or floored at 0.

    python python/fig6_playground.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

# ---------------- STYLE (edit me) ----------------
RENDER_A    = True
RENDER_B    = True
FMT         = "svg"          # "png" for quick view | "svg" for the Inkscape deliverable | "pdf"
DPI         = 600
A_SIZE      = (4.5, 4)
B_SIZE      = (4.5, 4)
CMAP        = "viridis"
A_YLIM      = (0, 95)        # cap to the capacity range so per-sample plateaus are visible
REF_LINE    = True           # dashed "realized = supply" envelope in Panel A
SHOW_TITLE  = False           # the descriptive two-line titles (set False for Inkscape if caption covers it)
A_TITLE     = "Each community plateaus at its potential,\nthen tracks the delivered amount"
B_TITLE     = "As substrate drops: variation collapses,\ncapacity decouples from composition"
CORR_METHOD = "pearson"      # "pearson" (matches 17b np.corrcoef) | "spearman"
CORR_YLIM   = (-0.1, 1.05)   # show the small negative dip honestly; set (0, 1.05) to floor at 0
CV_COLOR    = "tab:red"
CORR_COLOR  = "tab:blue"
OUT_A       = "data/processed/figures/results_R6_A_collapse"
OUT_B       = "data/processed/figures/results_R6_B_metrics"
# -------------------------------------------------

df = pd.read_parquet("data/processed/flux/substrate_sweep_45.parquet")
supplies = sorted(df.supply.unique())

if RENDER_A:
    fig, ax = plt.subplots(figsize=A_SIZE)
    norm = Normalize(df.stage1_cap.min(), df.stage1_cap.max()); cmap = plt.get_cmap(CMAP)
    for sid, g in df.groupby("sample_id"):
        g = g.sort_values("supply")
        ax.plot(g.supply, g.realized, "-", lw=0.8, alpha=0.7, color=cmap(norm(g.stage1_cap.iloc[0])))
    if REF_LINE:
        ax.plot(supplies, supplies, "k--", lw=1.2, label="realized = supply (fully delivery-limited)")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("SN-38G supply (relative)")
    ax.set_ylim(*A_YLIM); ax.set_ylabel("realized reactivation capacity")
    if SHOW_TITLE:
        ax.set_title(A_TITLE)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="reactivation potential (saturating)")
    plt.tight_layout(); plt.savefig(f"{OUT_A}.{FMT}", dpi=DPI); plt.close()
    print(f"Panel A -> {OUT_A}.{FMT}")

if RENDER_B:
    rows = []
    for S, g in df.groupby("supply"):
        cv = 100 * g.realized.std() / g.realized.mean() if g.realized.mean() else 0
        if g.realized.std() > 0:
            corr = (np.corrcoef(g.realized, g.stage1_cap)[0, 1] if CORR_METHOD == "pearson"
                    else spearmanr(g.realized, g.stage1_cap).correlation)
        else:
            corr = np.nan
        rows.append((S, cv, corr))
    summ = pd.DataFrame(rows, columns=["supply", "cv", "corr"]).sort_values("supply")

    fig, axB = plt.subplots(figsize=B_SIZE)
    axB.plot(summ.supply, summ.cv, "o-", color=CV_COLOR, label="CV% across individuals")
    axB.set_xscale("log"); axB.set_xlabel("SN-38G supply (relative)")
    axB.set_ylabel("CV% of realized capacity", color=CV_COLOR); axB.tick_params(axis="y", labelcolor=CV_COLOR)
    ax2 = axB.twinx(); ax2.grid(False)
    ax2.axhline(0, color=CORR_COLOR, lw=0.6, ls=":", alpha=0.5)   # zero reference for the dip
    ax2.plot(summ.supply, summ["corr"], "s-", color=CORR_COLOR,
             label=f"corr(realized, potential) [{CORR_METHOD}]")
    ax2.set_ylabel("corr(realized, carrier-abundance potential)", color=CORR_COLOR)
    ax2.tick_params(axis="y", labelcolor=CORR_COLOR); ax2.set_ylim(*CORR_YLIM)
    if SHOW_TITLE:
        axB.set_title(B_TITLE)
    plt.tight_layout(); plt.savefig(f"{OUT_B}.{FMT}", dpi=DPI); plt.close()
    lim = summ.iloc[0]; sat = summ.iloc[-1]
    print(f"Panel B -> {OUT_B}.{FMT}  | corr limiting={lim['corr']:.3f} saturating={sat['corr']:.3f} "
          f"| CV {sat['cv']:.0f}%->{lim['cv']:.0f}%")
