"""Fig 6 (manuscript Fig 5 = R6, potential vs realized) as two standalone SVG panels, viridis 17b style.

  A  results_R6_A_collapse.svg   per-community realized reactivation vs SN-38G supply, coloured by
                                 saturating capacity (viridis + colorbar); plateaus at potential, then
                                 tracks the delivered amount (realized = min(supply, capacity)).
  B  results_R6_B_metrics.svg    between-community CV% (left) + corr(realized, capacity) (right) vs
                                 supply; physiological (delivery-limited) band shaded.

Reads substrate_sweep_45.parquet (45-sample subset, 16b). Canonical viridis separate-panel version
(supersedes the earlier cohort-coloured fig6_panels and the combined 17b_q1_plot.py). Style mirrors
fig6_playground.py; tweak there, then port. Emits SVG (Inkscape) + PNG.

    python python/fig6_panels.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ---------------- STYLE ----------------
CMAP        = "viridis"
A_SIZE      = (7, 6)
B_SIZE      = (7, 6)
A_YLIM      = (0, 95)
SHOW_TITLE  = True
A_TITLE     = "Each community plateaus at its potential,\nthen tracks the delivered amount"
B_TITLE     = "As substrate drops: variation collapses,\ncapacity decouples from composition"
CORR_METHOD = "pearson"        # "pearson" (17b) | "spearman"
CORR_YLIM   = (-0.1, 1.05)     # show the small negative corr dip honestly; (0, 1.05) to floor at 0
CV_COLOR    = "tab:red"
CORR_COLOR  = "tab:blue"
PHYS_MAX    = 8
DPI         = 600
# ---------------------------------------

FIGDIR = "data/processed/figures"
df = pd.read_parquet("data/processed/flux/substrate_sweep_45.parquet")
supplies = sorted(df.supply.unique())


def panel_A(stem):
    fig, ax = plt.subplots(figsize=A_SIZE)
    norm = Normalize(df.stage1_cap.min(), df.stage1_cap.max()); cmap = plt.get_cmap(CMAP)
    for sid, g in df.groupby("sample_id"):
        g = g.sort_values("supply")
        ax.plot(g.supply, g.realized, "-", lw=0.8, alpha=0.7, color=cmap(norm(g.stage1_cap.iloc[0])))
    ax.plot(supplies, supplies, "k--", lw=1.2, label="realized = supply (fully delivery-limited)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("SN-38G supply (relative)")
    ax.set_ylim(*A_YLIM); ax.set_ylabel("realized reactivation capacity")
    if SHOW_TITLE:
        ax.set_title(A_TITLE)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="reactivation potential (saturating)")
    plt.tight_layout()
    for fmt in ("svg", "png"):
        plt.savefig(f"{stem}.{fmt}", dpi=DPI)
    plt.close()
    print(f"  {stem}.svg/.png  ({df.sample_id.nunique()} communities)")


def panel_B(stem):
    from scipy.stats import spearmanr
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
    ax2.axhline(0, color=CORR_COLOR, lw=0.6, ls=":", alpha=0.5)
    ax2.plot(summ.supply, summ["corr"], "s-", color=CORR_COLOR, label="corr(realized, potential)")
    ax2.set_ylabel("corr(realized, carrier-abundance potential)", color=CORR_COLOR)
    ax2.tick_params(axis="y", labelcolor=CORR_COLOR); ax2.set_ylim(*CORR_YLIM)
    if SHOW_TITLE:
        axB.set_title(B_TITLE)
    axB.axvspan(supplies[0], PHYS_MAX, color="0.9", zorder=0)
    axB.text(2.2, axB.get_ylim()[1] * 0.93, "physiological\n(delivery-limited)", fontsize=9, ha="center")
    plt.tight_layout()
    for fmt in ("svg", "png"):
        plt.savefig(f"{stem}.{fmt}", dpi=DPI)
    plt.close()
    print(f"  {stem}.svg/.png  CV {summ.iloc[-1]['cv']:.0f}%->{summ.iloc[0]['cv']:.0f}%  "
          f"corr {summ.iloc[-1]['corr']:.2f}->{summ.iloc[0]['corr']:.2f}")


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    print("Panel A (per-community collapse):"); panel_A(f"{FIGDIR}/results_R6_A_collapse")
    print("Panel B (CV% + correlation):");      panel_B(f"{FIGDIR}/results_R6_B_metrics")
    print("done -> 2 panels in", FIGDIR)
