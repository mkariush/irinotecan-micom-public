"""Fig 6 (manuscript) = R7 phase diagram as two standalone SVG panels for external (Inkscape) composition.

  A  results_R7_A_regime.svg     regime heatmap: % communities CAPACITY-limited (composition matters)
                                 over (substrate delivery D) x (community-capacity median); literature
                                 plausible box overlaid. Blue = host-gated (delivery-limited),
                                 red = composition-controlled.
  B  results_R7_B_localpeak.svg  local-peak sensitivity: % capacity-limited vs local-peak/whole-gut
                                 multiplier, at the central (geometric-mean) Guthrie capacity.

Uses the REAL 1,509-community relative-capacity DISTRIBUTION (shape fixed; only the absolute scale swept).
realized_i = min(D, capacity_i); composition matters where capacity_i < D. Canonical separate-panel
version of the combined 25_phase_diagram.py; no in-figure titles (interpretive content is in the caption).
Style mirrors fig7_playground.py; tweak there, then port. Emits SVG (Inkscape) + PNG.

    python python/fig7_panels.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- STYLE ----------------
CMAP       = "RdYlBu_r"     # blue = host-gated, red = composition-controlled
A_SIZE     = (7, 6)
B_SIZE     = (7, 6)
GRID_N     = 60
D_RANGE    = (-6, -2)       # log10 substrate delivery D (mmol/gDW/h)
CAP_RANGE  = (-2, 1.0)      # log10 community-capacity median (mmol/gDW/h)
D_BOX      = (1e-5, 1e-4)   # plausible delivery (Slatter mass-balance: avg -> peak)
CAP_BOX    = (0.05, 2.85)   # Guthrie 2017: high=high-metab midpoint (~52%); low=low-metab near-bottom
                            # (~1%, CONSERVATIVE; low-metab midpoint ~4.4% would give ~0.24)
D_AVG      = float(np.sqrt(D_BOX[0] * D_BOX[1]))   # panel B baseline: geometric-mean of the delivery
                                                  # band (~3.16e-5), symmetric with the geometric-mean
                                                  # capacity used in panel B (not the band floor)
PEAK_RANGE = (0, 5)         # panel B: log10 local-peak multiplier sweep
PEAK_MARKS = [(1, "whole-gut avg", "tab:green"), (10, "plausible local", "tab:orange"),
              (100, "high local", "tab:orange")]
DPI        = 600
# ---------------------------------------

FIGDIR = "data/processed/figures"
PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]
cap = pd.read_parquet("data/processed/flux/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
coh = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
cap = cap.merge(coh, on="sample_id").query("cohort in @PRIMARY")
rel = cap.sn38_capacity.values
rel_med = np.median(rel[rel > 0])


def frac_capacity_limited(cap_median, D):
    real = rel * (cap_median / rel_med)
    return float(np.mean(real < D)) * 100


def panel_A(stem):
    D_grid = np.logspace(*D_RANGE, GRID_N)
    cap_grid = np.logspace(*CAP_RANGE, GRID_N)
    Z = np.array([[frac_capacity_limited(cm, d) for d in D_grid] for cm in cap_grid])
    fig, ax = plt.subplots(figsize=A_SIZE)
    im = ax.pcolormesh(D_grid, cap_grid, Z, cmap=CMAP, shading="auto", vmin=0, vmax=100)
    ax.set_xscale("log"); ax.set_yscale("log")        # actual values on the axes, not log10
    cb = fig.colorbar(im, ax=ax); cb.set_label("% communities CAPACITY-limited (composition matters)")
    ax.add_patch(plt.Rectangle((D_BOX[0], CAP_BOX[0]), D_BOX[1] - D_BOX[0], CAP_BOX[1] - CAP_BOX[0],
                 fill=False, edgecolor="white", lw=2.5, ls="--"))
    ax.text(np.sqrt(D_BOX[0] * D_BOX[1]), np.sqrt(CAP_BOX[0] * CAP_BOX[1]),
            "physiologically\nplausible", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    # measured-anchor values labelled on the box edges (white): capacity (Guthrie) on the left edge,
    # delivery (Slatter) on the bottom edge
    akw = dict(color="white", fontsize=8, fontweight="bold")
    ax.text(D_BOX[0] * 0.7, CAP_BOX[1], f"{CAP_BOX[1]:g}", ha="right", va="center", **akw)
    ax.text(D_BOX[0] * 0.7, CAP_BOX[0], f"{CAP_BOX[0]:g}", ha="right", va="center", **akw)
    ax.text(D_BOX[0], CAP_BOX[0] * 0.65, r"$10^{-5}$", ha="center", va="top", **akw)
    ax.text(D_BOX[1], CAP_BOX[0] * 0.65, r"$10^{-4}$", ha="center", va="top", **akw)
    ax.set_xlabel("substrate delivery D  (mmol gDW$^{-1}$ h$^{-1}$)")
    ax.set_ylabel("community capacity median  (mmol gDW$^{-1}$ h$^{-1}$)")
    plt.tight_layout()
    for fmt in ("svg", "png"):
        plt.savefig(f"{stem}.{fmt}", dpi=DPI)
    plt.close()
    box = frac_capacity_limited(np.sqrt(CAP_BOX[0] * CAP_BOX[1]), np.sqrt(D_BOX[0] * D_BOX[1]))
    print(f"  {stem}.svg/.png  (plausible-box centre: {box:.1f}% capacity-limited)")


def panel_B(stem):
    CAP_CENTRAL = float(np.sqrt(CAP_BOX[0] * CAP_BOX[1]))
    mult = np.logspace(*PEAK_RANGE, 200)
    fr = np.array([frac_capacity_limited(CAP_CENTRAL, D_AVG * m) for m in mult])
    fig, ax2 = plt.subplots(figsize=B_SIZE)
    ax2.plot(mult, fr, lw=2, color="tab:purple")
    for m_lab, lab, c in PEAK_MARKS:
        ax2.axvline(m_lab, color=c, ls=":", lw=1.2)
        ax2.text(m_lab, 102, lab, rotation=90, va="bottom", ha="right", fontsize=8, color=c)
    ax2.set_xscale("log"); ax2.set_xlabel("local peak delivery / whole-gut average  (multiplier)")
    ax2.set_ylabel("% communities capacity-limited (composition matters)")
    ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()   # y-axis on the right
    ax2.set_ylim(-2, 105)
    plt.tight_layout()
    for fmt in ("svg", "png"):
        plt.savefig(f"{stem}.{fmt}", dpi=DPI)
    plt.close()
    thr = {X: (mult[np.argmax(fr >= X)] if fr.max() >= X else np.inf) for X in (5, 50)}
    print(f"  {stem}.svg/.png  (5%->{thr[5]:,.0f}x, 50%->{thr[50]:,.0f}x whole-gut average)")


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    print("Panel A (regime heatmap):");      panel_A(f"{FIGDIR}/results_R7_A_regime")
    print("Panel B (local-peak sensitivity):"); panel_B(f"{FIGDIR}/results_R7_B_localpeak")
    print("done -> 2 panels in", FIGDIR)
