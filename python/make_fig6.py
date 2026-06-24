"""Fast Fig 6 (manuscript) = R7 phase diagram playground -- TWO SEPARATE panels for Inkscape.

Panel A (regime map): heatmap of % communities CAPACITY-limited (composition matters) over
  (substrate delivery D) x (community-capacity median), with the literature plausible box overlaid.
Panel B (local-peak sensitivity): % capacity-limited vs local-peak/whole-gut-average multiplier.

Uses the REAL 1,509-community relative-capacity DISTRIBUTION (shape fixed by data; only the absolute
scale is swept). realized_i = min(D, capacity_i); composition matters where capacity_i < D.
Mirrors 25_phase_diagram.py (the combined canonical). Writes * files. Edit STYLE, run, view.

    python python/fig7_playground.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- STYLE (edit me) ----------------
RENDER_A   = True
RENDER_B   = True
FMT        = "svg"          # "png" quick view | "svg" Inkscape deliverable | "pdf"
DPI        = 600
A_SIZE     = (4.5, 4)
B_SIZE     = (4.5, 4)
CMAP       = "RdYlBu_r"     # blue = host-gated, red = composition-controlled 
GRID_N     = 60             # heatmap resolution
D_RANGE    = (-6, -2)       # log10 substrate delivery D (mmol/gDW/h)
CAP_RANGE  = (-2, 1.0)      # log10 community-capacity median (mmol/gDW/h)
D_BOX      = (1e-5, 1e-4)   # plausible delivery (Slatter mass-balance: avg -> peak)
CAP_BOX    = (0.05, 2.85)   # Guthrie 2017: high=high-metab midpoint (~52%); low=low-metab near-bottom
                            # (~1%, CONSERVATIVE; low-metab midpoint ~4.4% would give ~0.24)
D_AVG      = float(np.sqrt(D_BOX[0] * D_BOX[1]))   # panel B baseline: geometric-mean of the delivery
                                                  # band (~3.16e-5), symmetric with geometric-mean capacity
PEAK_RANGE = (0, 5)         # panel B: log10 local-peak multiplier sweep
PEAK_MARKS = [(1, "whole-gut avg", "tab:green"), (10, "plausible local", "tab:orange"),
              (100, "high local", "tab:orange")]
SHOW_TITLE = False         # titles moved to the figure caption (publication style)
A_TITLE    = ("(a) Regime map: when does composition control reactivation?\n"
             "blue = host-gated (delivery-limited) | red = microbiome-controlled")
B_TITLE    = "(b) How large a LOCAL PEAK before composition matters?\n(central Guthrie capacity)"
OUT_A      = "data/processed/figures/results_R7_A_regime"
OUT_B      = "data/processed/figures/results_R7_B_localpeak"
# -------------------------------------------------

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]
cap = pd.read_parquet("data/processed/flux/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
coh = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
cap = cap.merge(coh, on="sample_id").query("cohort in @PRIMARY")
rel = cap.sn38_capacity.values
rel_med = np.median(rel[rel > 0])

def frac_capacity_limited(cap_median, D):
    real = rel * (cap_median / rel_med)        # scale distribution to this median, keep shape
    return float(np.mean(real < D)) * 100

if RENDER_A:
    D_grid = np.logspace(*D_RANGE, GRID_N)
    cap_grid = np.logspace(*CAP_RANGE, GRID_N)
    Z = np.array([[frac_capacity_limited(cm, d) for d in D_grid] for cm in cap_grid])
    fig, ax = plt.subplots(figsize=A_SIZE)
    im = ax.pcolormesh(D_grid, cap_grid, Z, cmap=CMAP, shading="auto", vmin=0, vmax=100)
    ax.set_xscale("log"); ax.set_yscale("log")        # actual values on the axes, not log10
    cb = fig.colorbar(im, ax=ax); cb.set_label("% communities CAPACITY-limited\n (composition matters)")
    ax.add_patch(plt.Rectangle((D_BOX[0], CAP_BOX[0]), D_BOX[1] - D_BOX[0], CAP_BOX[1] - CAP_BOX[0],
                 fill=False, edgecolor="white", lw=2.5, ls="--"))
    ax.text(np.sqrt(D_BOX[0] * D_BOX[1]), np.sqrt(CAP_BOX[0] * CAP_BOX[1]),
            "physiologically\nplausible", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    # measured-anchor values on the box edges (white): capacity left, delivery bottom
    akw = dict(color="white", fontsize=8, fontweight="bold")
    ax.text(D_BOX[0] * 0.7, CAP_BOX[1], f"{CAP_BOX[1]:g}", ha="right", va="center", **akw)
    ax.text(D_BOX[0] * 0.7, CAP_BOX[0], f"{CAP_BOX[0]:g}", ha="right", va="center", **akw)
    ax.text(D_BOX[0], CAP_BOX[0] * 0.65, r"$10^{-5}$", ha="center", va="top", **akw)
    ax.text(D_BOX[1], CAP_BOX[0] * 0.65, r"$10^{-4}$", ha="center", va="top", **akw)
    ax.set_xlabel("substrate delivery D  (mmol gDW$^{-1}$ h$^{-1}$)")
    ax.set_ylabel("community capacity median  (mmol gDW$^{-1}$ h$^{-1}$)")
    if SHOW_TITLE:
        ax.set_title(A_TITLE)
    plt.tight_layout(); plt.savefig(f"{OUT_A}.{FMT}", dpi=DPI); plt.close()
    print(f"Panel A -> {OUT_A}.{FMT}")

if RENDER_B:
    CAP_CENTRAL = float(np.sqrt(CAP_BOX[0] * CAP_BOX[1]))
    mult = np.logspace(*PEAK_RANGE, 200)
    fr = np.array([frac_capacity_limited(CAP_CENTRAL, D_AVG * m) for m in mult])
    fig, ax2 = plt.subplots(figsize=B_SIZE)
    ax2.plot(mult, fr, lw=2, color="tab:purple")
    for m_lab, lab, c in PEAK_MARKS:
        ax2.axvline(m_lab, color=c, ls=":", lw=1.2)
        ax2.text(m_lab, 102, lab, rotation=45, va="bottom", ha="left", fontsize=10, color=c)
    ax2.set_xscale("log"); ax2.set_xlabel("local peak delivery / whole-gut average  (multiplier)")
    ax2.set_ylabel("% communities CAPACITY-limited\n (composition matters)")
    ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()   # y-axis on the right
    if SHOW_TITLE:
        ax2.set_title(B_TITLE)
    ax2.set_ylim(-2, 105)
    plt.tight_layout(); plt.savefig(f"{OUT_B}.{FMT}", dpi=DPI); plt.close()
    # threshold multipliers
    for X in [5, 10, 25, 50]:
        idx = np.argmax(fr >= X)
        m = mult[idx] if fr.max() >= X else np.inf
        print(f"  {X:2d}% capacity-limited needs local peak ~{m:,.0f}x whole-gut average")
    print(f"Panel B -> {OUT_B}.{FMT}  (central Guthrie cap={CAP_CENTRAL:.2f}, D_avg={D_AVG:g})")
