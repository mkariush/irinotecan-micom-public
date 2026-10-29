"""Fig S3 -- growth-reactivation trade-off front (mSystems revision, R2.3).

Reviewer 2 held that Arm A and Arm B cannot give the same EX_sn38_m. The answer is that the whole
front is flat, not just the 95% point: g(f) = max EX_sn38_m s.t. growth >= f * gr_max is
non-increasing in f, so equality at the extremes proves flatness throughout (Text S1.6).

  A  results_S3_A_front    per-community capacity vs growth floor, normalised to each community's
                           f=0 value so 45 flat lines are legible on one axis; inset shows the raw
                           spread of capacities being held constant
  B  results_S3_B_uptake   paired total nutrient uptake at f=0 vs f=0.999, showing the growth
                           constraint IS active (+6.7% median) while SN-38 output is unchanged

Panel B is the point reviewers actually asked for: if the front were flat because the constraint
never bound, that would be a degenerate model. It binds; it just does not compete with reactivation.

    python python/make_figS3_front.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- STYLE (edit me) ----------------
RENDER_A   = True
RENDER_B   = True
FMT        = "svg"           # "png" for quick view | "svg" for the Inkscape deliverable
DPI        = 600
A_SIZE     = (4.5, 4)
B_SIZE     = (4.5, 4)
SHOW_TITLE = False
LINE_COLOR = "0.45"
LINE_ALPHA = 0.55
HL_COLOR   = "tab:red"       # the one community that needs uptake beyond SN-38G at f=0
EXCEPTION  = "CCIS00281083ST-3-0"
OUT_A      = "data/processed/figures/results_S3_A_front"
OUT_B      = "data/processed/figures/results_S3_B_uptake"
SRC        = "data/processed/flux/growth_pareto.parquet"
# -------------------------------------------------

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_parquet(SRC)
wide = df.pivot(index="sample_id", columns="fraction", values="sn38")
fracs = sorted(wide.columns)
fmax = max(fracs)
base = wide[0.0]

# INVARIANT: the front must be flat. Assert rather than trust -- an earlier version of this
# analysis silently minimised instead of maximising and produced a NON-monotonic front.
dev = wide.sub(base, axis=0).abs().max().max()
assert dev < 1e-6, f"front is not flat (max deviation {dev:.2e}) -- check objective direction"

if RENDER_A:
    # ABSOLUTE capacity, not normalised: normalising collapses all 45 onto one line and the panel
    # says nothing (the same reason A==B itself gets no main figure). Plotted absolutely, each
    # community is a flat line at its own height -- the spread IS the information.
    fig, ax = plt.subplots(figsize=A_SIZE)
    for sid, row in wide.iterrows():
        ax.plot(fracs, row.values, color=LINE_COLOR, alpha=LINE_ALPHA, lw=0.9,
                solid_capstyle="round")
    ax.set_xlabel("growth floor (fraction of maximum community growth rate)")
    ax.set_ylabel("SN-38 reactivation capacity (relative units)")
    ax.set_ylim(0, 95)
    ax.set_xlim(-0.02, 1.02)
    ax.text(0.03, 92, f"{len(wide)} communities, each flat across the whole front\n"
            f"(max within-community deviation "
            f"{wide.sub(base, axis=0).abs().max().max():.0e}; capacity range "
            f"{base.min():.1f}–{base.max():.1f})", fontsize=7, va="top")
    if SHOW_TITLE:
        ax.set_title("Forcing growth does not reduce reactivation", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_A}.{FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_A}.{FMT}")

if RENDER_B:
    ends = df[df.fraction.isin([0.0, fmax])]
    u = ends.pivot(index="sample_id", columns="fraction", values="total_uptake")
    s = ends.pivot(index="sample_id", columns="fraction", values="sn38g_uptake")
    fig, ax = plt.subplots(figsize=B_SIZE)
    for sid in u.index:
        c = HL_COLOR if sid == EXCEPTION else LINE_COLOR
        z = 3 if sid == EXCEPTION else 1
        ax.plot([0, 1], [u.loc[sid, 0.0], u.loc[sid, fmax]], color=c,
                alpha=0.9 if sid == EXCEPTION else LINE_ALPHA, lw=1.0, marker="o",
                ms=2.6, zorder=z)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["f = 0\n(no growth required)", f"f = {fmax}\n(near-maximal growth)"])
    ax.set_ylabel("total nutrient uptake (relative units)")
    ax.set_xlim(-0.25, 1.25)
    med = (u[fmax] / u[0.0]).median()
    n_exact = int((u[0.0] - s[0.0]).abs().le(1e-6).sum())
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.12)                       # headroom so the note clears the top traces
    ax.text(-0.22, hi * 1.10,
            f"median +{100*(med-1):.1f}% total uptake\nSN-38 output unchanged\n"
            f"uptake = SN-38G alone in {n_exact}/{len(u)} at f = 0",
            fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    if EXCEPTION in u.index:                         # name the red trace rather than leave it unexplained
        ax.annotate(f"{EXCEPTION}\n(the 1 of 45 needing\nuptake beyond SN-38G)",
                    xy=(0, u.loc[EXCEPTION, 0.0]), xytext=(0.30, hi * 0.80),
                    fontsize=6.5, color=HL_COLOR, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color=HL_COLOR, lw=0.6,
                                    shrinkA=0, shrinkB=3))
    if SHOW_TITLE:
        ax.set_title("The growth constraint binds, but not on reactivation", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_B}.{FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_B}.{FMT}")
    print(f"   median uptake ratio {med:.4f} | exact SN-38G-only at f=0: {n_exact}/{len(u)}")
