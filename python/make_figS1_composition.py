"""Fig S1 -- cross-cohort community composition summary (mSystems revision, R2.4).

Reviewer 2: "Even though the authors use public data it would be good to still show the overall
compositional differences between cohorts as a summary."

  A  results_S1_A_phylum     mean phylum-level composition per cohort (stacked bars)
  B  results_S1_B_pcoa       PCoA on species-level Bray-Curtis, coloured by cohort
  C  results_S1_C_alpha      observed species richness per cohort

Phylum assignments come from the RAW curatedMetagenomicData lineage strings
(k__|p__|...|s__), which taxonomy_micom.parquet does not retain -- it keeps species only.

The two depth-sensitivity cohorts (GuptaA_2019, HanniganGD_2017) are shown alongside the nine
primary cohorts but hatched/outlined, because their low sequencing depth is visible as reduced
richness in panel C and that is the basis on which they were excluded from the primary analysis.

    python python/make_figS1_composition.py
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform

# ---------------- STYLE (edit me) ----------------
RENDER_A = True
RENDER_B = True
RENDER_C = True
FMT      = "svg"              # "png" for quick view | "svg" for the Inkscape deliverable
DPI      = 600
A_SIZE   = (7.2, 2.6)
B_SIZE   = (3.4, 3.0)
C_SIZE   = (7.2, 2.4)
N_PHYLA  = 6                  # top-N phyla by overall mean, remainder pooled as "Other"
MIN_PREV = 0.01               # species kept for the ordination: present in >=1% of samples
SEED     = 0
OUT      = "data/processed/figures"
# -------------------------------------------------

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]
SENSITIVITY = ["GuptaA_2019", "HanniganGD_2017"]
ALL_COHORTS = PRIMARY + SENSITIVITY

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs(OUT, exist_ok=True)


def load_long():
    """Raw (pre-cutoff) profiles in long form, with phylum and species parsed from the lineage."""
    frames = []
    for c in ALL_COHORTS:
        d = pd.read_parquet(f"data/raw/{c}_abundance.parquet")
        idv = [x for x in ("sample_id", "cohort") if x in d.columns]
        m = d.melt(id_vars=idv, var_name="lineage", value_name="ab")
        m = m[m.ab > 0].copy()
        m["cohort"] = c
        frames.append(m)
    long = pd.concat(frames, ignore_index=True)
    long["phylum"] = long.lineage.str.extract(r"\|p__([^|]+)")
    long["species"] = long.lineage.str.extract(r"\|s__([^|]+)$")
    long = long.dropna(subset=["species"])
    # raw values are PERCENT (sum ~100 per sample); convert to fractions
    tot = long.groupby("sample_id")["ab"].transform("sum")
    long["rel"] = long.ab / tot
    return long


long = load_long()
print(f"{long.sample_id.nunique()} samples | {long.species.nunique()} species | "
      f"{long.phylum.nunique()} phyla")

# ------------------------------------------------------------------ A: phylum
if RENDER_A:
    ph = (long.groupby(["cohort", "sample_id", "phylum"])["rel"].sum().reset_index()
              .groupby(["cohort", "phylum"])["rel"].mean().reset_index())
    top = (ph.groupby("phylum")["rel"].mean().nlargest(N_PHYLA).index.tolist())
    ph["grp"] = np.where(ph.phylum.isin(top), ph.phylum, "Other")
    mat = (ph.groupby(["cohort", "grp"])["rel"].sum().unstack(fill_value=0)
             .reindex(ALL_COHORTS))
    order = top + (["Other"] if "Other" in mat.columns else [])
    mat = mat[[c for c in order if c in mat.columns]]
    mat = mat.div(mat.sum(axis=1), axis=0)          # renormalise so bars total 1

    colors = plt.get_cmap("tab20")(np.linspace(0, 0.75, len(mat.columns)))
    fig, ax = plt.subplots(figsize=A_SIZE)
    bottom = np.zeros(len(mat))
    x = np.arange(len(mat))
    for col, cl in zip(mat.columns, colors):
        ax.bar(x, mat[col].values, bottom=bottom, color=cl, width=0.74,
               edgecolor="white", linewidth=0.5, label=col)
        bottom += mat[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(mat.index, rotation=30, ha="right", fontsize=8)
    for i, c in enumerate(mat.index):           # mark the excluded cohorts
        if c in SENSITIVITY:
            ax.get_xticklabels()[i].set_style("italic")
            ax.get_xticklabels()[i].set_color("0.45")
    ax.set_ylabel("mean relative abundance")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.axvline(len(PRIMARY) - 0.5, color="0.6", lw=1, ls="--")
    ax.text(len(PRIMARY) - 0.42, 1.01, "depth-sensitivity\n(excluded)", fontsize=6.5,
            color="0.45", va="bottom")
    fig.tight_layout()
    fig.savefig(f"{OUT}/results_S1_A_phylum.{FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT}/results_S1_A_phylum.{FMT}")

# ------------------------------------------------------------------ B: PCoA
if RENDER_B:
    wide = (long.pivot_table(index="sample_id", columns="species", values="rel",
                             aggfunc="sum", fill_value=0.0))
    keep = (wide > 0).mean() >= MIN_PREV
    wide = wide.loc[:, keep]
    print(f"ordination on {wide.shape[0]} samples x {wide.shape[1]} species "
          f"(prevalence >= {MIN_PREV:.0%})")
    D = squareform(pdist(wide.values, metric="braycurtis"))
    # classical PCoA: double-centre -D^2/2, then eigendecompose
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J.dot(D ** 2).dot(J)
    vals, vecs = np.linalg.eigh(G)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    pos = vals > 0
    coords = vecs[:, pos] * np.sqrt(vals[pos])
    expl = vals[pos] / vals[pos].sum()
    pc = pd.DataFrame(coords[:, :2], index=wide.index, columns=["PC1", "PC2"])
    pc["cohort"] = long.drop_duplicates("sample_id").set_index("sample_id")["cohort"]

    pal = dict(zip(ALL_COHORTS, plt.get_cmap("tab10")(np.linspace(0, 1, 10)).tolist()
                   + [(0.4, 0.4, 0.4, 1.0)]))
    fig, ax = plt.subplots(figsize=B_SIZE)
    # 1,650 points plotted opaquely is a hairball in which no cohort structure is legible.
    # Points go faint; centroids and 1-SD ellipses carry the between-cohort signal.
    from matplotlib.patches import Ellipse
    for c in ALL_COHORTS:
        s = pc[pc.cohort == c]
        ax.scatter(s.PC1, s.PC2, s=4, alpha=0.18, color=pal[c], edgecolors="none",
                   marker="^" if c in SENSITIVITY else "o", zorder=1)
    for c in ALL_COHORTS:
        s = pc[pc.cohort == c]
        mu = s[["PC1", "PC2"]].mean().values
        cov = np.cov(s[["PC1", "PC2"]].values.T)
        w, v = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
        ax.add_patch(Ellipse(mu, 2 * np.sqrt(w[-1]), 2 * np.sqrt(w[0]), angle=ang,
                             facecolor="none", edgecolor=pal[c], lw=1.3, zorder=2,
                             ls=":" if c in SENSITIVITY else "-"))
        ax.scatter(*mu, s=85, color=pal[c], edgecolors="white", linewidths=1.4,
                   marker="^" if c in SENSITIVITY else "o", zorder=3, label=c)
    ax.set_xlabel(f"PCo1 ({100*expl[0]:.1f}%)")
    ax.set_ylabel(f"PCo2 ({100*expl[1]:.1f}%)")
    ax.legend(fontsize=6.5, markerscale=0.8, frameon=False, ncol=1,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.text(0.02, 0.02, "points: samples (faint)\nmarkers: cohort centroid\nellipse: 1 SD",
            transform=ax.transAxes, fontsize=6.5, va="bottom", color="0.35")
    fig.tight_layout()
    fig.savefig(f"{OUT}/results_S1_B_pcoa.{FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT}/results_S1_B_pcoa.{FMT}  (PCo1 {100*expl[0]:.1f}%, "
          f"PCo2 {100*expl[1]:.1f}%)")

# ------------------------------------------------------------------ C: richness
if RENDER_C:
    rich = (long.groupby(["cohort", "sample_id"])["species"].nunique()
                .reset_index(name="richness"))
    fig, ax = plt.subplots(figsize=C_SIZE)
    data = [rich[rich.cohort == c]["richness"].values for c in ALL_COHORTS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for i, (box, c) in enumerate(zip(bp["boxes"], ALL_COHORTS)):
        box.set_facecolor("0.85" if c in SENSITIVITY else "#9fb0b6")
        box.set_edgecolor("0.25"); box.set_linewidth(0.9)
        if c in SENSITIVITY:
            box.set_hatch("///")
    for el in ("medians", "whiskers", "caps"):
        for a in bp[el]:
            a.set_color("0.25"); a.set_linewidth(0.9)
    ax.set_xticks(np.arange(1, len(ALL_COHORTS) + 1))
    ax.set_xticklabels(ALL_COHORTS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("observed species per sample")
    ax.axvline(len(PRIMARY) + 0.5, color="0.6", lw=1, ls="--")
    ax.legend(handles=[Patch(facecolor="#9fb0b6", edgecolor="0.25", label="primary"),
                       Patch(facecolor="0.85", edgecolor="0.25", hatch="///",
                             label="depth-sensitivity (excluded)")],
              fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/results_S1_C_alpha.{FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    med = rich.groupby("cohort")["richness"].median().reindex(ALL_COHORTS)
    print(f"saved -> {OUT}/results_S1_C_alpha.{FMT}")
    print("median observed species per cohort:")
    print(med.to_string())
