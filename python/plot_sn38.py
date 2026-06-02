"""Plot SN-38 reactivation capacity (Arm A vs Arm B) and driver taxa.

Improves on Heinken Fig 5 by putting a MEANINGFUL variable on the x-axis
(CRC vs healthy, or cohort) instead of jitter. Run after 05_drug_flux.py.

Outputs (data/processed/figures/):
  fig1_capacity_by_condition.png  - Arm A vs Arm B per group (the comparison)
  fig2_arm_gap.png                - per-sample A-B gap (theoretical vs realized)
  fig3_capacity_vs_abundance.png  - Heinken panel-b recreation: linearity test
  fig4_driver_taxa.png            - top beta-glucuronidase contributing taxa
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FLUX_DIR    = "data/processed/flux"
TAX_PATH    = "data/processed/taxonomy_micom.parquet"
META_PATH   = "data/processed/sample_metadata.parquet"  # optional: sample_id, study_condition, cohort
FIG_DIR     = "data/processed/figures"

# Known AGORA2 beta-glucuronidase producers (CLAUDE.md) -> for abundance proxy
GUS_GENERA = ("Bacteroides", "Parabacteroides", "Bifidobacterium", "Clostridium",
              "Enterocloster", "Erysipelatoclostridium", "Collinsella")

sns.set_theme(style="whitegrid", context="talk")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    cap = pd.read_parquet(os.path.join(FLUX_DIR, "sn38_capacity.parquet"))
    contrib = pd.read_parquet(os.path.join(FLUX_DIR, "sn38_taxa_contributions.parquet"))
    # attach grouping variable: prefer study_condition, fall back to cohort, else "all"
    if os.path.exists(META_PATH):
        meta = pd.read_parquet(META_PATH)
        cap = cap.merge(meta, on="sample_id", how="left")
    if "study_condition" in cap.columns:
        cap["group"] = cap["study_condition"]
    elif "cohort" in cap.columns:
        cap["group"] = cap["cohort"]
    else:
        cap["group"] = "all samples"
    return cap, contrib


def fig1_capacity_by_condition(cap):
    """Arm A (theoretical ceiling) vs Arm B (realized) per group."""
    long = cap.melt(
        id_vars=["sample_id", "group"],
        value_vars=["sn38_capacity_unconstrained", "sn38_capacity_constrained"],
        var_name="arm", value_name="flux",
    )
    long["arm"] = long["arm"].map({
        "sn38_capacity_unconstrained": "Arm A: theoretical max (Heinken-style)",
        "sn38_capacity_constrained":   "Arm B: growth-constrained (realized)",
    })
    plt.figure(figsize=(10, 7))
    ax = sns.violinplot(data=long, x="group", y="flux", hue="arm",
                        split=False, inner=None, cut=0)
    # make violin bodies semi-transparent so individual points show through
    for coll in ax.collections:
        coll.set_alpha(0.35)
    # individual sample points on top, dodged to match each violin
    sns.stripplot(data=long, x="group", y="flux", hue="arm",
                  dodge=True, jitter=0.08, size=9, alpha=0.95,
                  edgecolor="black", linewidth=0.8, ax=ax, legend=False)
    ax.set_ylabel("SN-38 reactivation flux\n(mmol / gDW / h)")
    ax.set_xlabel("")
    ax.set_title("SN-38 reactivation capacity: theoretical vs realized")
    # keep only the two violin (arm) legend entries, drop duplicates from stripplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="", loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_capacity_by_condition.png"), dpi=200)
    plt.close()


def fig2_arm_gap(cap):
    """Per-sample gap between theoretical ceiling and realized capacity."""
    cap = cap.copy()
    cap["gap"] = cap["sn38_capacity_unconstrained"] - cap["sn38_capacity_constrained"]
    cap["gap_pct"] = 100 * cap["gap"] / cap["sn38_capacity_unconstrained"].replace(0, np.nan)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=cap.sort_values("gap_pct"), x="sample_id", y="gap_pct",
                     hue="group", dodge=False)
    ax.set_ylabel("% of theoretical capacity\nLOST under growth constraint")
    ax.set_xlabel("sample")
    ax.set_title("How much gene-content capacity is NOT realized by a growing community")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_arm_gap.png"), dpi=200)
    plt.close()


def fig3_capacity_vs_abundance(cap, contrib=None):
    """Heinken panel-b recreation: does Arm B break the abundance-flux linearity?
    x-axis = summed relative abundance of taxa that ACTUALLY carry a SN38G_GLCAASE
    reaction (from the contributions table), not crude genus membership."""
    if not os.path.exists(TAX_PATH):
        print("  (skip fig3: taxonomy not found)")
        return
    tax = pd.read_parquet(TAX_PATH).copy()
    tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)   # match contrib naming
    if contrib is not None and not contrib.empty:
        # GUS-carrier taxa per sample = those with non-zero SN38G_GLCAASE flux
        carriers = contrib[["sample_id", "taxon"]].drop_duplicates()
        merged = carriers.merge(tax[["sample_id", "taxon", "abundance"]],
                                on=["sample_id", "taxon"], how="left")
        gus_ab = (merged.groupby("sample_id")["abundance"].sum()
                  .rename("gus_abundance").reset_index())
    else:
        tax["genus"] = tax["id"].str.split().str[0]
        gus_ab = (tax[tax["genus"].isin(GUS_GENERA)]
                  .groupby("sample_id")["abundance"].sum()
                  .rename("gus_abundance").reset_index())
    df = cap.merge(gus_ab, on="sample_id", how="left")
    print("  fig3 GUS-carrier abundance per sample:")
    print(df[["sample_id", "gus_abundance",
              "sn38_capacity_constrained"]].to_string(index=False))
    plt.figure(figsize=(9, 7))
    for col, lab, c in [("sn38_capacity_unconstrained", "Arm A (theoretical)", "tab:blue"),
                        ("sn38_capacity_constrained",   "Arm B (realized)",   "tab:red")]:
        sns.regplot(data=df, x="gus_abundance", y=col, label=lab,
                    scatter_kws=dict(s=40, alpha=0.6, color=c),
                    line_kws=dict(color=c), ci=None)
    plt.xlabel("Summed abundance of\nbeta-glucuronidase producer taxa")
    plt.ylabel("SN-38 reactivation flux\n(mmol / gDW / h)")
    plt.title("Does realized capacity track gene abundance?")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_capacity_vs_abundance.png"), dpi=200)
    plt.close()


def fig4_driver_taxa(contrib):
    """Top beta-glucuronidase flux contributors across samples (validation vs literature)."""
    if contrib.empty:
        print("  (skip fig4: no taxa contributions)")
        return
    top = (contrib.groupby("taxon")["gus_flux"].sum()
           .sort_values(ascending=False).head(15).reset_index())
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=top, y="taxon", x="gus_flux", color="tab:green")
    ax.set_xlabel("Total SN38G_GLCAASE flux (summed across samples)")
    ax.set_ylabel("")
    ax.set_title("Driver taxa of SN-38 reactivation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_driver_taxa.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    cap, contrib = load_data()
    print(f"Loaded {len(cap)} samples, {len(contrib)} taxa-contribution rows")
    fig1_capacity_by_condition(cap)
    fig2_arm_gap(cap)
    fig3_capacity_vs_abundance(cap, contrib)
    fig4_driver_taxa(contrib)
    print(f"Figures written to {FIG_DIR}")
