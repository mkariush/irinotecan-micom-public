"""Compare SN-38 reactivation flux: CRC vs healthy; meta-analysis across cohorts."""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

FLUX_DIR      = "data/processed/flux"
TAXONOMY_PATH = "data/processed/taxonomy_micom.parquet"
RESULTS_DIR   = "data/processed/results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    flux     = pd.read_parquet(os.path.join(FLUX_DIR, "sn38_flux.parquet"))
    taxonomy = pd.read_parquet(TAXONOMY_PATH)

    meta = (
        taxonomy[["sample_id", "cohort", "study_condition"]]
        .drop_duplicates("sample_id")
    )
    return flux.merge(meta, on="sample_id", how="left")


def crc_vs_control_per_cohort(df: pd.DataFrame, flux_col: str) -> pd.DataFrame:
    """Mann-Whitney U test per cohort; return effect sizes and p-values."""
    records = []
    for cohort, grp in df.groupby("cohort"):
        crc = grp.loc[grp["study_condition"] == "CRC", flux_col].dropna()
        ctl = grp.loc[grp["study_condition"] == "control", flux_col].dropna()

        if len(crc) < 3 or len(ctl) < 3:
            continue

        stat, pval = stats.mannwhitneyu(crc, ctl, alternative="greater")
        # rank-biserial correlation as effect size
        n1, n2 = len(crc), len(ctl)
        r = 1 - (2 * stat) / (n1 * n2)

        records.append({
            "cohort":  cohort,
            "n_crc":   n1,
            "n_ctl":   n2,
            "median_crc": crc.median(),
            "median_ctl": ctl.median(),
            "U_stat":  stat,
            "p_value": pval,
            "effect_r": r,
        })

    result = pd.DataFrame(records)
    _, result["p_adj"], _, _ = multipletests(result["p_value"], method="fdr_bh")
    return result


def taxa_contributions(df: pd.DataFrame, growth_rates_path: str) -> pd.DataFrame:
    """Correlate per-taxon growth rate with community SN-38 flux."""
    growth = pd.read_parquet(growth_rates_path)
    merged = growth.merge(
        df[["sample_id", "sn38_flux", "study_condition"]],
        on="sample_id", how="inner"
    )

    records = []
    for taxon, grp in merged.groupby("taxon"):
        r, p = stats.spearmanr(grp["growth_rate"], grp["sn38_flux"])
        records.append({"taxon": taxon, "spearman_r": r, "p_value": p, "n": len(grp)})

    result = pd.DataFrame(records).sort_values("spearman_r", ascending=False)
    _, result["p_adj"], _, _ = multipletests(result["p_value"], method="fdr_bh")
    return result


def plot_cohort_comparison(df: pd.DataFrame, flux_col: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="cohort", y=flux_col, hue="study_condition",
        order=sorted(df["cohort"].unique()),
        palette={"CRC": "#d62728", "control": "#1f77b4"},
        ax=ax,
    )
    ax.set_xlabel("Cohort")
    ax.set_ylabel("SN-38 reactivation flux (mmol/gDW/h)")
    ax.set_title("Predicted SN-38 reactivation: CRC vs healthy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_data()

    flux_cols = [c for c in df.columns if c not in ("sample_id", "cohort", "study_condition")]
    flux_col  = flux_cols[0]

    # Rename for convenience
    df = df.rename(columns={flux_col: "sn38_flux"})

    cohort_results = crc_vs_control_per_cohort(df, "sn38_flux")
    cohort_results.to_csv(os.path.join(RESULTS_DIR, "cohort_comparison.csv"), index=False)
    print(cohort_results.to_string())

    taxa_results = taxa_contributions(
        df, os.path.join("data/processed/growth", "growth_rates.parquet")
    )
    taxa_results.to_csv(os.path.join(RESULTS_DIR, "taxa_contributions.csv"), index=False)
    print("\nTop 10 taxa by Spearman r:")
    print(taxa_results.head(10).to_string())

    plot_cohort_comparison(df, "sn38_flux", os.path.join(RESULTS_DIR, "cohort_comparison.png"))
    print(f"\nPlot saved to {RESULTS_DIR}/cohort_comparison.png")
