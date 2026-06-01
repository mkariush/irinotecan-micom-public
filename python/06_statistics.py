"""Compare SN-38 reactivation flux: CRC vs healthy; staging analysis; meta-analysis."""

import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

FLUX_DIR      = "data/processed/flux"
TAXONOMY_PATH = "data/processed/taxonomy_micom.parquet"
GROWTH_DIR    = "data/processed/growth"
RESULTS_DIR   = "data/processed/results"

# Candidate staging columns in curatedMetagenomicData — tried in order
STAGING_COLS = ["disease_stage", "ajcc_stage", "tumor_stage", "disease_stage_I_III"]

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    flux = pd.read_parquet(os.path.join(FLUX_DIR, "sn38_flux.parquet"))
    taxonomy = pd.read_parquet(TAXONOMY_PATH)
    meta = taxonomy.drop_duplicates("sample_id").drop(columns=["id", "abundance"], errors="ignore")
    return flux.merge(meta, on="sample_id", how="left")


def crc_vs_control_per_cohort(df: pd.DataFrame, flux_col: str) -> pd.DataFrame:
    """Mann-Whitney U (one-sided: CRC > control) per cohort with FDR correction."""
    records = []
    for cohort, grp in df.groupby("cohort"):
        crc = grp.loc[grp["study_condition"] == "CRC", flux_col].dropna()
        ctl = grp.loc[grp["study_condition"] == "control", flux_col].dropna()
        if len(crc) < 3 or len(ctl) < 3:
            continue
        stat, pval = stats.mannwhitneyu(crc, ctl, alternative="greater")
        n1, n2 = len(crc), len(ctl)
        records.append({
            "cohort":      cohort,
            "n_crc":       n1,
            "n_ctl":       n2,
            "median_crc":  crc.median(),
            "median_ctl":  ctl.median(),
            "fold_change": crc.median() / ctl.median() if ctl.median() > 0 else float("nan"),
            "U_stat":      stat,
            "p_value":     pval,
            "effect_r":    1 - (2 * stat) / (n1 * n2),
        })
    result = pd.DataFrame(records)
    if len(result):
        _, result["p_adj"], _, _ = multipletests(result["p_value"], method="fdr_bh")
    return result


def staging_analysis(df: pd.DataFrame, flux_col: str) -> dict:
    """
    Within-CRC: test association between disease stage and SN-38 flux.
    Uses Kruskal-Wallis + pairwise Mann-Whitney with FDR correction.
    Returns empty dict if no staging data available.
    """
    crc = df[df["study_condition"] == "CRC"].copy()

    stage_col = next(
        (c for c in STAGING_COLS if c in crc.columns and crc[c].notna().any()), None
    )
    if stage_col is None:
        print("No staging column found in data.")
        return {}

    crc = crc[crc[stage_col].notna()].copy()
    print(f"\nStaging column: '{stage_col}'")
    print(crc.groupby([stage_col, "cohort"]).size().unstack(fill_value=0).to_string())

    stages = sorted(crc[stage_col].unique())
    groups = {s: crc.loc[crc[stage_col] == s, flux_col].dropna() for s in stages}
    groups = {s: g for s, g in groups.items() if len(g) >= 3}

    if len(groups) < 2:
        print("Fewer than 2 stages with n≥3 — skipping staging analysis.")
        return {}

    h_stat, kw_p = stats.kruskal(*groups.values())
    print(f"\nKruskal-Wallis H={h_stat:.3f}, p={kw_p:.4f}")

    pairs = []
    stage_list = list(groups.keys())
    for i, s1 in enumerate(stage_list):
        for s2 in stage_list[i + 1:]:
            u, p = stats.mannwhitneyu(groups[s1], groups[s2], alternative="two-sided")
            pairs.append({
                "stage1": s1, "stage2": s2,
                "n1": len(groups[s1]), "n2": len(groups[s2]),
                "median1": groups[s1].median(), "median2": groups[s2].median(),
                "U_stat": u, "p_value": p,
            })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        _, pairs_df["p_adj"], _, _ = multipletests(pairs_df["p_value"], method="fdr_bh")

    return {
        "stage_col": stage_col,
        "kw_stat":   h_stat,
        "kw_p":      kw_p,
        "pairwise":  pairs_df,
    }


def taxa_contributions(df: pd.DataFrame, flux_col: str) -> pd.DataFrame:
    """Spearman correlation between per-taxon growth rate and community SN-38 flux."""
    growth = pd.read_parquet(os.path.join(GROWTH_DIR, "growth_rates.parquet"))
    merged = growth.merge(
        df[["sample_id", flux_col, "study_condition"]], on="sample_id", how="inner"
    )
    records = []
    for taxon, grp in merged.groupby("taxon"):
        r, p = stats.spearmanr(grp["growth_rate"], grp[flux_col])
        records.append({"taxon": taxon, "spearman_r": r, "p_value": p, "n": len(grp)})
    result = pd.DataFrame(records).sort_values("spearman_r", ascending=False)
    _, result["p_adj"], _, _ = multipletests(result["p_value"], method="fdr_bh")
    return result


def plot_cohort_comparison(df: pd.DataFrame, flux_col: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df, x="cohort", y=flux_col, hue="study_condition",
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


def plot_staging(df: pd.DataFrame, flux_col: str, stage_col: str, out_path: str):
    crc = df[df["study_condition"] == "CRC"].dropna(subset=[stage_col])
    order = sorted(crc[stage_col].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=crc, x=stage_col, y=flux_col, order=order,
                palette="Reds", ax=ax)
    ax.set_xlabel("Disease stage")
    ax.set_ylabel("SN-38 reactivation flux (mmol/gDW/h)")
    ax.set_title("Predicted SN-38 reactivation by CRC stage")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    flux_cols = [c for c in df.columns
                 if c not in {"sample_id", "cohort", "study_condition"} | set(STAGING_COLS)
                 and df[c].dtype.kind == "f"]
    flux_col = flux_cols[0]
    df = df.rename(columns={flux_col: "sn38_flux"})

    # 1. CRC vs healthy per cohort
    cohort_results = crc_vs_control_per_cohort(df, "sn38_flux")
    cohort_results.to_csv(os.path.join(RESULTS_DIR, "cohort_comparison.csv"), index=False)
    print(cohort_results.to_string())
    plot_cohort_comparison(df, "sn38_flux", os.path.join(RESULTS_DIR, "cohort_comparison.png"))

    # 2. Staging analysis (within CRC)
    staging = staging_analysis(df, "sn38_flux")
    if staging:
        staging["pairwise"].to_csv(os.path.join(RESULTS_DIR, "staging_pairwise.csv"), index=False)
        print("\nPairwise staging comparisons:")
        print(staging["pairwise"].to_string())
        plot_staging(df, "sn38_flux", staging["stage_col"],
                     os.path.join(RESULTS_DIR, "staging_comparison.png"))

    # 3. Taxa contributions
    taxa_results = taxa_contributions(df, "sn38_flux")
    taxa_results.to_csv(os.path.join(RESULTS_DIR, "taxa_contributions.csv"), index=False)
    print("\nTop 10 taxa by Spearman r:")
    print(taxa_results.head(10).to_string())
