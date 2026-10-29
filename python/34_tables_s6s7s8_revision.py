"""Supplementary Tables S6-S8 for the mSystems revision.

  S6  capacity under five growth media, 45-sample subset        (R2.2; from 31)
  S7  substrate uptake at the extreme growth floors             (R2.3; from 32)
  S8  total-community renormalization sensitivity, per cohort   (R1.1; from 10 w/ SENS_COVERAGE=1)

Appends to docs/supplementary_tables.md, so run AFTER 26 -> 28 -> 29 -> 30 (26 overwrites the file).
Build order is therefore:  26 -> 28 -> 29 -> 30 -> 34

    python python/34_tables_s6s7s8_revision.py
"""
import os
import pandas as pd

FLUX = "data/processed/flux"
OUT_DIR = "data/processed/tables"
OUT_MD = "docs/supplementary_tables.md"

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]
MEDIA_ORDER = ["western", "eu_average", "high_fiber", "hf_lowcarb", "himalaya"]
MEDIA_LABEL = {"western": "Western", "eu_average": "European average",
               "high_fiber": "High fibre", "hf_lowcarb": "High fat / low carbohydrate",
               "himalaya": "Non-western (Himalayan)"}

os.makedirs(OUT_DIR, exist_ok=True)
md = []


def cohort_sort(df):
    df = df.copy()
    df["cohort"] = pd.Categorical(df["cohort"], categories=PRIMARY, ordered=True)
    return df.sort_values(["cohort", "sample_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- S6: diet
d = pd.read_parquet(f"{FLUX}/diet_sensitivity.parquet")
wide = d.pivot(index="sample_id", columns="diet", values="sn38_capacity")
wide = wide[[c for c in MEDIA_ORDER if c in wide.columns]]
meta = d[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")
s6 = cohort_sort(wide.reset_index().merge(meta, on="sample_id"))
s6["max_abs_dev"] = (wide.sub(wide["western"], axis=0).abs().max(axis=1)).values

tab6 = pd.DataFrame({"Sample ID": s6.sample_id, "Cohort": s6.cohort.astype(str),
                     "Condition": s6.study_condition})
for m in MEDIA_ORDER:
    tab6[MEDIA_LABEL[m]] = s6[m].round(4)
tab6["Max |deviation from Western|"] = s6.max_abs_dev.map(lambda v: f"{v:.1e}")
tab6.to_csv(f"{OUT_DIR}/table_S6_diet.csv", index=False)

md.append(
    "## Table S6. SN-38 reactivation capacity under five growth media\n\n"
    f"Capacity (relative units) for the stratified {len(tab6)}-sample subset (five samples from each "
    "of the nine primary cohorts) recomputed under five growth media. As applied to the community "
    "models the media differ from the Western diet in 64–82 metabolites and span a 37-fold range in "
    "total applied uptake flux (Text S1.5). Capacity is identical under all five media in every "
    f"sample; the largest absolute deviation across the whole table is "
    f"{s6.max_abs_dev.max():.1e} relative units "
    f"({(wide.sub(wide['western'],axis=0).abs().div(wide['western'].where(wide['western']>0),axis=0)).max().max():.1e} "
    "relative), i.e. numerical noise.\n\n" + tab6.to_markdown(index=False) + "\n")

# ---------------------------------------------------------------- S7: uptake
p = pd.read_parquet(f"{FLUX}/growth_pareto.parquet")
fmax = p.fraction.max()
ends = p[p.fraction.isin([0.0, fmax])]
u = ends.pivot(index="sample_id", columns="fraction",
               values=["sn38", "total_uptake", "sn38g_uptake", "growth"])
# flatten the two-level pivot columns so the metadata merge is single-level
u.columns = [f"{var}@{frac}" for var, frac in u.columns]
meta7 = p[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")
s7 = cohort_sort(u.reset_index().merge(meta7, on="sample_id"))

tab7 = pd.DataFrame({
    "Sample ID": s7.sample_id, "Cohort": s7.cohort.astype(str), "Condition": s7.study_condition,
    "SN-38 (f=0)": s7["sn38@0.0"].round(4),
    f"SN-38 (f={fmax})": s7[f"sn38@{fmax}"].round(4),
    "SN-38G uptake (f=0)": s7["sn38g_uptake@0.0"].round(4),
    f"SN-38G uptake (f={fmax})": s7[f"sn38g_uptake@{fmax}"].round(4),
    "Total uptake (f=0)": s7["total_uptake@0.0"].round(4),
    f"Total uptake (f={fmax})": s7[f"total_uptake@{fmax}"].round(4),
})
tab7["Extra uptake (%)"] = (100 * (s7[f"total_uptake@{fmax}"] / s7["total_uptake@0.0"] - 1)).round(1)
tab7.to_csv(f"{OUT_DIR}/table_S7_uptake.csv", index=False)

ratio = (s7[f"total_uptake@{fmax}"] / s7["total_uptake@0.0"]).median()
# NOT 45/45: one community (CCIS00281083ST-3-0) draws 1.1% beyond SN-38G at f=0. Count, don't assume.
n_exact = int((tab7["Total uptake (f=0)"] - tab7["SN-38G uptake (f=0)"]).abs().le(1e-6).sum())
md.append(
    "## Table S7. Substrate uptake at the extremes of the growth-reactivation front\n\n"
    "For each sample, SN-38 secretion and substrate uptake with no growth requirement (f = 0) and "
    f"with community growth forced to {fmax:.3f} of its maximum (f = {fmax}). Uptake profiles are "
    "deterministic: `EX_sn38_m` was pinned at its optimum and total uptake minimized (Text S1.6). "
    f"SN-38 secretion and SN-38G uptake are unchanged between the two; at f = 0 total uptake equals "
    f"SN-38G uptake exactly in {n_exact} of {len(tab7)} communities, which therefore take up no other "
    f"metabolite, whereas forcing near-maximal growth raises total nutrient uptake by a median of "
    f"{100*(ratio-1):.1f}% with SN-38 unchanged.\n\n"
    + tab7.to_markdown(index=False) + "\n")

# ---------------------------------------------------------------- S8: renormalization
a = pd.read_csv(f"{FLUX}/results_crc_meta.csv")
b = pd.read_csv(f"{FLUX}/results_crc_meta_covsens.csv")
m8 = a.merge(b, on="cohort", suffixes=("_pub", "_sens"))
m8["cohort"] = pd.Categorical(m8["cohort"], categories=PRIMARY, ordered=True)
m8 = m8.sort_values("cohort")


def direction(rb):
    return "control > CRC" if rb > 0 else "CRC > control"


tab8 = pd.DataFrame({
    "Cohort": m8.cohort.astype(str), "n CRC": m8.n_CRC_pub, "n control": m8.n_ctrl_pub,
    "Median CRC (primary)": m8.med_CRC_pub, "Median control (primary)": m8.med_ctrl_pub,
    "p (primary)": m8.p_pub.map(lambda v: f"{v:.4g}"),
    "p (total-community)": m8.p_sens.map(lambda v: f"{v:.4g}"),
    "Direction (primary)": m8.rank_biserial_pub.map(direction),
    "Direction (total-community)": m8.rank_biserial_sens.map(direction),
})
tab8.to_csv(f"{OUT_DIR}/table_S8_renorm.csv", index=False)

md.append(
    "## Table S8. CRC versus control under the total-community renormalization sensitivity\n\n"
    "Per-cohort Mann–Whitney comparisons of reactivation capacity on the primary basis (capacity "
    "renormalized within the AGORA2-matched community) and on the total-community basis (capacity × "
    "AGORA2-matched abundance fraction, removing the model-assembly renormalization; Text S1.4). "
    "The meta-analysis is null on both bases (Stouffer z = 0.80, p = 0.42 primary; z = 0.99, p = 0.32 "
    "sensitivity) and per-sample ranking is near-identical (Spearman ρ = 0.985). ZellerG_2014 "
    "attenuates from p = 0.016 to p = 0.066, leaving three rather than four individually significant "
    "cohorts; the opposing-direction cohorts on which the null rests are unaffected. VogtmannE_2016 "
    "changes nominal direction but is null on both bases.\n\n" + tab8.to_markdown(index=False) + "\n")

with open(OUT_MD, "a", encoding="utf-8") as f:
    f.write("\n---\n\n" + "\n---\n\n".join(md))

print(f"S6 {tab6.shape} -> {OUT_DIR}/table_S6_diet.csv")
print(f"S7 {tab7.shape} -> {OUT_DIR}/table_S7_uptake.csv")
print(f"S8 {tab8.shape} -> {OUT_DIR}/table_S8_renorm.csv")
print(f"appended markdown -> {OUT_MD}")
