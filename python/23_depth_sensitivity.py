"""Depth-sensitivity analysis for the cohort expansion (Methods/limitations support).

Two questions:
  1. Is predicted capacity confounded by sequencing depth? (deeper -> more detected carriers -> higher
     capacity). Test corr(capacity, reads) within and across cohorts; locate Gupta/Hannigan.
  2. Would including the shallow cohorts (Gupta, Hannigan) change the primary CRC-vs-control verdict?
     Re-run the Stouffer meta WITH them and confirm the null is unchanged.
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, norm

FLUX = "data/processed/flux"
PRIMARY = ["ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a","ThomasAM_2018b",
           "WirbelJ_2018","VogtmannE_2016","YachidaS_2019","ThomasAM_2019_c"]
SENS = ["GuptaA_2019","HanniganGD_2017"]

cap = pd.read_parquet(f"{FLUX}/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
smeta = tax[["sample_id","cohort","study_condition","cohort_set"]].drop_duplicates("sample_id")
cap = cap.merge(smeta, on="sample_id", how="left")

# per-sample read depth from the raw cMD metadata
depth = []
for p in glob.glob("data/raw/*_metadata.parquet"):
    d = pd.read_parquet(p)
    col = "number_reads" if "number_reads" in d.columns else None
    if col: depth.append(d[["sample_id", col]].rename(columns={col: "reads"}))
depth = pd.concat(depth, ignore_index=True).drop_duplicates("sample_id")
cap = cap.merge(depth, on="sample_id", how="left")
cap["reads_M"] = cap.reads/1e6

print("=== 1. capacity vs sequencing depth ===")
rs, p = spearmanr(cap.reads_M, cap.sn38_capacity, nan_policy="omit")
print(f"all samples: Spearman(capacity, reads) rho={rs:.3f} (p={p:.2g}), n={cap.reads.notna().sum()}")
# within-cohort (controls for cohort-level biology)
within = []
for c, g in cap.groupby("cohort"):
    if g.reads.notna().sum() >= 10:
        rr, _ = spearmanr(g.reads_M, g.sn38_capacity, nan_policy="omit")
        within.append((c, round(rr,3), round(g.reads_M.median(),1), round(g.sn38_capacity.median(),1)))
w = pd.DataFrame(within, columns=["cohort","rho_within","reads_M_med","cap_med"]).sort_values("reads_M_med")
print("\nwithin-cohort corr + medians (sorted by depth):")
print(w.to_string(index=False))
print("\n-> if shallow cohorts (Gupta/Hannigan) have low cap_med AND positive within-cohort rho,")
print("   their low capacity is at least partly depth-driven -> justifies primary exclusion.")

# ---- 2. does adding the shallow cohorts change the CRC-vs-control meta? ----
def stouffer(df):
    rows=[]
    for c,g in df.groupby("cohort"):
        crc=g.loc[g.study_condition=="CRC","sn38_capacity"]; ctl=g.loc[g.study_condition=="control","sn38_capacity"]
        if len(crc)<3 or len(ctl)<3: continue
        u,pp=mannwhitneyu(crc,ctl,alternative="two-sided"); rbc=1-2*u/(len(crc)*len(ctl))
        rows.append((c,pp,rbc,len(crc)+len(ctl)))
    r=pd.DataFrame(rows,columns=["cohort","p","rbc","n"])
    zs=norm.isf(r.p/2)*np.sign(r.rbc); wt=np.sqrt(r.n)
    z=(zs*wt).sum()/np.sqrt((wt**2).sum()); return z, 2*norm.sf(abs(z)), len(r)

z_p, p_p, k_p = stouffer(cap[cap.cohort.isin(PRIMARY)])
z_a, p_a, k_a = stouffer(cap[cap.cohort.isin(PRIMARY+SENS)])
print("\n=== 2. CRC vs control meta, primary vs +sensitivity ===")
print(f"PRIMARY ({k_p} cohorts):       Stouffer z={z_p:.2f}, p={p_p:.3g}")
print(f"+SENSITIVITY ({k_a} cohorts):  Stouffer z={z_a:.2f}, p={p_a:.3g}")
print("-> if both remain non-significant, the shallow cohorts do not change the central R5 null.")
