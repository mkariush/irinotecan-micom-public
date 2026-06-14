"""Focused A==B re-validation on the cohorts ADDED after the original 35-sample check.
A==B is mechanistic (cost-free hydrolysis -> growth non-binding), so a handful of new-cohort samples
suffices to confirm it generalizes. Processes smallest models first (fastest cooperative-tradeoff QP).
Writes data/processed/flux/AeqB_validation_new.parquet; combine with the original AeqB table for R3.
"""
import os, glob
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR  = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
SN38, SN38G = "EX_sn38_m", "EX_sn38g_m"
NEW = ["YachidaS_2019", "ThomasAM_2019_c", "GuptaA_2019", "HanniganGD_2017"]
PER = 2                    # smallest models per new cohort
TRADEOFF, GROWTH_FRACTION, REL_TOL = 0.5, 0.95, 1e-3

def _med(com):
    m = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))
    m[SN38G] = 1000.0
    have = {r.id for r in com.exchanges}
    com.medium = {r: f for r, f in m.items() if r in have}

def arm_a(com):
    with com:
        _med(com); com.variables.community_objective.lb = 0.0
        com.objective = com.reactions.get_by_id(SN38); s = com.optimize()
        return float(s.objective_value) if s else 0.0

def arm_b(com):
    with com:
        _med(com)
        gr = com.cooperative_tradeoff(fraction=TRADEOFF).growth_rate
        com.objective = com.reactions.get_by_id(SN38)
        com.variables.community_objective.lb = GROWTH_FRACTION * gr
        s = com.optimize()
        return (float(s.objective_value) if s else 0.0), gr

if __name__ == "__main__":
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    coh = tax[["sample_id", "cohort"]].drop_duplicates("sample_id")
    # pick PER smallest pickles per new cohort (smallest model => fastest cooperative-tradeoff QP)
    sel = []
    for c in NEW:
        ids = [s for s in coh[coh.cohort == c].sample_id if os.path.exists(f"{MODELS_DIR}/{s}.pickle")]
        ids = sorted(ids, key=lambda s: os.path.getsize(f"{MODELS_DIR}/{s}.pickle"))[:PER]
        sel += [(s, c) for s in ids]
    sel = sorted(sel, key=lambda sc: os.path.getsize(f"{MODELS_DIR}/{sc[0]}.pickle"))
    print(f"Focused A==B on {len(sel)} new-cohort samples (smallest models first)", flush=True)
    rows = []
    for i, (s, c) in enumerate(sel, 1):
        com = load_pickle(f"{MODELS_DIR}/{s}.pickle")
        if SN38 not in com.reactions:
            print(f"  [{i}/{len(sel)}] {s} ({c}): no carriers (cap 0) -> trivially A==B", flush=True)
            rows.append({"sample_id": s, "cohort": c, "arm_a": 0.0, "arm_b": 0.0, "rel_diff": 0.0, "A_eq_B": True}); continue
        a = arm_a(com); b, gr = arm_b(com)
        rd = abs(a - b) / max(a, 1e-9); eq = rd < REL_TOL
        rows.append({"sample_id": s, "cohort": c, "arm_a": a, "arm_b": b, "rel_diff": rd, "A_eq_B": eq})
        print(f"  [{i}/{len(sel)}] {s} ({c}): A={a:.3f} B={b:.3f} reldiff={rd:.1e} {'OK' if eq else '*** DIVERGES ***'}", flush=True)
    df = pd.DataFrame(rows); df.to_parquet("data/processed/flux/AeqB_validation_new.parquet")
    print(f"\n=== {int(df.A_eq_B.sum())}/{len(df)} agree across {df.cohort.nunique()} new cohorts ===", flush=True)
