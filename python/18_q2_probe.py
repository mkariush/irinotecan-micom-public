"""Q2 feasibility probe: does growth-coupled, carbon-limited competition produce NON-ADDITIVE
reactivation (i.e. does community structure matter beyond the threshold arithmetic)?

Mechanism tested: carriers deconjugate SN-38G to harvest GLUCURONATE as carbon. Under a carbon-
limited medium with finite SN-38G, the cooperative-tradeoff GROWTH solution (not max-secretion)
will route flux through GUS only if glucuronate genuinely helps growth. We read the INCIDENTAL
EX_sn38_m in that growth solution.

Decision:
  incidental ~0 across conditions           -> no harvesting incentive -> Q2 not feasible (bank Q1)
  incidental == min(supply,capacity)        -> delivery-limited, ADDITIVE (same as Q1) -> Q2 null
  0 < incidental < min(supply,capacity)      -> PARTIAL, community-limited -> NON-ADDITIVE -> Q2 GO
"""
import glob, os, gc
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR = "data/processed/models"
MEDIUM = "data/media/western_diet_gut_agora.qza"
SN38, SN38G = "EX_sn38_m", "EX_sn38g_m"
base_med = dict(zip(*[load_qiime_medium(MEDIUM)[c] for c in ("reaction", "flux")]))
cap1 = pd.read_parquet("data/processed/flux/full_capacity.parquet").set_index("sample_id")["sn38_capacity"]

# one higher-capacity and one mid-capacity YuJ model
SAMPLES = ["SZAXPI003410-3", "SZAXPI003417-4"]
# (medium scale, SN-38G supply)
CONDITIONS = [(1.00, 10), (0.10, 10), (0.05, 10), (0.05, 50), (0.02, 5)]

def cap_armA(com, supply):
    have = {r.id for r in com.exchanges}
    with com:                                  # isolate state
        med = dict(base_med); med[SN38G] = supply
        com.medium = {r: f for r, f in med.items() if r in have}
        com.objective = com.reactions.get_by_id(SN38)
        s = com.optimize()
        val = float(s.objective_value) if s else 0.0
    return val

rows = []
for sid in SAMPLES:
    p = f"{MODELS_DIR}/{sid}.pickle"
    if not os.path.exists(p):
        print(f"{sid}: not found"); continue
    com = load_pickle(p)
    if SN38 not in com.reactions:
        print(f"{sid}: no EX_sn38_m"); del com; gc.collect(); continue
    have = {r.id for r in com.exchanges}
    print(f"\n{sid}: stage1 capacity (saturating) = {cap1.get(sid, float('nan')):.1f}", flush=True)
    for scale, supply in CONDITIONS:
        # growth solution (cooperative tradeoff) under carbon-limited medium + finite SN-38G.
        # Mirror the working script 09: with-context, call cooperative_tradeoff on a clean model
        # (do NOT pre-set community_objective.lb).
        growth, incidental = float("nan"), float("nan")
        try:
            with com:
                med = {r: f * scale for r, f in base_med.items()}
                med[SN38G] = supply                   # SN-38G NOT scaled (finite supply)
                com.medium = {r: f for r, f in med.items() if r in have}
                sol = com.cooperative_tradeoff(fraction=0.5)
                growth = sol.growth_rate
                incidental = com.reactions.get_by_id(SN38).flux
        except Exception as e:
            print(f"  scale={scale} supply={supply}: ERR {repr(e)[:90]}", flush=True)
        armA = cap_armA(com, supply)              # max-secretion ceiling = min(supply, capacity)
        rows.append({"sample": sid, "scale": scale, "supply": supply,
                     "growth": round(growth, 4), "incidental_sn38": round(incidental, 3),
                     "ceiling_minSC": round(armA, 3)})
        print(f"  scale={scale:>4} supply={supply:>3}: growth={growth:.4f}  "
              f"incidental_SN38={incidental:.3f}  ceiling={armA:.3f}", flush=True)
    del com; gc.collect()

df = pd.DataFrame(rows)
df.to_csv("data/processed/flux/q2_probe.csv", index=False)
print("\n=== Q2 probe summary ===")
print(df.to_string(index=False))
print("\nVerdict guide: incidental==ceiling -> additive/delivery-limited (Q2 null);")
print("0<incidental<ceiling -> NON-ADDITIVE (Q2 go); incidental~0 -> no incentive.")
