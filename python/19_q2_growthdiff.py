"""Q2 feasibility, cleaner test: does the community GROW MORE when SN-38G is available under carbon
limitation? If carriers harvest SN-38G's glucuronate as carbon, growth(supply>0) > growth(supply=0).
No growth gain => no harvesting => incidental SN-38 = 0 => Q2 null. Uses only growth_rate (robust).
"""
import os, gc
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MED = "data/media/western_diet_gut_agora.qza"
SN38G = "EX_sn38g_m"
base = dict(zip(*[load_qiime_medium(MED)[c] for c in ("reaction","flux")]))
SAMPLES = ["SZAXPI003410-3", "SZAXPI003417-4"]
SCALES = [1.0, 0.1, 0.05, 0.02]
SUPPLIES = [0, 50]

def growth(com, have, scale, supply):
    with com:
        med = {r: f*scale for r,f in base.items()}; med[SN38G] = supply
        com.medium = {r: f for r,f in med.items() if r in have}
        try:
            return com.cooperative_tradeoff(fraction=0.5).growth_rate
        except Exception:
            return float("nan")

rows = []
for sid in SAMPLES:
    com = load_pickle(f"data/processed/models/{sid}.pickle")
    have = {r.id for r in com.exchanges}
    print(f"\n{sid}", flush=True)
    for scale in SCALES:
        g0 = growth(com, have, scale, 0)
        g50 = growth(com, have, scale, 50)
        delta = g50 - g0
        rel = 100*delta/g0 if g0 and g0==g0 else float("nan")
        rows.append({"sample": sid, "scale": scale, "growth_noSN38G": round(g0,5),
                     "growth_SN38G50": round(g50,5), "delta": round(delta,6),
                     "pct_growth_gain": round(rel,3)})
        print(f"  scale={scale:>4}: growth(no SN38G)={g0:.5f}  growth(+SN38G)={g50:.5f}  "
              f"delta={delta:+.6f}  ({rel:+.2f}%)", flush=True)
    del com; gc.collect()

df = pd.DataFrame(rows); df.to_csv("data/processed/flux/q2_growthdiff.csv", index=False)
print("\n=== Q2 growth-difference summary ===")
print(df.to_string(index=False))
print("\nVerdict: delta>0 (meaningful) => community HARVESTS SN-38G glucuronate -> Q2 feasible.")
print("delta~0 => no harvesting incentive -> incidental SN-38 = 0 -> Q2 NULL (bank Q1).")
