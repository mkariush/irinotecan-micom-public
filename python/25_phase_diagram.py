"""Prototype phase diagram: when does microbial composition control SN-38 reactivation?

We CANNOT pin the two absolute scales (substrate delivery D, real community capacity), so we sweep
them and map the regime, using our REAL 1509-community relative-capacity DISTRIBUTION (shape fixed by
data; only the absolute scale is unknown). realized_i = min(D, capacity_i); composition matters where
capacity_i < D (capacity-limited / "protected"); host-gated where capacity_i > D.

ANCHORS (real units, mmol/gDW/h), all order-of-magnitude:
  Capacity (Guthrie 2017 ex-vivo, 100 uM SN-38G, 200 ug/ml protein, ~0.55 g protein/gDW):
     high = high-metabolizer MIDPOINT (~52% conv) -> ~2.75; low = low-metabolizer near bottom of range
     (~1% conv) -> ~0.05 (CONSERVATIVE; low-metabolizer midpoint ~4.4% would give ~0.24).
     -> plausible community-capacity MEDIAN band ~0.05-2.75 mmol/gDW/h.
  Delivery D (Slatter mass-balance: ~0.02-0.05 mmol SN-38G to gut/dose, /~1 day, /~60 gDW biomass):
     ~1e-5 (avg) to ~1e-4 (peak); widen to 1e-6..1e-3 for uncertainty (dose, UGT1A1, local peaks).
Output: heatmap of % communities capacity-limited over (D x capacity-scale); plausible box overlaid.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY = ["ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a","ThomasAM_2018b",
           "WirbelJ_2018","VogtmannE_2016","YachidaS_2019","ThomasAM_2019_c"]
cap = pd.read_parquet("data/processed/flux/full_capacity.parquet")
tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
coh = tax[["sample_id","cohort"]].drop_duplicates("sample_id")
cap = cap.merge(coh, on="sample_id").query("cohort in @PRIMARY")
rel = cap.sn38_capacity.values                     # relative capacity (0-93), shape from real data
rel_med = np.median(rel[rel > 0])

# axes (log): delivery D and real capacity MEDIAN (the unknown absolute scale)
D_grid   = np.logspace(-6, -2, 60)                 # mmol/gDW/h
cap_grid = np.logspace(-2, 1.0, 60)                # community-capacity median, mmol/gDW/h

def frac_capacity_limited(cap_median, D):
    real = rel * (cap_median / rel_med)             # scale distribution to this median, keep shape
    return float(np.mean(real < D)) * 100           # % capacity-limited (composition matters)

Z = np.array([[frac_capacity_limited(cm, d) for d in D_grid] for cm in cap_grid])

# plausible boxes (literature)
D_box   = (1e-5, 1e-4)      # delivery avg->peak
cap_box = (0.05, 2.75)      # Guthrie low->high metabolizer

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
im = ax.pcolormesh(np.log10(D_grid), np.log10(cap_grid), Z, cmap="RdYlBu_r", shading="auto", vmin=0, vmax=100)
cb = fig.colorbar(im, ax=ax); cb.set_label("% communities CAPACITY-limited (composition matters)")
ax.add_patch(plt.Rectangle((np.log10(D_box[0]), np.log10(cap_box[0])),
             np.log10(D_box[1])-np.log10(D_box[0]), np.log10(cap_box[1])-np.log10(cap_box[0]),
             fill=False, edgecolor="k", lw=2.5, ls="--"))
ax.text(np.log10(np.sqrt(D_box[0]*D_box[1])), np.log10(np.sqrt(cap_box[0]*cap_box[1])),
        "physiologically\nplausible", ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("log10  substrate delivery D  (mmol/gDW/h)")
ax.set_ylabel("log10  community capacity median  (mmol/gDW/h)")
# title removed -- interpretive content (blue=host-gated, red=composition-controlled) is in the caption

# Panel (b): local-peak sensitivity -- how far above average delivery before composition matters?
D_AVG = 1e-5
CAP_CENTRAL = float(np.sqrt(cap_box[0]*cap_box[1]))   # Guthrie geometric-mean capacity
mult = np.logspace(0, 5, 200)                         # local-peak multiplier over whole-gut average
fr = np.array([frac_capacity_limited(CAP_CENTRAL, D_AVG*m) for m in mult])
ax2.plot(mult, fr, lw=2, color="tab:purple")
for m_lab, lab, c in [(1,"whole-gut avg","tab:green"), (10,"plausible local","tab:orange"),
                      (100,"high local","tab:orange")]:
    ax2.axvline(m_lab, color=c, ls=":", lw=1.2); ax2.text(m_lab, 102, lab, rotation=90, va="bottom", ha="right", fontsize=8, color=c)
ax2.set_xscale("log"); ax2.set_xlabel("local peak delivery / whole-gut average  (multiplier)")
ax2.set_ylabel("% communities capacity-limited (composition matters)")
# title removed -- "(b) local-peak sensitivity, central Guthrie capacity" is in the caption
ax2.set_ylim(-2, 105)
plt.tight_layout(); plt.savefig("data/processed/figures/results_phase_diagram.png", dpi=200); plt.close()

# threshold multipliers for X% capacity-limited
print("\nLOCAL-PEAK sensitivity (central Guthrie capacity, avg delivery 1e-5 mmol/gDW/h):")
for X in [5, 10, 25, 50]:
    idx = np.argmax(fr >= X)
    m = mult[idx] if fr.max() >= X else np.inf
    print(f"  {X:2d}% of communities capacity-limited needs local peak ~{m:,.0f}x the whole-gut average")

# headline numbers at the plausible box corners + centre
print("real-data relative capacity: median(nonzero)=%.1f, zero-carrier=%d/%d" % (rel_med, (rel==0).sum(), len(rel)))
print("\n%% CAPACITY-LIMITED (composition matters) across the plausible box:")
for cm,lbl in [(cap_box[0],"low-cap (Guthrie low)"),(np.sqrt(cap_box[0]*cap_box[1]),"central"),(cap_box[1],"high-cap")]:
    row=[f"{frac_capacity_limited(cm,d):5.1f}%" for d in [D_box[0], np.sqrt(D_box[0]*D_box[1]), D_box[1]]]
    print(f"  cap_med={cm:6.2f} ({lbl:22}): D=1e-5 {row[0]} | D=3e-5 {row[1]} | D=1e-4 {row[2]}")
print("\n(>~0% capacity-limited across the plausible box => host-gated; only the zero/near-zero-GUS")
print(" tail is capacity-limited. Composition would control only if D rose ~3 orders or capacity fell ~3 orders.)")
print("saved -> data/processed/figures/results_phase_diagram.png")
