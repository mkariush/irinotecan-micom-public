"""Tier-1 absolute-units placement: physiological SN-38G delivery vs reactivation capacity.

Converts relative capacity (mmol/gDW_community/h) to mmol/person/day, overlays the physiological
biliary delivery band, and quantifies the delivery-limited fraction + protected tail. See
docs/absolute_units_plan.md. ALL magnitude parameters are [VERIFY] placeholders, swept for robustness.

NOTE: the qualitative host-gating conclusion rests on the MODEL-INDEPENDENT Slatter anchor (gut
deconjugates ~all delivered SN-38G in vivo); this script quantifies the margin, it does not prove it.
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLUX, FIG = "data/processed/flux", "data/processed/figures"
PRIMARY = ["ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a","ThomasAM_2018b",
           "WirbelJ_2018","VogtmannE_2016","YachidaS_2019","ThomasAM_2019_c"]

# ---- [VERIFY] magnitude parameters (see docs/absolute_units_plan.md) ----
BIOMASS_gDW   = 60.0            # gut bacterial dry weight per person [VERIFY Heinken refs 10/37 / Sender 2016]
BIOMASS_RANGE = (20.0, 100.0)   # plausible range for the sensitivity sweep
HOURS_PER_DAY = 24.0
# physiological biliary SN-38G delivery, mmol/person/day [VERIFY Slatter/Sun/Guan + dose]
DELIVERY      = 0.05
DELIVERY_RANGE = (0.01, 0.15)

def to_abs(cap_rel, biomass):
    """relative capacity (mmol/gDW/h) -> mmol/person/day."""
    return cap_rel * biomass * HOURS_PER_DAY

def main():
    cap = pd.read_parquet(f"{FLUX}/full_capacity.parquet")
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    coh = tax[["sample_id","cohort"]].drop_duplicates("sample_id")
    cap = cap.merge(coh, on="sample_id", how="left")
    cap = cap[cap.cohort.isin(PRIMARY)].copy()
    cap["cap_abs"] = to_abs(cap.sn38_capacity, BIOMASS_gDW)

    # point estimate
    dl = (cap.cap_abs > DELIVERY).mean() * 100        # delivery-limited fraction
    tail = (cap.cap_abs <= DELIVERY).mean() * 100     # protected (capacity-limited) tail
    ratio = (cap.cap_abs / DELIVERY).replace([np.inf,-np.inf], np.nan)
    print(f"BIOMASS={BIOMASS_gDW} gDW, DELIVERY={DELIVERY} mmol/d")
    print(f"capacity_abs (mmol/person/day): median={cap.cap_abs.median():.1f}, "
          f"range {cap.cap_abs.min():.2f}-{cap.cap_abs.max():.0f}")
    print(f"delivery-limited: {dl:.1f}%  |  protected tail (cap<delivery): {tail:.1f}%")
    print(f"capacity:delivery ratio median = {ratio.median():.0f}x")

    # robustness: worst-case delivery-limited fraction across the biomass x delivery box
    print("\nsensitivity (delivery-limited %) across BIOMASS x DELIVERY box:")
    grid = []
    for b in np.linspace(*BIOMASS_RANGE, 4):
        row = []
        for d in np.linspace(*DELIVERY_RANGE, 4):
            row.append(round((to_abs(cap.sn38_capacity, b) > d).mean()*100, 1))
        grid.append([round(b,0)] + row)
    cols = ["biomass\\delivery"] + [f"{d:.3f}" for d in np.linspace(*DELIVERY_RANGE, 4)]
    print(pd.DataFrame(grid, columns=cols).to_string(index=False))
    worst = min(min(r[1:]) for r in grid)
    print(f"WORST-CASE delivery-limited fraction in the box: {worst:.1f}%")

    # figure: capacity_abs distribution + delivery band
    plt.figure(figsize=(9,5))
    plt.hist(np.log10(cap.cap_abs.replace(0, np.nan).dropna()), bins=40, color="0.7", edgecolor="w")
    plt.axvspan(np.log10(DELIVERY_RANGE[0]), np.log10(DELIVERY_RANGE[1]), color="tab:red", alpha=0.25,
                label="physiological delivery band")
    plt.axvline(np.log10(DELIVERY), color="tab:red", lw=1.5)
    plt.xlabel("log10  mmol per person per day"); plt.ylabel("communities")
    plt.title(f"Capacity vs physiological delivery (n={len(cap)})\n"
              f"{dl:.0f}% delivery-limited; protected tail {tail:.1f}%")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/results_R7_absolute_units.png", dpi=200); plt.close()
    print(f"\nsaved -> {FIG}/results_R7_absolute_units.png")
    print("REMINDER: fill BIOMASS_gDW + DELIVERY from literature before quoting numbers.")

if __name__ == "__main__":
    main()
