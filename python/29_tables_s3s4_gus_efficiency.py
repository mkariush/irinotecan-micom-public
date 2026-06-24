"""Supplementary Tables S3/S4: GUS loop-class efficiency scheme + per-taxon assignments.

Generated DIRECTLY from the authoritative weighting dicts in gus_efficiency.py
(CLASS_EFF, SPECIES_CLASS, GENUS_DEFAULT) so the supplement can never drift from the code that
produces Fig 4C. The empirical anchors (Pellock 2018 SN-38G kcat/Km) are documented in the
'Empirical basis' column. Idempotent: re-running replaces any existing S3/S4 block.

Build order for docs/supplementary_tables.md:  26 (S1, overwrites) -> 28 (S2) -> 29 (S3, S4).

    python python/29_tables_s3s4_gus_efficiency.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from gus_efficiency import CLASS_EFF, SPECIES_CLASS, GENUS_DEFAULT  # single source of truth

OUT_DIR   = "data/processed/tables"
OUT_CSV3  = f"{OUT_DIR}/table_S3_class_efficiency.csv"
OUT_CSV4  = f"{OUT_DIR}/table_S4_taxon_efficiency.csv"
OUT_MD    = "docs/supplementary_tables.md"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAME = {"L1": "Loop-1", "NL": "No-Loop", "mL1": "mini-Loop-1",
              "L2": "Loop-2", "mL2": "mini-Loop-2", "NC": "No-Coverage"}
CLASS_ORDER = ["L1", "NL", "L2", "mL1", "mL2", "NC"]
CLASS_BASIS = {
    "L1":  "Loop-1 reference; *E. coli* GUS SN-38G kcat/Km ~1.2x10^6 s^-1 M^-1 (Pellock 2018, Fig 8B)",
    "NL":  "**Measured**: *B. uniformis* BuGUS-1 rivals L1 (~1.2x10^6 s^-1 M^-1) (Pellock 2018)",
    "L2":  "**Measured**: *B. uniformis* BuGUS-2 ~4x10^5 s^-1 M^-1, ~1/3 of L1 (Pellock 2018)",
    "mL1": "**Interpolated**: no characterized SN-38G representative; set equal to L2",
    "mL2": "**Measured ~0**: *B. uniformis* BuGUS-3 no detectable SN-38G activity; small non-zero (Pellock 2018)",
    "NC":  "**Convention**: no structural coverage / unannotated; conservative exclusion (not a measured efficiency)",
}

# ---- Table S3: class scheme ----
s3 = pd.DataFrame([{
    "Loop class": f"{CLASS_NAME[c]} ({c})",
    "Relative efficiency": CLASS_EFF[c],
    "Empirical basis": CLASS_BASIS[c],
} for c in CLASS_ORDER])
s3.to_csv(OUT_CSV3, index=False)

# ---- Table S4: per-taxon assignments (species-level, then genus defaults) ----
rows = []
for sp, cls in sorted(SPECIES_CLASS.items()):
    rows.append({"Taxon": "*" + sp.replace("_", " ") + "*", "Assignment level": "species",
                 "Class": cls, "Efficiency": CLASS_EFF[cls],
                 "Basis": "individually characterized loop class (Pellock 2018; Biernat 2019)"})
for genus, val in sorted(GENUS_DEFAULT.items()):
    rows.append({"Taxon": "*" + genus + "* spp.", "Assignment level": "genus default",
                 "Class": "—", "Efficiency": val,
                 "Basis": "genus-typical estimate (fallback for carriers lacking a species-level class)"})
rows.append({"Taxon": "any other genus", "Assignment level": "global default",
             "Class": "—", "Efficiency": 0.5, "Basis": "neutral fallback"})
s4 = pd.DataFrame(rows)
s4.to_csv(OUT_CSV4, index=False)

# ---- markdown ----
s3_hdr = (
    "## Table S3. Gut-microbial β-glucuronidase loop-class efficiency coefficients\n\n"
    "Relative SN-38-glucuronide processing efficiencies used to re-weight each carrier's contribution "
    "in the enzyme-class sensitivity analysis (Fig 4C; Methods). Efficiencies are expressed relative to "
    "the Loop-1 reference (= 1.0). No-Loop (NL), Loop-2 (L2) and mini-Loop-2 (mL2) are anchored to the "
    "measured SN-38-glucuronide catalytic efficiencies of the three *B. uniformis* enzymes "
    "(BuGUS-1/-2/-3; Pellock et al. 2018, Fig 8B); mini-Loop-1 (mL1) is interpolated and No-Coverage (NC) "
    "is a conservative exclusion. This weighting is reported only as a sensitivity analysis; all primary "
    "results use the uniform-efficiency scheme.\n\n"
)
s4_hdr = (
    "## Table S4. Per-taxon β-glucuronidase efficiency assignments\n\n"
    "Efficiency assigned to each carrier taxon. Species with an individually characterized "
    "β-glucuronidase loop class receive that class's coefficient (Table S3); carriers lacking a "
    "species-level class fall back to a genus-typical default estimate, or 0.5 for genera not listed. "
    "Where a species encodes several β-glucuronidases of different classes, it is assigned the "
    "efficiency of its highest-efficiency (best) enzyme.\n\n"
)
block = (s3_hdr + s3.to_markdown(index=False) + "\n\n" + s4_hdr + s4.to_markdown(index=False) + "\n")

prev = ""
if os.path.exists(OUT_MD):
    prev = open(OUT_MD, encoding="utf-8").read()
# idempotent: drop any existing S3 block onward, then re-append
marker = "## Table S3."
if marker in prev:
    prev = prev[:prev.index(marker)]
prev = prev.rstrip() + "\n\n"

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(prev + block)

print(f"S3: {len(s3)} classes | S4: {len(s4)} taxon rows "
      f"({len(SPECIES_CLASS)} species + {len(GENUS_DEFAULT)} genus defaults + 1 global)")
print(f"saved -> {OUT_CSV3}, {OUT_CSV4}  and appended S3/S4 to {OUT_MD}")
