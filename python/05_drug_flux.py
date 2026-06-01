"""Compute per-sample SN-38 reactivation CAPACITY from community models.

Two arms (see CLAUDE.md "Comparison with Heinken"):
  Arm A (Heinken replication): pure FBA max secretion of EX_sn38_m, with
    precursor SN-38G AND oxygen supplied unlimited (1000), NO growth constraint.
    Purpose: benchmark/validation against Heinken et al. 2023.
  Arm B (our contribution): max EX_sn38_m subject to the community holding its
    cooperative-tradeoff growth solution, under realistic ANAEROBIC western-diet
    conditions (no O2 flood). Purpose: reactivation in a functioning community.

Key finding motivating this design: plain grow() returns zero SN-38 flux because
reactivation is not growth-coupled -- the solver has no incentive to route flux
through beta-glucuronidase. Capacity must be optimized for explicitly.
"""

import os
import glob
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR  = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
FLUX_DIR    = "data/processed/flux"

SN38_EXCHANGE   = "EX_sn38_m"    # secreted toxic SN-38 (the readout)
SN38G_EXCHANGE  = "EX_sn38g_m"   # SN-38G precursor supply
O2_EXCHANGE     = "EX_o2_m"
UNLIMITED       = 1000.0
TRADEOFF        = 0.5
GROWTH_FRACTION = 0.95           # Arm B: hold community growth >= 95% of tradeoff max

# Test scope (match 03/04). Set to None for the full run.
TEST_N_MODELS = None


def _apply_medium(com, base_medium: dict, anaerobic: bool):
    """Set the community medium from a {reaction: flux} dict, keeping only
    exchanges that exist in this model. Supplies SN-38G; floods O2 unless anaerobic."""
    med = dict(base_medium)
    med[SN38G_EXCHANGE] = UNLIMITED
    if not anaerobic:
        med[O2_EXCHANGE] = UNLIMITED
    have = {r.id for r in com.exchanges}
    com.medium = {r: f for r, f in med.items() if r in have}


def arm_b_growth_constrained(com, base_medium) -> float:
    """Max SN-38 secretion while community holds its cooperative-tradeoff growth.
    NOTE: cobra's context manager does NOT roll back optlang variable-bound changes,
    so community_objective.lb is set/reset EXPLICITLY (verified 2026-06-01)."""
    with com:
        _apply_medium(com, base_medium, anaerobic=True)
        sol = com.cooperative_tradeoff(fraction=TRADEOFF)
        gr = sol.growth_rate
        com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
        com.variables.community_objective.lb = GROWTH_FRACTION * gr
        s = com.optimize()
        flux = float(s.objective_value) if s is not None else 0.0
        contribs = _gus_contributions(com)
        com.variables.community_objective.lb = 0.0   # explicit reset (not auto-rolled back)
    return gr, flux, contribs


def arm_a_heinken(com, base_medium) -> float:
    """Pure FBA max SN-38 secretion, precursor + O2 unlimited, NO growth constraint."""
    with com:
        _apply_medium(com, base_medium, anaerobic=False)
        com.variables.community_objective.lb = 0.0   # explicit: ensure truly unconstrained
        com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
        s = com.optimize()
        flux = float(s.objective_value) if s is not None else 0.0
        com.variables.community_objective.lb = 0.0
        return flux


def _gus_contributions(com) -> dict:
    """Per-taxon flux through SN38G_GLCAASE* reactions in the current solution."""
    out = {}
    for r in com.reactions:
        if "SN38G_GLCAASE" in r.id:
            taxon = r.id.split("__")[-1]
            out[taxon] = out.get(taxon, 0.0) + abs(r.flux)
    return {k: v for k, v in out.items() if v > 1e-9}


if __name__ == "__main__":
    os.makedirs(FLUX_DIR, exist_ok=True)
    base_medium = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))

    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pickle")))
    if TEST_N_MODELS:
        model_files = model_files[:TEST_N_MODELS]
    print(f"Computing SN-38 capacity for {len(model_files)} samples (Arms A + B)")

    cap_rows, contrib_rows = [], []
    for i, path in enumerate(model_files, 1):
        sample = os.path.basename(path).replace(".pickle", "")
        com = load_pickle(path)
        gr, flux_b, contribs = arm_b_growth_constrained(com, base_medium)
        flux_a = arm_a_heinken(com, base_medium)
        cap_rows.append({
            "sample_id": sample,
            "growth_rate": gr,
            "sn38_capacity_constrained": flux_b,   # Arm B (primary)
            "sn38_capacity_unconstrained": flux_a, # Arm A (Heinken)
        })
        for taxon, f in contribs.items():
            contrib_rows.append({"sample_id": sample, "taxon": taxon, "gus_flux": f})
        print(f"  [{i}/{len(model_files)}] {sample}: "
              f"B={flux_b:.3f}  A={flux_a:.3f}  growth={gr:.4f}  GUS taxa={len(contribs)}")

    cap = pd.DataFrame(cap_rows)
    contrib = pd.DataFrame(contrib_rows)
    cap.to_parquet(os.path.join(FLUX_DIR, "sn38_capacity.parquet"))
    contrib.to_parquet(os.path.join(FLUX_DIR, "sn38_taxa_contributions.parquet"))

    print("\n=== SN-38 reactivation capacity ===")
    print(f"Arm B (constrained)   median: {cap['sn38_capacity_constrained'].median():.3f}")
    print(f"Arm A (unconstrained) median: {cap['sn38_capacity_unconstrained'].median():.3f}")
    print(f"Samples with non-zero Arm B: {(cap['sn38_capacity_constrained'] > 0).sum()}/{len(cap)}")
    print(f"Results -> {FLUX_DIR}")
