"""Full-cohort SN-38 reactivation capacity (Arm A only, fast LP) with checkpointing.

Rationale: A == B (growth/oxygen non-binding), so realized capacity equals the unconstrained
maximum, which is a single LP per sample -- fast on hybrid/HiGHS, no QP cooperative_tradeoff
needed. This makes the full ~813-sample run tractable on the open-source solver; Gurobi is
NOT required (build, not solving, is the wall-time bottleneck).

Checkpointing: appends each sample's result to a CSV as it finishes and skips samples already
present on restart, so a multi-hour run survives interruption (rerun to resume).
"""

import os
import glob
import pandas as pd
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR  = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
FLUX_DIR    = "data/processed/flux"
CKPT_CSV    = os.path.join(FLUX_DIR, "full_capacity_checkpoint.csv")
CONTRIB_CSV = os.path.join(FLUX_DIR, "full_taxa_contributions_checkpoint.csv")

SN38_EXCHANGE  = "EX_sn38_m"
SN38G_EXCHANGE = "EX_sn38g_m"
UNLIMITED      = 1000.0


def capacity_and_contribs(com):
    """Arm A: max SN-38 secretion (LP), anaerobic, SN-38G supplied. Returns (flux, {taxon: gus})."""
    med = dict(zip(*[load_qiime_medium(MEDIUM_PATH)[c] for c in ("reaction", "flux")]))
    med[SN38G_EXCHANGE] = UNLIMITED
    have = {r.id for r in com.exchanges}
    com.medium = {r: f for r, f in med.items() if r in have}
    com.variables.community_objective.lb = 0.0          # explicit: unconstrained
    com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
    s = com.optimize()
    flux = float(s.objective_value) if s is not None else 0.0
    contribs = {}
    for r in com.reactions:
        if "SN38G_GLCAASE" in r.id and abs(r.flux) > 1e-9:
            contribs[r.id.split("__")[-1]] = contribs.get(r.id.split("__")[-1], 0.0) + abs(r.flux)
    return flux, contribs


def done_samples() -> set:
    if os.path.exists(CKPT_CSV):
        return set(pd.read_csv(CKPT_CSV)["sample_id"])
    return set()


if __name__ == "__main__":
    os.makedirs(FLUX_DIR, exist_ok=True)
    models = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pickle")))
    already = done_samples()
    todo = [m for m in models if os.path.basename(m).replace(".pickle", "") not in already]
    print(f"{len(models)} models total | {len(already)} done | {len(todo)} to process")

    for i, path in enumerate(todo, 1):
        sample = os.path.basename(path).replace(".pickle", "")
        try:
            com = load_pickle(path)
            flux, contribs = capacity_and_contribs(com)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {sample}: ERROR {repr(e)[:120]}")
            continue
        # append capacity row
        pd.DataFrame([{"sample_id": sample, "sn38_capacity": flux,
                       "n_gus_taxa": len(contribs)}]).to_csv(
            CKPT_CSV, mode="a", header=not os.path.exists(CKPT_CSV), index=False)
        # append contributions
        if contribs:
            pd.DataFrame([{"sample_id": sample, "taxon": t, "gus_flux": f}
                          for t, f in contribs.items()]).to_csv(
                CONTRIB_CSV, mode="a", header=not os.path.exists(CONTRIB_CSV), index=False)
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {sample}: capacity={flux:.2f}, GUS taxa={len(contribs)}")

    # consolidate to parquet at the end
    cap = pd.read_csv(CKPT_CSV).drop_duplicates("sample_id")
    cap.to_parquet(os.path.join(FLUX_DIR, "full_capacity.parquet"))
    print(f"\nDone. {len(cap)} samples -> {os.path.join(FLUX_DIR, 'full_capacity.parquet')}")
    print(f"Median capacity: {cap['sn38_capacity'].median():.2f}")
