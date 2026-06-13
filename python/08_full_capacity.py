"""Full-cohort SN-38 reactivation capacity (Arm A only, fast LP) -- PARALLEL + checkpointed.

A == B (verified across all 6 cohorts, 09), so capacity = unconstrained max EX_sn38_m, a single
LP per sample. Embarrassingly parallel across models. Workers load the medium ONCE; the main
process writes each result as it arrives (no concurrent-write race), so the run stays resumable:
rerun to continue, skipping samples already in the checkpoint.
"""

import os
import glob
import gc
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
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
N_WORKERS      = int(os.environ.get("CAP_WORKERS", "4"))   # override via $env:CAP_WORKERS

_MED = None   # per-worker cached medium


def _init(medium_path):
    """Worker initializer: load the medium once per process."""
    global _MED
    m = load_qiime_medium(medium_path)
    _MED = dict(zip(m["reaction"], m["flux"]))
    _MED[SN38G_EXCHANGE] = UNLIMITED


def _score(path):
    """Arm A: max EX_sn38_m (LP), anaerobic, SN-38G supplied. Returns (sample, flux, contribs, err)."""
    sample = os.path.basename(path).replace(".pickle", "")
    try:
        com = load_pickle(path)
        # No SN-38 community exchange => the community has NO beta-glucuronidase carriers
        # => reactivation capacity is genuinely zero (a real low-risk data point, not an error).
        if SN38_EXCHANGE not in com.reactions:
            return sample, 0.0, {}, None
        have = {r.id for r in com.exchanges}
        com.medium = {r: f for r, f in _MED.items() if r in have}
        com.variables.community_objective.lb = 0.0          # unconstrained
        com.objective = com.reactions.get_by_id(SN38_EXCHANGE)
        s = com.optimize()
        flux = float(s.objective_value) if s is not None else 0.0
        contribs = {}
        for r in com.reactions:
            if "SN38G_GLCAASE" in r.id and abs(r.flux) > 1e-9:
                t = r.id.split("__")[-1]
                contribs[t] = contribs.get(t, 0.0) + abs(r.flux)
        return sample, flux, contribs, None
    except Exception as e:
        return sample, None, None, repr(e)[:150]


def done_samples():
    if os.path.exists(CKPT_CSV):
        return set(pd.read_csv(CKPT_CSV)["sample_id"])
    return set()


if __name__ == "__main__":
    os.makedirs(FLUX_DIR, exist_ok=True)
    models = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pickle")))
    already = done_samples()
    todo = [m for m in models if os.path.basename(m).replace(".pickle", "") not in already]
    print(f"{len(models)} models total | {len(already)} done | {len(todo)} to process "
          f"| {N_WORKERS} workers", flush=True)

    # Process in BATCHES, each with a FRESH ProcessPoolExecutor. Closing the pool after each batch
    # terminates the workers and frees their accumulated memory -- this bounds RAM WITHOUT using
    # max_tasks_per_child, which deadlocks on Windows when all workers recycle simultaneously.
    BATCH = int(os.environ.get("CAP_BATCH", "60"))
    n = 0
    for bstart in range(0, len(todo), BATCH):
        batch = todo[bstart:bstart + BATCH]
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init,
                                 initargs=(MEDIUM_PATH,)) as ex:
            futures = {ex.submit(_score, p): p for p in batch}
            for fut in as_completed(futures):
                sample, flux, contribs, err = fut.result()
                n += 1
                if err is not None:
                    print(f"  [{n}/{len(todo)}] {sample}: ERROR {err}", flush=True)
                    continue
                pd.DataFrame([{"sample_id": sample, "sn38_capacity": flux,
                               "n_gus_taxa": len(contribs)}]).to_csv(
                    CKPT_CSV, mode="a", header=not os.path.exists(CKPT_CSV), index=False)
                if contribs:
                    pd.DataFrame([{"sample_id": sample, "taxon": t, "gus_flux": f}
                                  for t, f in contribs.items()]).to_csv(
                        CONTRIB_CSV, mode="a", header=not os.path.exists(CONTRIB_CSV), index=False)
                if n % 10 == 0 or n == len(todo):
                    print(f"  [{n}/{len(todo)}] {sample}: capacity={flux:.2f}, GUS taxa={len(contribs)}",
                          flush=True)
        # pool closed -> workers exited -> their memory is released before the next batch
        gc.collect()

    cap = pd.read_csv(CKPT_CSV).drop_duplicates("sample_id")
    cap.to_parquet(os.path.join(FLUX_DIR, "full_capacity.parquet"))
    # also refresh the taxa-contributions parquet from its checkpoint (10_results reads the parquet)
    if os.path.exists(CONTRIB_CSV):
        con = pd.read_csv(CONTRIB_CSV).drop_duplicates(["sample_id", "taxon"])
        con.to_parquet(os.path.join(FLUX_DIR, "full_taxa_contributions.parquet"))
        print(f"Refreshed full_taxa_contributions.parquet ({len(con)} rows)", flush=True)
    print(f"\nDone. {len(cap)} samples -> {os.path.join(FLUX_DIR, 'full_capacity.parquet')}", flush=True)
    print(f"Median capacity: {cap['sn38_capacity'].median():.2f}", flush=True)
