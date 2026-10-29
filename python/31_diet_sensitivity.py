"""Diet sensitivity of SN-38 reactivation capacity (mSystems R2 comment 2).

Reviewer 2 asked whether the single Western-diet medium drives the capacity result, noting
that GEMs are sensitive to uptake constraining and that the cohorts span geographies.

STRUCTURAL EXPECTATION (why this should be invariant): Arm A imposes NO growth requirement
(community_objective.lb = 0) and supplies SN-38G at UNLIMITED independently of the diet file.
The hydrolysis SN-38G + H2O -> SN-38 + glucuronate needs only SN-38G and water; H2O is present
in every medium and glucuronate leaves by secretion (no medium supplies EX_glcur_m, so its
disposal is not gated by an uptake bound). Diet metabolites therefore enter only via growth,
which is unconstrained here and non-binding in Arm B (the A==B result, 09/26).

The residual risk this script actually tests: `com.medium = {...}` closes uptake on everything
absent from the medium, so a restrictive diet could in principle starve a carrier of something
needed to sustain flux. High-fat/low-carb is included as the stress case.

Reuses the Arm A logic of 08_full_capacity.py verbatim so numbers are directly comparable.
Each model is loaded ONCE and solved under every medium (1 load, 5 LPs) -- loading dominates.

Run:
  python python/31_diet_sensitivity.py            # stratified pilot (N_PER_COHORT per cohort)
  DIET_ALL=1 python python/31_diet_sensitivity.py # all built models
Resumable: rerun to continue from the checkpoint.
"""

import os
import glob
import gc
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR = "data/processed/models"
MEDIA_DIR = "data/media"
FLUX_DIR = "data/processed/flux"
CKPT_CSV = os.path.join(FLUX_DIR, "diet_sensitivity_checkpoint.csv")
OUT_PARQUET = os.path.join(FLUX_DIR, "diet_sensitivity.parquet")

# label -> medium file. "western" is the manuscript baseline (must reproduce 08 exactly).
MEDIA = {
    "western": "western_diet_gut_agora.qza",
    "eu_average": "vmh_eu_average_agora.qza",
    "high_fiber": "vmh_high_fiber_agora.qza",
    "hf_lowcarb": "vmh_high_fat_low_carb_agora.qza",
    "himalaya": "himalaya.qza",
}

SN38_EXCHANGE = "EX_sn38_m"
SN38G_EXCHANGE = "EX_sn38g_m"
UNLIMITED = 1000.0
N_WORKERS = int(os.environ.get("CAP_WORKERS", "4"))
BATCH = int(os.environ.get("CAP_BATCH", "60"))
N_PER_COHORT = int(os.environ.get("DIET_N_PER_COHORT", "5"))

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

_MEDIA = None  # per-worker cache: {label: {reaction: flux}}


def _init(media_dir):
    """Worker initializer: load every medium once per process."""
    global _MEDIA
    _MEDIA = {}
    for label, fname in MEDIA.items():
        m = load_qiime_medium(os.path.join(media_dir, fname))
        d = dict(zip(m["reaction"], m["flux"]))
        d[SN38G_EXCHANGE] = UNLIMITED
        _MEDIA[label] = d


def _score(path):
    """Load the model once; max EX_sn38_m under each medium. Returns (sample, rows, err)."""
    sample = os.path.basename(path).replace(".pickle", "")
    try:
        com = load_pickle(path)
        # No SN-38 community exchange => no beta-glucuronidase carriers => capacity is
        # genuinely zero under every diet (a real low-risk data point, not an error).
        if SN38_EXCHANGE not in com.reactions:
            return sample, [{"sample_id": sample, "diet": lab, "sn38_capacity": 0.0,
                             "n_gus_taxa": 0} for lab in MEDIA], None
        have = {r.id for r in com.exchanges}
        obj = com.reactions.get_by_id(SN38_EXCHANGE)
        rows = []
        for label, med in _MEDIA.items():
            # assigning com.medium fully redefines uptake (absent exchanges are closed),
            # so no bounds leak between diets; lb is reset explicitly (cobra's context
            # manager does not roll back optlang variable bounds -- see CLAUDE.md).
            com.medium = {r: f for r, f in med.items() if r in have}
            com.variables.community_objective.lb = 0.0
            com.objective = obj
            s = com.optimize()
            flux = float(s.objective_value) if s is not None else 0.0
            n_gus = sum(1 for r in com.reactions
                        if "SN38G_GLCAASE" in r.id and abs(r.flux) > 1e-9)
            rows.append({"sample_id": sample, "diet": label,
                         "sn38_capacity": flux, "n_gus_taxa": n_gus})
        return sample, rows, None
    except Exception as e:
        return sample, None, repr(e)[:150]


def _meta():
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    return tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")


def pick_samples():
    """All built primary-cohort models, or a stratified pilot of N_PER_COHORT each."""
    built = {os.path.basename(p).replace(".pickle", "")
             for p in glob.glob(os.path.join(MODELS_DIR, "*.pickle"))}
    meta = _meta()
    meta = meta[meta.cohort.isin(PRIMARY) & meta.sample_id.isin(built)]
    if os.environ.get("DIET_ALL"):
        return sorted(meta["sample_id"].tolist())
    return (meta.sort_values("sample_id")
            .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())


if __name__ == "__main__":
    os.makedirs(FLUX_DIR, exist_ok=True)
    samples = pick_samples()
    done = set(pd.read_csv(CKPT_CSV)["sample_id"]) if os.path.exists(CKPT_CSV) else set()
    todo = [s for s in samples if s not in done]
    print(f"Diet sensitivity | {len(MEDIA)} media | {len(samples)} samples "
          f"| {len(done)} done | {len(todo)} to do | {N_WORKERS} workers", flush=True)

    n = 0
    for bstart in range(0, len(todo), BATCH):
        batch = [os.path.join(MODELS_DIR, s + ".pickle") for s in todo[bstart:bstart + BATCH]]
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init,
                                 initargs=(MEDIA_DIR,)) as ex:
            futures = {ex.submit(_score, p): p for p in batch}
            for fut in as_completed(futures):
                sample, rows, err = fut.result()
                n += 1
                if err is not None:
                    print(f"  [{n}/{len(todo)}] {sample}: ERROR {err}", flush=True)
                    continue
                pd.DataFrame(rows).to_csv(CKPT_CSV, mode="a",
                                          header=not os.path.exists(CKPT_CSV), index=False)
                if n % 10 == 0 or n == len(todo):
                    caps = {r["diet"]: r["sn38_capacity"] for r in rows}
                    spread = max(caps.values()) - min(caps.values())
                    print(f"  [{n}/{len(todo)}] {sample}: west={caps['western']:.2f} "
                          f"max-min across diets={spread:.2e}", flush=True)
        gc.collect()

    df = pd.read_csv(CKPT_CSV).drop_duplicates(["sample_id", "diet"])
    df = df.merge(_meta(), on="sample_id", how="left")
    df.to_parquet(OUT_PARQUET)

    wide = df.pivot(index="sample_id", columns="diet", values="sn38_capacity")
    wide = wide[[c for c in MEDIA if c in wide.columns]]
    dev = wide.sub(wide["western"], axis=0)
    reldev = dev.div(wide["western"].where(wide["western"] > 1e-9), axis=0).abs()

    print(f"\n=== Diet sensitivity: {len(wide)} samples x {wide.shape[1]} media ===")
    print("\nCapacity by diet (median / IQR):")
    for c in wide.columns:
        q1, q3 = wide[c].quantile([0.25, 0.75])
        print(f"  {c:12s} median={wide[c].median():8.3f}  IQR {q1:.3f}-{q3:.3f}")
    print("\nMax |deviation from western| per diet (absolute / relative):")
    for c in wide.columns:
        print(f"  {c:12s} abs={dev[c].abs().max():.3e}  rel={reldev[c].max():.3e}")
    n_identical = int((dev.abs().max(axis=1) < 1e-6).sum())
    print(f"\n{n_identical}/{len(wide)} samples identical across ALL media (tol 1e-6)")
    print(f"Saved -> {OUT_PARQUET}")
