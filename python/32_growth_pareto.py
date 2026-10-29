"""Growth vs SN-38 reactivation trade-off, and Arm A/B substrate uptake (mSystems R2 comment 3).

Reviewer 2 objected that Arm A (max EX_sn38_m) and Arm B (max EX_sn38_m s.t. >=95% biomass)
cannot give identical flux "unless the substrates available to the model are unconstrained",
asked us to show how substrate uptake compares between the arms, and suggested a dual objective
over EX_sn38_m and biomass.

WHAT THIS COMPUTES
  g(f) = max EX_sn38_m  s.t.  community growth >= f * gr_max,  for f over [0, 0.999].
  g is non-increasing in f (raising the floor only shrinks the feasible set), so if g(0) == g(1)
  the ENTIRE growth/reactivation Pareto front is flat -- a stronger statement than A == B at one
  arbitrary threshold, and the direct answer to the dual-objective suggestion.
  At the endpoints we also record a DETERMINISTIC uptake profile.

WHY NOT pfba / cooperative_tradeoff
  - MICOM's optimize(pfba=True) does not honour a custom objective: it optimises growth and then
    minimises total flux, collapsing EX_sn38_m to its parsimonious minimum (47.26 -> 3.28 on a test
    model). That is the FBA-parsimony artifact already documented and rejected in the Q2 analysis;
    it also makes sol.objective_value the total-flux sum, not the SN-38 flux.
  - cooperative_tradeoff is a QP costing ~400 s/model. The paper's Arm A vs Arm B comparison is
    already banked in data/processed/flux/AeqB_validation.parquet (45/45 exact), so this script
    sweeps against gr_max with plain LPs instead.

DETERMINISTIC UPTAKE: the Pareto LP fixes only EX_sn38_m, leaving other fluxes degenerate. At each
endpoint we therefore pin EX_sn38_m to its optimum and run a second LP minimising total uptake
(sum of community-exchange reverse variables), giving a reproducible minimal nutrient draw.

Run:
  python python/32_growth_pareto.py           # stratified 45 (5/cohort, matches Tables S1/S2)
  PARETO_TEST=<sample_id> python ...          # single-model check
Resumable via checkpoint.
"""

import os
import glob
import gc
import pandas as pd
from optlang.symbolics import Zero
from concurrent.futures import ProcessPoolExecutor, as_completed
from micom.util import load_pickle
from micom.qiime_formats import load_qiime_medium

MODELS_DIR = "data/processed/models"
MEDIUM_PATH = "data/media/western_diet_gut_agora.qza"
FLUX_DIR = "data/processed/flux"
CKPT_CSV = os.path.join(FLUX_DIR, "growth_pareto_checkpoint.csv")
OUT_PARQUET = os.path.join(FLUX_DIR, "growth_pareto.parquet")

SN38_EXCHANGE = "EX_sn38_m"
SN38G_EXCHANGE = "EX_sn38g_m"
UNLIMITED = 1000.0
# 0.999 rather than 1.0: at an exactly-binding floor the LP can be reported infeasible on
# numerical tolerance. 0.999*gr_max is the same claim without the knife edge.
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
UPTAKE_AT = [0.0, 0.999]          # endpoints where the deterministic uptake LP is also run
N_WORKERS = int(os.environ.get("CAP_WORKERS", "4"))
BATCH = int(os.environ.get("CAP_BATCH", "20"))
N_PER_COHORT = int(os.environ.get("PARETO_N_PER_COHORT", "5"))

PRIMARY = ["ZellerG_2014", "YuJ_2015", "FengQ_2015", "ThomasAM_2018a", "ThomasAM_2018b",
           "WirbelJ_2018", "VogtmannE_2016", "YachidaS_2019", "ThomasAM_2019_c"]

_MED = None


def _init(medium_path):
    global _MED
    m = load_qiime_medium(medium_path)
    _MED = dict(zip(m["reaction"], m["flux"]))
    _MED[SN38G_EXCHANGE] = UNLIMITED


def _uptake_profile(com, comm_ex, sn38_rx, sn38_value):
    """Pin EX_sn38_m at its optimum, then minimise total uptake for a reproducible profile."""
    lo, hi = sn38_rx.bounds
    # Assigning a new Objective sets the objective DIRECTION too, and a later
    # `com.objective = <reaction>` restores only the expression -- leaving direction="min"
    # so every subsequent solve silently MINIMISES SN-38. Capture and restore it here, and
    # set the direction explicitly at each Pareto solve as well.
    direction = com.objective_direction
    try:
        sn38_rx.bounds = (sn38_value - 1e-9, sn38_value + 1e-9)
        com.objective = com.problem.Objective(
            Zero + sum(r.reverse_variable for r in comm_ex), direction="min")
        s = com.optimize()
        if s is None:
            return {}
        total_uptake = sum(-r.flux for r in comm_ex if r.flux < 0)
        return {"total_uptake": float(total_uptake),
                "n_uptake_rx": int(sum(1 for r in comm_ex if r.flux < -1e-9)),
                "sn38g_uptake": float(-com.reactions.get_by_id(SN38G_EXCHANGE).flux)}
    except Exception:
        return {}
    finally:
        sn38_rx.bounds = (lo, hi)
        com.objective_direction = direction


def _score(path):
    sample = os.path.basename(path).replace(".pickle", "")
    try:
        com = load_pickle(path)
        if SN38_EXCHANGE not in com.reactions:
            return sample, [{"sample_id": sample, "fraction": f, "sn38": 0.0,
                             "gr_max": 0.0, "growth": 0.0} for f in FRACTIONS], None
        have = {r.id for r in com.exchanges}
        com.medium = {r: f for r, f in _MED.items() if r in have}
        comm_ex = [r for r in com.exchanges if r.id.endswith("_m")]
        sn38_rx = com.reactions.get_by_id(SN38_EXCHANGE)
        co = com.variables.community_objective

        # maximum community growth (the denominator for every floor)
        co.lb = 0.0
        com.objective = co
        com.objective_direction = "max"
        s = com.optimize()
        gr_max = float(s.growth_rate) if s is not None else 0.0

        rows = []
        for f in FRACTIONS:
            co.lb = f * gr_max
            com.objective = sn38_rx
            com.objective_direction = "max"     # never rely on the inherited direction
            s = com.optimize()
            sn38 = float(s.objective_value) if s is not None else float("nan")
            growth = float(s.growth_rate) if s is not None else float("nan")
            row = {"sample_id": sample, "fraction": f, "sn38": sn38,
                   "gr_max": gr_max, "growth": growth}
            if f in UPTAKE_AT and s is not None:
                row.update(_uptake_profile(com, comm_ex, sn38_rx, sn38))
            rows.append(row)
        co.lb = 0.0
        return sample, rows, None
    except Exception as e:
        return sample, None, repr(e)[:200]


def _meta():
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    return tax[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")


def pick_samples():
    built = {os.path.basename(p).replace(".pickle", "")
             for p in glob.glob(os.path.join(MODELS_DIR, "*.pickle"))}
    one = os.environ.get("PARETO_TEST")
    if one:
        return [one]
    meta = _meta()
    meta = meta[meta.cohort.isin(PRIMARY) & meta.sample_id.isin(built)]
    return (meta.sort_values("sample_id")
            .groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist())


if __name__ == "__main__":
    os.makedirs(FLUX_DIR, exist_ok=True)
    samples = pick_samples()
    done = set(pd.read_csv(CKPT_CSV)["sample_id"]) if os.path.exists(CKPT_CSV) else set()
    todo = [s for s in samples if s not in done]
    print(f"Growth/SN-38 Pareto | {len(FRACTIONS)} floors | {len(samples)} samples "
          f"| {len(done)} done | {len(todo)} to do | {N_WORKERS} workers", flush=True)

    n = 0
    for bstart in range(0, len(todo), BATCH):
        batch = [os.path.join(MODELS_DIR, s + ".pickle") for s in todo[bstart:bstart + BATCH]]
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init,
                                 initargs=(MEDIUM_PATH,)) as ex:
            futures = {ex.submit(_score, p): p for p in batch}
            for fut in as_completed(futures):
                sample, rows, err = fut.result()
                n += 1
                if err is not None:
                    print(f"  [{n}/{len(todo)}] {sample}: ERROR {err}", flush=True)
                    continue
                pd.DataFrame(rows).to_csv(CKPT_CSV, mode="a",
                                          header=not os.path.exists(CKPT_CSV), index=False)
                d = pd.DataFrame(rows)
                lo = d.loc[d.fraction == 0.0, "sn38"].iloc[0]
                hi = d.loc[d.fraction == max(FRACTIONS), "sn38"].iloc[0]
                print(f"  [{n}/{len(todo)}] {sample}: sn38(f=0)={lo:.4f} "
                      f"sn38(f=.999)={hi:.4f} drop={lo - hi:.2e}", flush=True)
        gc.collect()

    df = pd.read_csv(CKPT_CSV).drop_duplicates(["sample_id", "fraction"])
    df = df.merge(_meta(), on="sample_id", how="left")
    df.to_parquet(OUT_PARQUET)

    wide = df.pivot(index="sample_id", columns="fraction", values="sn38")
    fmax = max(FRACTIONS)
    drop = (wide[0.0] - wide[fmax]).abs()
    rel = (drop / wide[0.0].where(wide[0.0] > 1e-9)).abs()
    print(f"\n=== Growth/SN-38 Pareto front: {len(wide)} samples ===")
    print("\nMedian SN-38 capacity at each growth floor:")
    for f in FRACTIONS:
        print(f"  f={f:<6} median={wide[f].median():9.4f}")
    print(f"\n|sn38(f=0) - sn38(f={fmax})|: max abs={drop.max():.3e}  max rel={rel.max():.3e}")
    print(f"{int((drop < 1e-6).sum())}/{len(wide)} samples FLAT across the whole front (tol 1e-6)")
    if "total_uptake" in df.columns:
        up = df[df.fraction.isin(UPTAKE_AT)].pivot(index="sample_id", columns="fraction",
                                                   values="total_uptake")
        if up.notna().all(axis=None):
            print("\nTotal nutrient uptake (deterministic min-uptake LP):")
            print(f"  median at f=0     : {up[0.0].median():.3f}")
            print(f"  median at f={fmax}: {up[fmax].median():.3f}")
            ratio = (up[fmax] / up[0.0].where(up[0.0] > 1e-9)).median()
            print(f"  median ratio f={fmax} / f=0 : {ratio:.3f}")
    print(f"Saved -> {OUT_PARQUET}")
