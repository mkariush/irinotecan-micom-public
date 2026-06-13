"""Resume the model build ROBUSTLY by pre-filtering to UNBUILT samples only.

MICOM's build() submits EVERY sample in the taxonomy to its worker pool (the "skip" happens inside
each task), so resuming with 1187 existing models floods the pool with no-op skip-tasks and it
deadlocks on Windows. Here we pass ONLY the samples that lack a pickle -> the pool sees just real
build work (the condition under which it built 1187 fine). Existing pickles are never touched.

NOTE: the build MUST run under `if __name__ == "__main__":` -- on Windows, multiprocessing re-imports
this module in every worker, and an unguarded build() call recurses into infinite process spawning.
"""
import os, glob, sys
import pandas as pd
from micom.workflows import build

TAX     = "data/processed/taxonomy_micom.parquet"
DB      = r"databases\AGORA2_json"
OUT     = "data/processed/models"
CUTOFF  = 0.001
THREADS = int(os.environ.get("BUILD_THREADS", "4"))
SOLVER  = "hybrid"


def main():
    tax = pd.read_parquet(TAX)
    built = {os.path.basename(p)[:-len(".pickle")] for p in glob.glob(os.path.join(OUT, "*.pickle"))}
    todo = tax[~tax.sample_id.isin(built)].copy()

    mt = todo[["sample_id", "id", "abundance"]].copy()
    mt["genus"]   = mt["id"].str.split().str[0]
    mt["species"] = mt["id"].str.split().str[1]

    n_todo = mt.sample_id.nunique()
    print(f"already built: {len(built)} | UNBUILT to build now: {n_todo} | threads={THREADS}", flush=True)
    if n_todo == 0:
        print("Nothing to build -- all samples present."); return

    manifest = build(mt, model_db=DB, out_folder=OUT, cutoff=CUTOFF, threads=THREADS, solver=SOLVER)
    total = len(glob.glob(os.path.join(OUT, "*.pickle")))
    print(f"\nbuilt {len(manifest)} this run; total pickles now: {total} / 1650", flush=True)


if __name__ == "__main__":
    main()
