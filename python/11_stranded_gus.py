"""Stranded-GUS validation: is it real biology or an AGORA2 reconstruction gap?

Stranded taxon = has SN38G_GLCAASE__taxon but NO EX_sn38g(e)__taxon (no SN-38G uptake from the
lumen) -> can never carry SN-38G flux -> contributes 0 despite having the GUS gene.

Memory-safe: load community models ONE AT A TIME (del + gc between) to identify the stranded
SPECIES set (intrinsic to AGORA2). Then use the light taxonomy table to measure prevalence/impact
and flag whether any stranded species are KNOWN SN-38G reactivators (=> likely AGORA2 artifact).
"""

import glob
import os
import gc
import collections
import pandas as pd
from micom.util import load_pickle

MODELS_DIR = "data/processed/models"
N_PER_COHORT = 10          # sample this many models per cohort to surface the stranded set

# Known SN-38-G reactivators (literature: Wallace, Pellock, Jariwala, CLAUDE.md) -- underscored
KNOWN_REACTIVATORS = {
    "Escherichia_coli", "Bacteroides_fragilis", "Bacteroides_uniformis", "Bacteroides_vulgatus",
    "Bacteroides_thetaiotaomicron", "Bacteroides_ovatus", "Parabacteroides_distasonis",
    "Clostridium_perfringens", "Faecalibacterium_prausnitzii", "Roseburia_hominis",
    "Clostridium_leptum", "Eubacterium_eligens",
}


def pick_models():
    meta = pd.read_parquet("data/processed/sample_metadata.parquet")
    built = {os.path.basename(p).replace(".pickle", "") for p in glob.glob(f"{MODELS_DIR}/*.pickle")}
    meta = meta[meta.sample_id.isin(built)]
    keep = meta.sort_values("sample_id").groupby("cohort").head(N_PER_COHORT)["sample_id"].tolist()
    return [f"{MODELS_DIR}/{s}.pickle" for s in keep]


if __name__ == "__main__":
    paths = pick_models()
    print(f"Scanning {len(paths)} models (sequential, memory-safe) to find stranded GUS taxa\n")

    gus_occur = collections.Counter()       # species -> # models where it has GUS
    stranded_occur = collections.Counter()  # species -> # models where it is stranded
    for i, p in enumerate(paths, 1):
        com = load_pickle(p)
        gus = {r.id.split("__")[-1] for r in com.reactions if "SN38G_GLCAASE" in r.id}
        upt = {r.id.split("__")[-1] for r in com.reactions if r.id.startswith("EX_sn38g(e)__")}
        for t in gus:
            gus_occur[t] += 1
        for t in (gus - upt):
            stranded_occur[t] += 1
        del com
        gc.collect()
        if i % 10 == 0:
            print(f"  scanned {i}/{len(paths)}")

    # species ever stranded
    stranded_species = sorted(stranded_occur)
    print(f"\n=== {len(stranded_species)} species appear STRANDED (GUS but no uptake) ===")
    print(f"{'species':40s} {'stranded_in':>11s} {'gus_in':>7s}  known_reactivator")
    for s in sorted(stranded_species, key=lambda x: -stranded_occur[x]):
        flag = "*** KNOWN REACTIVATOR ***" if s in KNOWN_REACTIVATORS else ""
        print(f"{s:40s} {stranded_occur[s]:>11d} {gus_occur[s]:>7d}  {flag}")

    if not stranded_species:
        print("\n=== VERDICT: NO stranded taxa found across the scanned models. ===")
        print("AGORA2 consistently pairs the GUS enzyme with its SN-38G uptake transport "
              "-> no pathway-completeness gap. (NB: SN38G_GLCAASE has variants per taxon; count "
              "UNIQUE taxa, not reactions.)")
        raise SystemExit(0)

    # impact: how common/abundant are stranded species in the full cohort taxonomy
    tax = pd.read_parquet("data/processed/taxonomy_micom.parquet")
    tax["taxon"] = tax["id"].str.replace(" ", "_", regex=False)
    n_samples = tax.sample_id.nunique()
    print(f"\n=== prevalence/abundance of stranded species across all {n_samples} samples ===")
    rows = []
    for s in stranded_species:
        sub = tax[tax.taxon == s]
        rows.append({"species": s, "n_samples": sub.sample_id.nunique(),
                     "pct_samples": round(100*sub.sample_id.nunique()/n_samples, 1),
                     "mean_abund_when_present": round(sub.abundance.mean(), 4) if len(sub) else 0,
                     "known_reactivator": s in KNOWN_REACTIVATORS})
    imp = pd.DataFrame(rows).sort_values("pct_samples", ascending=False)
    print(imp.to_string(index=False))

    any_known = [s for s in stranded_species if s in KNOWN_REACTIVATORS]
    print("\n=== VERDICT ===")
    if any_known:
        print(f"ARTIFACT LIKELY: {len(any_known)} stranded species are KNOWN reactivators -> AGORA2"
              f" omitted their transport. These SHOULD reactivate. Consider forcing transport.")
        print("  ", any_known)
    else:
        print("Stranded species are NOT known reactivators -> plausibly real / low-impact.")
    imp.to_csv("data/processed/flux/stranded_gus.csv", index=False)
    print("\nsaved -> data/processed/flux/stranded_gus.csv")
