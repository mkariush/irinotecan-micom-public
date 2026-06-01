import pandas as pd

taxonomy = pd.read_parquet("data/processed/taxonomy_micom.parquet")
manifest = pd.read_csv(r"databases\AGORA2_SBML\manifest.csv")  # direct CSV path still valid here

crc_taxa  = set(taxonomy["id"].unique())
agora_spp = set((manifest["genus"] + " " + manifest["species"]).unique())

matched   = crc_taxa & agora_spp
unmatched = crc_taxa - agora_spp

print(f"Unique taxa in CRC cohorts:    {len(crc_taxa)}")
print(f"Match to AGORA2 genus+species: {len(matched)}")
print(f"Unmatched:                     {len(unmatched)}")
print(f"\nFirst 20 unmatched:")
for t in sorted(unmatched)[:20]:
    print(f"  {t}")

# Abundance coverage: what fraction of per-sample abundance is captured by matched taxa
coverage = (
    taxonomy
    .assign(in_agora=taxonomy["id"].isin(matched))
    .groupby(["sample_id", "in_agora"])["abundance"]
    .sum()
    .unstack(fill_value=0)
)
coverage["pct_covered"] = coverage[True] / (coverage[True] + coverage[False])
print(f"\nPer-sample abundance coverage by matched AGORA2 taxa:")
print(f"  Median: {coverage['pct_covered'].median():.1%}")
print(f"  Min:    {coverage['pct_covered'].min():.1%}")
print(f"  Max:    {coverage['pct_covered'].max():.1%}")
