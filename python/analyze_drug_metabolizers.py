import pandas as pd

df = pd.read_excel("data/raw/knownDrugMetabolizers.xlsx", sheet_name="Table_Sx2")

# Filter for SN-38G beta-glucuronidase entries
sn38 = df[df["VMH ID of corresponding reaction(s)"].str.contains("SN38G", na=False)].copy()
print(f"Total drug metabolizer entries: {len(df)}")
print(f"SN-38G beta-glucuronidase entries: {len(sn38)}")
print(f"\nAll SN-38G producers in AGORA2:")
print(sn38[["Strain/species", "Drug metabolism activity", "In vitro", "In silico"]].to_string())

# Normalize strain names to "Genus species" format for matching
sn38["species_name"] = (
    sn38["Strain/species"]
    .str.replace("_", " ")
    .str.extract(r"^(\w+ \w+)")  # first two words = genus species
)

# Check overlap with our CRC cohort taxa
taxonomy = pd.read_parquet("data/processed/taxonomy_micom.parquet")
crc_taxa = set(taxonomy["id"].unique())

sn38["in_crc_cohorts"] = sn38["species_name"].apply(
    lambda s: any(str(s).lower() in t.lower() for t in crc_taxa) if pd.notna(s) else False
)

print(f"\nSN-38G producers present in CRC cohorts: "
      f"{sn38['in_crc_cohorts'].sum()} / {len(sn38)}")
print(sn38[sn38["in_crc_cohorts"]][["Strain/species", "In vitro", "In silico"]].to_string())

# Summary of all drugs covered
print(f"\nAll drugs covered in knownDrugMetabolizers:")
print(df["Drug metabolism activity"].value_counts().to_string())
