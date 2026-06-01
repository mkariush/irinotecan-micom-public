import pandas as pd

manifest = pd.read_csv(r"databases\AGORA2_json\manifest.csv")

# Probiotic species of interest
probiotics = [
    # Lactobacillaceae
    "Lactobacillus rhamnosus",
    "Lactobacillus acidophilus",
    "Lactobacillus plantarum",
    "Lactobacillus reuteri",
    "Lactobacillus casei",
    "Lactobacillus helveticus",
    "Lactobacillus fermentum",
    "Lactobacillus gasseri",
    "Lactobacillus johnsonii",
    "Lactobacillus salivarius",
    "Lactiplantibacillus plantarum",   # reclassified name
    "Lacticaseibacillus rhamnosus",    # reclassified name
    "Lacticaseibacillus casei",        # reclassified name
    "Ligilactobacillus salivarius",    # reclassified name
    "Limosilactobacillus reuteri",     # reclassified name
    "Ligilactobacillus acidophilus",
    # Bifidobacterium (some are known GUS producers)
    "Bifidobacterium longum",
    "Bifidobacterium breve",
    "Bifidobacterium adolescentis",
    "Bifidobacterium animalis",
    "Bifidobacterium bifidum",
    "Bifidobacterium infantis",
    "Bifidobacterium lactis",
    # Other common probiotics
    "Streptococcus thermophilus",
    "Enterococcus faecium",
    "Bacillus subtilis",
    "Bacillus coagulans",
    "Pediococcus acidilactici",
    "Pediococcus pentosaceus",
]

# Known beta-glucuronidase producers from Heinken et al.
known_gus = {
    "Bifidobacterium adolescentis",
    "Bifidobacterium breve",
    "Bifidobacterium longum",
}

manifest["species_name"] = manifest["genus"] + " " + manifest["species"]
results = []
for prob in probiotics:
    genus, species = prob.split(" ", 1)
    matches = manifest[
        (manifest["genus"].str.lower() == genus.lower()) &
        (manifest["species"].str.lower() == species.lower())
    ]
    results.append({
        "probiotic":    prob,
        "in_agora2":    len(matches) > 0,
        "n_strains":    len(matches),
        "example_strain": matches["id"].iloc[0] if len(matches) > 0 else "",
        "known_gus":    prob in known_gus,
    })

df = pd.DataFrame(results)

print("=== Probiotic strains in AGORA2 ===\n")
print("PRESENT (sorted by GUS status):")
present = df[df["in_agora2"]].sort_values("known_gus", ascending=False)
for _, r in present.iterrows():
    gus_flag = " *** GUS PRODUCER ***" if r["known_gus"] else ""
    print(f"  {r['probiotic']:45s} {r['n_strains']} strain(s)  {r['example_strain']}{gus_flag}")

print(f"\nNOT FOUND in AGORA2:")
missing = df[~df["in_agora2"]]
for _, r in missing.iterrows():
    print(f"  {r['probiotic']}")

print(f"\nSummary: {df['in_agora2'].sum()}/{len(df)} probiotic species found in AGORA2")
print(f"Of which known GUS producers: {(df['in_agora2'] & df['known_gus']).sum()}")
