"""Single source of truth for GUS enzyme-class catalytic-efficiency weighting.

Used by the efficiency-reweighted analyses and figures (make_fig3, make_fig4) and by the
supplementary GUS-efficiency tables (29_tables_s3s4_gus_efficiency.py), so the figures, the
supplement, and the pipeline can never drift. Class efficiencies are anchored to the SN-38-G
kinetics of representative enzymes (Pellock et al. 2018; see Text S1.3); the pipeline script
13_r6_refined.py holds an identical copy for the locked capacity computation.

  CLASS_EFF       relative SN-38-G efficiency per GUS structural class
  SPECIES_CLASS   species -> GUS class (characterized reconstructions)
  GENUS_DEFAULT   genus-level fallback efficiency for uncharacterized species
  eff(taxon)      resolve a taxon to its relative efficiency (species, else genus, else 0.5)
"""

CLASS_EFF = {"L1": 1.0, "NL": 1.0, "mL1": 0.4, "L2": 0.4, "mL2": 0.05, "NC": 0.0}
SPECIES_CLASS = {
    "Faecalibacterium_prausnitzii": "L1", "Eubacterium_eligens": "L1", "Escherichia_coli": "L1",
    "Clostridium_perfringens": "L1", "Bacteroides_uniformis": "NL", "Bacteroides_ovatus": "NL",
    "Bacteroides_dorei": "NL", "Bacteroides_massiliensis": "NL", "Parabacteroides_merdae": "NL",
    "Bacteroides_vulgatus": "mL1", "Bacteroides_fragilis": "mL1", "Ruminococcus_gnavus": "mL1",
    "Bacteroides_cellulosilyticus": "L2", "Lactobacillus_rhamnosus": "L1", "Prevotella_copri": "L1",
}
GENUS_DEFAULT = {"Bacteroides": 0.7, "Parabacteroides": 0.7, "Escherichia": 1.0,
                 "Faecalibacterium": 1.0, "Eubacterium": 0.7, "Clostridium": 0.7,
                 "Roseburia": 0.5, "Ruminococcus": 0.4, "Prevotella": 0.5, "Paraprevotella": 0.5}


def eff(taxon):
    if taxon in SPECIES_CLASS:
        return CLASS_EFF[SPECIES_CLASS[taxon]]
    return GENUS_DEFAULT.get(taxon.split("_")[0], 0.5)
