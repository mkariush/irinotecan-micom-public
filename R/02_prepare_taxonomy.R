library(tidyverse)
library(arrow)

# Bioconductor packages loaded in the same session mask several dplyr verbs;
# resolve conflicts explicitly.
conflicted::conflict_prefer("select", "dplyr")
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("rename", "dplyr")

COHORTS <- c(
  "ZellerG_2014",
  "YuJ_2015",
  "FengQ_2015",
  "ThomasAM_2018a",
  "ThomasAM_2018b",
  "WirbelJ_2018",
  "VogtmannE_2016"
)

RAW_DIR    <- "data/raw"
PROC_DIR   <- "data/processed"
ABUND_CUTOFF <- 0.001

dir.create(PROC_DIR, showWarnings = FALSE, recursive = TRUE)

prepare_micom_table <- function(cohort, cutoff = ABUND_CUTOFF) {
  abund <- read_parquet(file.path(RAW_DIR, paste0(cohort, "_abundance.parquet")))
  meta  <- read_parquet(file.path(RAW_DIR, paste0(cohort, "_metadata.parquet")))

  long <- abund |>
    pivot_longer(-c(sample_id, cohort), names_to = "taxon", values_to = "abundance") |>
    dplyr::filter(abundance >= cutoff)

  long <- long |>
    dplyr::group_by(sample_id) |>
    dplyr::mutate(abundance = abundance / sum(abundance)) |>
    dplyr::ungroup()

  # curatedMetagenomicData column names use MetaPhlAn format:
  # "k__Bacteria|p__Firmicutes|...|s__Bacteroides_fragilis"
  # Extract species name (after |s__) and convert underscores to spaces.
  long <- long |>
    dplyr::mutate(
      taxon = stringr::str_extract(taxon, "(?<=\\|s__)[^|]+$"),
      taxon = stringr::str_replace_all(taxon, "_", " ")
    ) |>
    dplyr::filter(!is.na(taxon))

  # Join all metadata columns (drop cohort from meta — already in long from abund)
  meta_cols <- dplyr::select(meta, -dplyr::any_of("cohort"))
  long |>
    dplyr::left_join(meta_cols, by = "sample_id") |>
    dplyr::rename(id = taxon)
}

all_cohorts <- map(COHORTS, prepare_micom_table)
combined    <- bind_rows(all_cohorts)

write_parquet(combined, file.path(PROC_DIR, "taxonomy_micom.parquet"))

combined |>
  dplyr::group_by(cohort, study_condition) |>
  dplyr::summarise(n_samples = n_distinct(sample_id), n_taxa = n_distinct(id), .groups = "drop") |>
  print()

message("Done. Written to ", PROC_DIR, "/taxonomy_micom.parquet")
