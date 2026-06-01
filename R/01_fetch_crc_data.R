library(curatedMetagenomicData)
library(SummarizedExperiment)
library(tidyverse)
library(arrow)

COHORTS <- c(
  "ZellerG_2014",
  "YuJ_2015",
  "FengQ_2015",
  "ThomasAM_2018a",
  "ThomasAM_2018b",
  "WirbelJ_2018",
  "VogtmannE_2016"
)

OUT_DIR <- "data/raw"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

fetch_cohort <- function(cohort) {
  message("Fetching: ", cohort)
  se_list <- curatedMetagenomicData(
    paste0(cohort, ".relative_abundance"),
    dryrun = FALSE,
    counts = FALSE
  )
  se <- se_list[[1]]

  abund <- assay(se, "relative_abundance")
  meta  <- as.data.frame(colData(se))

  list(abundance = abund, metadata = meta)
}

for (cohort in COHORTS) {
  res <- fetch_cohort(cohort)

  abund_df <- as.data.frame(t(res$abundance)) |>
    rownames_to_column("sample_id") |>
    mutate(cohort = cohort)

  meta_df <- res$metadata |>
    rownames_to_column("sample_id") |>
    select(any_of(c("sample_id", "study_condition", "age", "gender", "BMI", "country"))) |>
    mutate(cohort = cohort)

  write_parquet(abund_df, file.path(OUT_DIR, paste0(cohort, "_abundance.parquet")))
  write_parquet(meta_df,  file.path(OUT_DIR, paste0(cohort, "_metadata.parquet")))

  message("  Saved: ", nrow(meta_df), " samples, ", ncol(abund_df) - 2, " taxa")
}

message("Done. Files written to ", OUT_DIR)
