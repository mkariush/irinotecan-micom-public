library(curatedMetagenomicData)
library(SummarizedExperiment)
library(tidyverse)
library(arrow)

# Primary cohorts (original 7 study-names = "6 cohorts"; ThomasAM_2018a/b = one Thomas cohort)
# + expansion (2026-06-12): YachidaS_2019 (Heinken's cohort, JPN) and ThomasAM_2019_c (JPN),
#   both verified distinct (R/99*.R) and good depth (>40M reads).
# Depth-SENSITIVITY only (shallow: <10M median reads) — fetched but excluded from primary:
#   GuptaA_2019 (IND, 8.7M), HanniganGD_2017 (USA, 5.9M; some 0-read samples).
COHORTS_PRIMARY <- c(
  "ZellerG_2014", "YuJ_2015", "FengQ_2015",
  "ThomasAM_2018a", "ThomasAM_2018b", "WirbelJ_2018", "VogtmannE_2016",
  "YachidaS_2019", "ThomasAM_2019_c"
)
COHORTS_SENSITIVITY <- c("GuptaA_2019", "HanniganGD_2017")
COHORTS <- c(COHORTS_PRIMARY, COHORTS_SENSITIVITY)

OUT_DIR <- "data/raw"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

fetch_cohort <- function(cohort) {
  # resolve resource(s); if multiple dated versions exist, take the most recent and log it
  res_names <- curatedMetagenomicData(paste0(cohort, ".relative_abundance"), dryrun = TRUE)
  chosen <- tail(sort(res_names), 1)
  message("Fetching: ", cohort, "  (version: ", chosen, ")")
  se_list <- curatedMetagenomicData(chosen, dryrun = FALSE, counts = FALSE)
  se <- se_list[[length(se_list)]]
  list(abundance = assay(se, "relative_abundance"),
       metadata  = as.data.frame(colData(se)),
       version   = chosen)
}

versions <- c()
for (cohort in COHORTS) {
  abund_path <- file.path(OUT_DIR, paste0(cohort, "_abundance.parquet"))
  if (file.exists(abund_path)) { message("Skip (exists): ", cohort); next }

  res <- fetch_cohort(cohort)
  versions[cohort] <- res$version

  abund_df <- as.data.frame(t(res$abundance)) |>
    rownames_to_column("sample_id") |>
    mutate(cohort = cohort)
  meta_df <- res$metadata |>
    rownames_to_column("sample_id") |>
    mutate(cohort = cohort)

  write_parquet(abund_df, abund_path)
  write_parquet(meta_df,  file.path(OUT_DIR, paste0(cohort, "_metadata.parquet")))
  message("  Saved: ", nrow(meta_df), " samples, ", ncol(abund_df) - 2, " taxa")
}

# record resolved versions for Methods [verify]
if (length(versions)) {
  writeLines(paste0(names(versions), "\t", versions),
             file.path(OUT_DIR, "cohort_versions_new.tsv"))
  message("Versions written to ", file.path(OUT_DIR, "cohort_versions_new.tsv"))
}
message("Done. Files in ", OUT_DIR)
