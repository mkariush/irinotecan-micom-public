suppressMessages({library(curatedMetagenomicData); library(dplyr); library(tidyr)})
sm <- sampleMetadata
all_crc <- c("ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a","ThomasAM_2018b",
             "WirbelJ_2018","VogtmannE_2016","YachidaS_2019","ThomasAM_2019_c",
             "GuptaA_2019","HanniganGD_2017")

cat("available id-ish columns:",
    paste(intersect(c("NCBI_accession","subject_id","sample_id"), names(sm)), collapse=", "), "\n\n")

# explode NCBI_accession (may be ';'-separated runs) into one row per accession
acc <- sm |> filter(study_name %in% all_crc, body_site=="stool") |>
  select(study_name, NCBI_accession) |>
  separate_rows(NCBI_accession, sep=";") |>
  mutate(NCBI_accession=trimws(NCBI_accession)) |>
  filter(!is.na(NCBI_accession), NCBI_accession!="", NCBI_accession!="NA")

cat("accession counts per study (after exploding):\n")
print(as.data.frame(acc |> group_by(study_name) |> summarise(n_acc=n(), n_unique=n_distinct(NCBI_accession), .groups="drop")), row.names=FALSE)

# pairwise accession overlap matrix
studies <- unique(acc$study_name)
sets <- split(acc$NCBI_accession, acc$study_name)
cat("\nPairwise shared-accession counts (only nonzero shown):\n")
found <- FALSE
for (i in seq_along(studies)) for (j in seq_along(studies)) if (i<j) {
  ov <- length(intersect(sets[[studies[i]]], sets[[studies[j]]]))
  if (ov>0) { cat(sprintf("  %s  <->  %s : %d shared\n", studies[i], studies[j], ov)); found<-TRUE }
}
if (!found) cat("  NONE — all cohorts are disjoint by NCBI_accession (no double-counting).\n")

# focus: Thomas_2019_c sample of accessions vs Yachida
cat("\nThomasAM_2019_c accession sample:", paste(head(sets[["ThomasAM_2019_c"]],3),collapse=", "), "\n")
cat("YachidaS_2019  accession sample:", paste(head(sets[["YachidaS_2019"]],3),collapse=", "), "\n")
