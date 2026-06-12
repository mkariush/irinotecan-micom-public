suppressMessages({library(curatedMetagenomicData); library(dplyr); library(tidyr)})
sm <- sampleMetadata
all_crc <- c("ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a","ThomasAM_2018b",
             "WirbelJ_2018","VogtmannE_2016","YachidaS_2019","ThomasAM_2019_c",
             "GuptaA_2019","HanniganGD_2017")

# Build, per study, the UNION of all identifier-like values:
#   sample_id, subject_id, and exploded NCBI_accession. Overlap in ANY field counts.
id_set <- function(study) {
  s <- sm |> filter(study_name==study, body_site=="stool")
  acc <- unlist(strsplit(paste(na.omit(s$NCBI_accession), collapse=";"), ";"))
  ids <- unique(trimws(c(s$sample_id, s$subject_id, acc)))
  ids[ids!="" & ids!="NA" & !is.na(ids)]
}
sets <- setNames(lapply(all_crc, id_set), all_crc)

cat("=== identifier-set sizes per study ===\n")
for (s in all_crc) cat(sprintf("  %-16s %d ids (n_samples=%d)\n", s, length(sets[[s]]),
                                sum(sm$study_name==s & sm$body_site=="stool")))

cat("\n=== ALL pairwise overlaps across union(sample_id, subject_id, NCBI_accession) ===\n")
found <- FALSE
for (i in seq_along(all_crc)) for (j in seq_along(all_crc)) if (i<j) {
  ov <- length(intersect(sets[[all_crc[i]]], sets[[all_crc[j]]]))
  if (ov>0) { cat(sprintf("  ** %s <-> %s : %d shared ids **\n", all_crc[i], all_crc[j], ov)); found<-TRUE }
}
if (!found) cat("  NONE. All 11 CRC cohorts are fully disjoint across every identifier field.\n")

cat("\n=== focused: Yachida vs each Thomas cohort ===\n")
for (th in c("ThomasAM_2018a","ThomasAM_2018b","ThomasAM_2019_c")) {
  cat(sprintf("  Yachida vs %s: %d shared ids\n", th,
              length(intersect(sets[["YachidaS_2019"]], sets[[th]]))))
}
cat("\nID format samples:\n")
cat("  Yachida        :", paste(head(sets[["YachidaS_2019"]],2),collapse=", "), "\n")
cat("  ThomasAM_2018a :", paste(head(sets[["ThomasAM_2018a"]],2),collapse=", "), "\n")
cat("  ThomasAM_2018b :", paste(head(sets[["ThomasAM_2018b"]],2),collapse=", "), "\n")
cat("  ThomasAM_2019_c:", paste(head(sets[["ThomasAM_2019_c"]],2),collapse=", "), "\n")
