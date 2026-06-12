suppressMessages({library(curatedMetagenomicData); library(dplyr)})
sm <- sampleMetadata

tc <- sm |> filter(study_name=="ThomasAM_2019_c")
ya <- sm |> filter(study_name=="YachidaS_2019")

cat("ThomasAM_2019_c: n=", nrow(tc), " country=", paste(unique(tc$country),collapse=","), "\n", sep="")
cat("YachidaS_2019  : n=", nrow(ya), " country=", paste(unique(ya$country),collapse=","), "\n\n", sep="")

cat("subject_id overlap:", length(intersect(tc$subject_id, ya$subject_id)), "\n")
cat("sample_id  overlap:", length(intersect(tc$sample_id,  ya$sample_id)),  "\n\n")

cat("ThomasAM_2019_c sample_id format (first 5):\n"); print(head(tc$sample_id,5))
cat("\nYachidaS_2019 sample_id format (first 5):\n");   print(head(ya$sample_id,5))

# substring cross-check (in case one re-labels the other)
hit <- sum(sapply(tc$sample_id, function(s) any(grepl(s, ya$sample_id, fixed=TRUE)))) +
       sum(sapply(ya$sample_id, function(s) any(grepl(s, tc$sample_id, fixed=TRUE))))
cat("\nsubstring cross-matches between sample_ids:", hit, "\n")

# distributional sanity (age/gender) — would be suspicious if identical
cat("\nThomasAM_2019_c: age", sprintf("%.1f±%.1f", mean(tc$age,na.rm=TRUE), sd(tc$age,na.rm=TRUE)),
    " gender", paste(names(table(tc$gender)), table(tc$gender), collapse=" "), "\n")
cat("YachidaS_2019  : age", sprintf("%.1f±%.1f", mean(ya$age,na.rm=TRUE), sd(ya$age,na.rm=TRUE)),
    " gender", paste(names(table(ya$gender)), table(ya$gender), collapse=" "), "\n")

# any PMID/DOI-like provenance column?
prov <- intersect(c("PMID","study_name","curator"), names(sm))
cat("\nprovenance cols present:", paste(prov, collapse=", "), "\n")
if ("PMID" %in% names(sm)) {
  cat("ThomasAM_2019_c PMID:", paste(unique(tc$PMID),collapse=","), "\n")
  cat("YachidaS_2019  PMID:", paste(unique(ya$PMID),collapse=","), "\n")
}
