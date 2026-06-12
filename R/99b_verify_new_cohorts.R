suppressMessages({library(curatedMetagenomicData); library(dplyr); library(tidyr)})
sm <- sampleMetadata

current <- c("ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a",
             "ThomasAM_2018b","WirbelJ_2018","VogtmannE_2016")
new     <- c("YachidaS_2019","ThomasAM_2019_c","GuptaA_2019","HanniganGD_2017")
all_crc <- c(current, new)

## ---- 1. ThomasAM_2019_c subject overlap with our Thomas 2018a/b ----
cat("\n================ THOMAS OVERLAP CHECK ================\n")
thomas <- sm |> filter(study_name %in% c("ThomasAM_2018a","ThomasAM_2018b","ThomasAM_2019_c"))
cat("Per-study summary:\n")
print(as.data.frame(thomas |> group_by(study_name) |>
  summarise(n=n(), countries=paste(unique(country),collapse=","),
            age=sprintf("%.0f-%.0f", min(age,na.rm=TRUE), max(age,na.rm=TRUE)),
            .groups="drop")), row.names=FALSE)

sid_2018 <- thomas |> filter(study_name %in% c("ThomasAM_2018a","ThomasAM_2018b")) |> pull(subject_id)
sid_2019 <- thomas |> filter(study_name=="ThomasAM_2019_c") |> pull(subject_id)
cat("\nsubject_id overlap (2019_c vs 2018a/b):",
    length(intersect(sid_2019, sid_2018)), "of", length(sid_2019), "2019_c subjects\n")
cat("Sample of 2019_c subject_ids:", paste(head(sid_2019,4),collapse=", "), "\n")
cat("Sample of 2018a/b subject_ids:", paste(head(sid_2018,4),collapse=", "), "\n")
# also check sample_id overlap as a fallback
cat("sample_id overlap (2019_c vs 2018a/b):",
    length(intersect(thomas |> filter(study_name=="ThomasAM_2019_c") |> pull(sample_id),
                     thomas |> filter(study_name!="ThomasAM_2019_c") |> pull(sample_id))), "\n")

## ---- 2. Sequencing depth + platform across ALL CRC cohorts ----
cat("\n================ DEPTH / PLATFORM CHECK ================\n")
depth <- sm |> filter(study_name %in% all_crc, body_site=="stool") |>
  group_by(study_name) |>
  summarise(n=n(),
            reads_M_median = round(median(number_reads/1e6, na.rm=TRUE),1),
            reads_M_min    = round(min(number_reads/1e6, na.rm=TRUE),1),
            platform = paste(unique(na.omit(sequencing_platform)), collapse=";"),
            .groups="drop") |>
  mutate(status=ifelse(study_name %in% new,"NEW","current")) |>
  arrange(reads_M_median)
print(as.data.frame(depth), row.names=FALSE)
cat("\n(FengQ is the existing low-depth flag; compare new cohorts against it.)\n")
