suppressMessages({library(curatedMetagenomicData); library(dplyr)})

sm <- sampleMetadata

# Studies that contain any CRC samples (stool)
crc_studies <- sm |>
  filter(body_site == "stool") |>
  group_by(study_name) |>
  summarise(
    n_total   = n(),
    n_CRC     = sum(study_condition == "CRC", na.rm = TRUE),
    n_adenoma = sum(study_condition == "adenoma", na.rm = TRUE),
    n_control = sum(study_condition == "control", na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(n_CRC > 0) |>
  arrange(desc(n_CRC))

have <- c("ZellerG_2014","YuJ_2015","FengQ_2015","ThomasAM_2018a",
          "ThomasAM_2018b","WirbelJ_2018","VogtmannE_2016")

crc_studies <- crc_studies |>
  mutate(status = ifelse(study_name %in% have, "INCLUDED", "** MISSED **"))

cat("\n=== All curatedMetagenomicData studies with CRC stool samples ===\n")
print(as.data.frame(crc_studies), row.names = FALSE)

cat("\n=== Yachida dryrun (availability check) ===\n")
yd <- tryCatch(
  curatedMetagenomicData("YachidaS_2019.relative_abundance", dryrun = TRUE),
  error = function(e) paste("ERROR:", conditionMessage(e))
)
print(yd)
