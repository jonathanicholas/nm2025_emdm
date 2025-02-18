dpath = file.path(getwd(), "data")
opath = file.path(getwd(), "output")
exp_versions = c("1A", "1B", "2")
manip_exp_versions = c("3A", "3B", "4")

analysis_scripts <- c(
  "scripts/analyze_exp_memory.R",
  "scripts/analyze_exp_overall_perf.R",
  "scripts/analyze_exp.R"
)

manip_analysis_scripts <- c(
  "scripts/analyze_manipulation_exp.R",
  "scripts/analyze_manipulation_exp_overall_perf.R",
  "scripts/analyze_manipulation_exp_memory.R"  # Removed trailing comma
)

for (version in exp_versions) {
  for (script in analysis_scripts) {
    cat(sprintf("\nRunning %s for version %s...\n", script, version))
    system2("Rscript", args = c(script, dpath, opath, version))
    cat("Completed.\n")
  }
}

for (version in manip_exp_versions) {
  for (script in manip_analysis_scripts) {
    cat(sprintf("\nRunning %s for version %s...\n", script, version))
    system2("Rscript", args = c(script, dpath, opath, version))
    cat("Completed.\n")
  }
}

cat(sprintf("\nRunning %s for version %s...\n", "scripts/analyze_final_round.R", "4"))
system2("Rscript", args = c("scripts/analyze_final_round.R", dpath, opath, "4"))
cat("Completed.\n")