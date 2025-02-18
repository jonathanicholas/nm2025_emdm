# Set parameters
N_CHAINS = 4  # Number of chains for MCMC sampling
N_ITER = 2000 # Number of iterations per chain

args = commandArgs(trailingOnly = TRUE)
dpath = args[1]
opath = args[2]
exp_version = args[3]

library(brms)
library(rstan)
library(tidyverse)
options(mc.cores = parallel::detectCores())

# Load data
diff_df <- read.csv(file.path(dpath, paste0("last_round_differences_", exp_version, ".csv")))

get_hypothesis_CIs <- function(model, hypothesis_string) {
  h95 <- hypothesis(model, hypothesis_string, alpha = 0.05)
  h80 <- hypothesis(model, hypothesis_string, alpha = 0.20)
  h50 <- hypothesis(model, hypothesis_string, alpha = 0.50)
  
  return(list(
    estimate = h95$hypothesis$Estimate,
    error = h95$hypothesis$Est.Error,
    ci_50_lower = h50$hypothesis$CI.Lower,
    ci_50_upper = h50$hypothesis$CI.Upper,
    ci_80_lower = h80$hypothesis$CI.Lower,
    ci_80_upper = h80$hypothesis$CI.Upper,
    ci_95_lower = h95$hypothesis$CI.Lower,
    ci_95_upper = h95$hypothesis$CI.Upper
  ))
}

# Create empty list to store results
all_effects <- list()

# For each difference measure, fit model and test against 0
difference_columns <- setdiff(names(diff_df), "wid")

for(col in difference_columns) {
  # Fit model
  model <- brm(
    formula = as.formula(paste(col, "~ 1")),
    data = diff_df,
    family = gaussian(),
    chains = N_CHAINS,
    iter = N_ITER
  )
  
  # Get hypothesis test results
  h <- get_hypothesis_CIs(model, "Intercept = 0")
  
  # Store results
  effects <- data.frame(
    comparison = col,
    parameter = "Intercept",
    estimate = h$estimate,
    error = h$error,
    ci_50_lower = h$ci_50_lower,
    ci_50_upper = h$ci_50_upper,
    ci_80_lower = h$ci_80_lower,
    ci_80_upper = h$ci_80_upper,
    ci_95_lower = h$ci_95_lower,
    ci_95_upper = h$ci_95_upper
  )
  
  all_effects[[col]] <- effects
}

# Combine all results
results_df <- do.call(rbind, all_effects)

# Save results
write.csv(results_df,
          file.path(opath, paste0("last_round_differences_effects_", exp_version, ".csv")), 
          row.names = FALSE)