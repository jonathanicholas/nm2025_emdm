# Set parameters
N_CHAINS = 4  # Number of chains for MCMC sampling
N_ITER = 2000 # Number of iterations per chain

library(brms)
library(rstan)
library(lme4)
library(loo)
library(ggplot2)
library(tibble)
library(kableExtra)
library(dplyr)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Function to extract parameter info from fit models
get_credible_intervals <- function(fit) {
  # Get posterior summaries with multiple probability levels
  post_summary <- posterior_summary(fit, probs = c(0.25, 0.75, 0.1, 0.9, 0.025, 0.975))
  
  # Extract different credible intervals
  data.frame(
    parameter = rownames(post_summary),
    estimate = post_summary[,"Estimate"],
    error = post_summary[,"Est.Error"],
    ci_50_lower = post_summary[,"Q25"],
    ci_50_upper = post_summary[,"Q75"],
    ci_80_lower = post_summary[,"Q10"],
    ci_80_upper = post_summary[,"Q90"],
    ci_95_lower = post_summary[,"Q2.5"],
    ci_95_upper = post_summary[,"Q97.5"]
  )
}

# Load datasets
dpath = file.path(getwd(), "data")
choice_df <- read.csv("choiceDataPilot.csv")
choice_df$value_z = scale(choice_df$value)
choice_df$value_memory_z = scale(choice_df$value_memory)

# Fit choice models

fit1 <- brm(choice_bin ~ value_z + (value_z | wid), 
            family = bernoulli, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

fit2 <- brm(choice_bin ~ value_memory_z + (value_memory_z | wid), 
            family = bernoulli, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Extract and save fixed effects for choice models
choice_effects <- data.frame(
   model = c(rep("true", nrow(posterior_summary(fit1))),
             rep("memory", nrow(posterior_summary(fit2)))),
   rbind(
     get_credible_intervals(fit1),
     get_credible_intervals(fit2)
   )
 )
write.csv(choice_effects, 
          "choice_effects_pilot.csv", 
          row.names = FALSE)

# # Create folds and run kfold cross validation (10-fold)
fit1 = add_criterion(fit1, "kfold")
fit2 = add_criterion(fit2, "kfold")
 # Extract ELPD estimates
elpd_true = fit1$criteria$kfold$pointwise[,1]
elpd_mem = fit2$criteria$kfold$pointwise[,1]

elpd_diff = (elpd_mem - elpd_true)
elpd_se = sqrt(length(elpd_diff)) * sd(elpd_diff)

elpd_df_out <- data.frame("elpd_diff" = c(sum(elpd_diff)),
                          "se_diff" = c(elpd_se))
elpd_raw <- data.frame(elpd_mem = elpd_mem, elpd_true = elpd_true, elpd_diff = elpd_diff)

# # Write output files with version in filename
write.csv(elpd_df_out, "choice_elpd_diff_pilot.csv")
write.csv(elpd_raw, "choice_elpd_diff_raw_pilot.csv")

# Fit combined model

# Fit total memories model
fit3 <- brm(rt ~ total_pairs_remembered + (total_pairs_remembered | wid), 
            family = shifted_lognormal, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

rt_effects <- data.frame(
  model = c(rep("total_pairs_remembered", nrow(posterior_summary(fit3)))),
  rbind(
    get_credible_intervals(fit3)
  )
)
write.csv(rt_effects, 
          "rt_effects_pilot.csv", 
          row.names = FALSE)

fit4 <- brm(rt ~ Combined + (Combined | wid), 
            family = shifted_lognormal, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

rt_effects2 <- data.frame(
  model = c(rep("Combined", nrow(posterior_summary(fit4)))),
  rbind(
    get_credible_intervals(fit4)
  )
)
write.csv(rt_effects2, 
          "rt_effects_pilot_combined.csv", 
          row.names = FALSE)