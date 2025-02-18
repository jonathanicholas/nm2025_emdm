# Set parameters
N_CHAINS = 4  # Number of chains for MCMC sampling
N_ITER = 2000 # Number of iterations per chain

args = commandArgs(trailingOnly = TRUE)
dpath = args[1]
opath = args[2]
exp_version = args[3]

library(brms)
library(rstan)
library(lme4)
library(loo)
library(ggplot2)
library(tibble)
library(kableExtra)
library(dplyr)
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
choice_df <- read.csv(file.path(dpath, paste0("choiceDataFull_", exp_version, ".csv")))
choice_df_mem <- read.csv(file.path(dpath, paste0("choiceDataMemValue_", exp_version, ".csv")))
choice_df = choice_df[choice_df$rt!=0,]
choice_df_mem = choice_df_mem[choice_df_mem$rt!=0,]
choice_df = na.omit(choice_df)
choice_df_mem = na.omit(choice_df_mem)

#######################################
#### Run choice models and compare ####
#######################################

# Separate into Before and After conditions for choice comparison
before_df = choice_df_mem[choice_df_mem$multi_time == "Before",]
after_df = choice_df_mem[choice_df_mem$multi_time == "After",]
before_df$value_z = scale(before_df$value)
after_df$value_z = scale(after_df$value)
before_df$recalled_mem_value_z = scale(before_df$recalled_mem_value)
after_df$recalled_mem_value_z = scale(after_df$recalled_mem_value)

# Run choice models with specified chains and iterations
fit1_before <- brm(choice ~ value_z + (value_z | wid), 
                   family = bernoulli, 
                   data = before_df,
                   chains = N_CHAINS,
                   iter = N_ITER)

fit1_after <- brm(choice ~ value_z + (value_z | wid), 
                  family = bernoulli, 
                  data = after_df,
                  chains = N_CHAINS,
                  iter = N_ITER)

fit2_before <- brm(choice ~ recalled_mem_value_z + (recalled_mem_value_z | wid), 
                   family = bernoulli, 
                   data = before_df,
                   chains = N_CHAINS,
                   iter = N_ITER)

fit2_after <- brm(choice ~ recalled_mem_value_z + (recalled_mem_value_z | wid), 
                  family = bernoulli, 
                  data = after_df,
                  chains = N_CHAINS,
                  iter = N_ITER)

# Extract and save all effects for choice models
choice_effects <- data.frame(
  model = c(rep("before_true", nrow(posterior_summary(fit1_before))),
            rep("before_memory", nrow(posterior_summary(fit2_before))),
            rep("after_true", nrow(posterior_summary(fit1_after))),
            rep("after_memory", nrow(posterior_summary(fit2_after)))),
  rbind(
    get_credible_intervals(fit1_before),
    get_credible_intervals(fit2_before),
    get_credible_intervals(fit1_after),
    get_credible_intervals(fit2_after)
  )
)
write.csv(choice_effects, 
          file.path(opath, paste0("choice_effects_", exp_version, ".csv")), 
          row.names = FALSE)

# Create folds and run kfold cross validation (10-fold)
fit1_before = add_criterion(fit1_before, "kfold")
fit2_before = add_criterion(fit2_before, "kfold")
fit1_after = add_criterion(fit1_after, "kfold")
fit2_after = add_criterion(fit2_after, "kfold")

# Extract ELPD estimates
elpd_true_before = fit1_before$criteria$kfold$pointwise[,1]
elpd_mem_before = fit2_before$criteria$kfold$pointwise[,1]
elpd_true_after = fit1_after$criteria$kfold$pointwise[,1]
elpd_mem_after = fit2_after$criteria$kfold$pointwise[,1]

elpd_diff_before = (elpd_mem_before - elpd_true_before)
elpd_se_before = sqrt(length(elpd_diff_before)) * sd(elpd_diff_before)
elpd_diff_after = (elpd_mem_after - elpd_true_after)
elpd_se_after = sqrt(length(elpd_diff_after)) * sd(elpd_diff_after)

elpd_df_out <- data.frame("elpd_diff" = c(sum(elpd_diff_before),sum(elpd_diff_after)),
                          "se_diff" = c(elpd_se_before,elpd_se_after),
                          "timepoint" = c("Before", "After"))
elpd_raw_before <- data.frame(elpd_mem = elpd_mem_before, elpd_true = elpd_true_before, elpd_diff = elpd_diff_before)
elpd_raw_after <- data.frame(elpd_mem = elpd_mem_after, elpd_true = elpd_true_after, elpd_diff = elpd_diff_after)

# Write output files with version in filename
write.csv(elpd_df_out, file.path(opath, paste0("choice_elpd_diff_", exp_version, ".csv")))
write.csv(elpd_raw_before, file.path(opath, paste0("choice_elpd_diff_raw_before_", exp_version, ".csv")))
write.csv(elpd_raw_after, file.path(opath, paste0("choice_elpd_diff_raw_after_", exp_version, ".csv")))

#######################################
###### Run RT models and compare ######
#######################################

# Run RT models with specified chains and iterations
fit3_before <- brm(rt ~ n_total_memories_shown + (n_total_memories_shown | wid), 
                   family = shifted_lognormal, 
                   data = before_df,
                   chains = N_CHAINS,
                   iter = N_ITER)

fit3_after <- brm(rt ~ n_total_memories_shown + (n_total_memories_shown | wid), 
                  family = shifted_lognormal, 
                  data = after_df,
                  chains = N_CHAINS,
                  iter = N_ITER)

fit4_before <- brm(rt ~ n_remembered_shown + (n_remembered_shown | wid), 
                   family = shifted_lognormal, 
                   data = before_df,
                   chains = N_CHAINS,
                   iter = N_ITER)

fit4_after <- brm(rt ~ n_remembered_shown + (n_remembered_shown | wid), 
                  family = shifted_lognormal, 
                  data = after_df,
                  chains = N_CHAINS,
                  iter = N_ITER)

# Extract and save all effects for RT models
rt_effects <- data.frame(
  model = c(rep("before_rel_only", nrow(posterior_summary(fit4_before))),
            rep("before_memory", nrow(posterior_summary(fit3_before))),
            rep("after_rel_only", nrow(posterior_summary(fit4_after))),
            rep("after_memory", nrow(posterior_summary(fit3_after)))),
  rbind(
    get_credible_intervals(fit4_before),
    get_credible_intervals(fit3_before),
    get_credible_intervals(fit4_after),
    get_credible_intervals(fit3_after)
  )
)
write.csv(rt_effects, 
          file.path(opath, paste0("rt_effects_", exp_version, ".csv")), 
          row.names = FALSE)