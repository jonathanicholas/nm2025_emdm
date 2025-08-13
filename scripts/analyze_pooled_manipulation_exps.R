# Set parameters
N_CHAINS = 4  # Number of chains for MCMC sampling
N_ITER = 2000 # Number of iterations per chain

args = commandArgs(trailingOnly = TRUE)
dpath = args[1]
opath = args[2]

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
# Function to calculate summary statistics with multiple intervals from posterior samples
get_posterior_summary <- function(samples, probs = c(.025, .05, .10, .90, .95, .975)) {
  c(
    mean = mean(samples),
    lower_95 = quantile(samples, probs[1]),
    upper_95 = quantile(samples, probs[6]),
    lower_90 = quantile(samples, probs[2]),
    upper_90 = quantile(samples, probs[5]),
    lower_80 = quantile(samples, probs[3]),
    upper_80 = quantile(samples, probs[4])
  )
}
# Function to create summary data frame for a set of effects and deviations
summarize_effects <- function(effects_A, effects_B, effects_2, 
                              dev_A, dev_B, 
                              model_names) {
  
  # Convert to matrix if vector
  if(!is.matrix(effects_A)) {
    effects_A <- matrix(effects_A, nrow=1)
    effects_B <- matrix(effects_B, nrow=1)
    effects_2 <- matrix(effects_2, nrow=1)
    dev_A <- matrix(dev_A, nrow=1)
    dev_B <- matrix(dev_B, nrow=1)
  }
  
  n_models <- length(model_names)
  
  # Combine all effects into matrices for each model
  effects_all <- list()
  dev_all <- list()
  
  for(i in 1:n_models) {
    effects_all[[i]] <- cbind(
      get_posterior_summary(effects_A[i,]),
      get_posterior_summary(effects_B[i,]),
      get_posterior_summary(effects_2[i,])
    )
    
    dev_all[[i]] <- cbind(
      get_posterior_summary(dev_A[i,]),
      get_posterior_summary(dev_B[i,]),
      get_posterior_summary(-(dev_A[i,] + dev_B[i,]))
    )
  }
  
  # Create data frame
  data.frame(
    model = rep(model_names, each = 3),
    experiment = rep(c("3A", "3B", "4"), n_models),
    # Effects
    mean = unlist(lapply(effects_all, function(x) x[1,])),
    lower_95 = unlist(lapply(effects_all, function(x) x[2,])),
    upper_95 = unlist(lapply(effects_all, function(x) x[3,])),
    lower_90 = unlist(lapply(effects_all, function(x) x[4,])),
    upper_90 = unlist(lapply(effects_all, function(x) x[5,])),
    lower_80 = unlist(lapply(effects_all, function(x) x[6,])),
    upper_80 = unlist(lapply(effects_all, function(x) x[7,])),
    # Deviations
    dev_mean = unlist(lapply(dev_all, function(x) x[1,])),
    dev_lower_95 = unlist(lapply(dev_all, function(x) x[2,])),
    dev_upper_95 = unlist(lapply(dev_all, function(x) x[3,])),
    dev_lower_90 = unlist(lapply(dev_all, function(x) x[4,])),
    dev_upper_90 = unlist(lapply(dev_all, function(x) x[5,])),
    dev_lower_80 = unlist(lapply(dev_all, function(x) x[6,])),
    dev_upper_80 = unlist(lapply(dev_all, function(x) x[7,]))
  )
}

# Updated summary function for multitime effects
summarize_multitime_effects <- function(
    int_effects_A, int_effects_B, int_effects_4,    # interaction effects
    int_dev_A, int_dev_B, int_dev_4,               # interaction deviations
    before_effects_A, before_effects_B, before_effects_4,  # before effects
    before_dev_A, before_dev_B, before_dev_4,      # before effect deviations
    after_effects_A, after_effects_B, after_effects_4,    # after effects
    after_dev_A, after_dev_B, after_dev_4,         # after effect deviations
    model_names) {
  
  n_models <- length(model_names)
  
  # Combine interaction effects
  int_m1 <- cbind(
    get_posterior_summary(int_effects_A),
    get_posterior_summary(int_effects_B),
    get_posterior_summary(int_effects_4)
  )
  
  # Combine interaction deviations
  int_dev_m1 <- cbind(
    get_posterior_summary(int_dev_A),
    get_posterior_summary(int_dev_B),
    get_posterior_summary(int_dev_4)
  )
  
  # Combine before effects
  before_m1 <- cbind(
    get_posterior_summary(before_effects_A),
    get_posterior_summary(before_effects_B),
    get_posterior_summary(before_effects_4)
  )
  
  # Combine before effect deviations
  before_dev_m1 <- cbind(
    get_posterior_summary(before_dev_A),
    get_posterior_summary(before_dev_B),
    get_posterior_summary(before_dev_4)
  )
  
  # Combine after effects
  after_m1 <- cbind(
    get_posterior_summary(after_effects_A),
    get_posterior_summary(after_effects_B),
    get_posterior_summary(after_effects_4)
  )
  
  # Combine after effect deviations
  after_dev_m1 <- cbind(
    get_posterior_summary(after_dev_A),
    get_posterior_summary(after_dev_B),
    get_posterior_summary(after_dev_4)
  )
  
  # Create data frame
  data.frame(
    model = rep(model_names, each = 3),
    experiment = rep(c("3A", "3B", "4"), n_models),
    # Before effects
    before_mean = c(before_m1[1,]),
    before_lower_95 = c(before_m1[2,]),
    before_upper_95 = c(before_m1[3,]),
    before_lower_90 = c(before_m1[4,]),
    before_upper_90 = c(before_m1[5,]),
    before_lower_80 = c(before_m1[6,]),
    before_upper_80 = c(before_m1[7,]),
    # Before effect deviations
    before_dev_mean = c(before_dev_m1[1,]),
    before_dev_lower_95 = c(before_dev_m1[2,]),
    before_dev_upper_95 = c(before_dev_m1[3,]),
    before_dev_lower_90 = c(before_dev_m1[4,]),
    before_dev_upper_90 = c(before_dev_m1[5,]),
    before_dev_lower_80 = c(before_dev_m1[6,]),
    before_dev_upper_80 = c(before_dev_m1[7,]),
    # After effects
    after_mean = c(after_m1[1,]),
    after_lower_95 = c(after_m1[2,]),
    after_upper_95 = c(after_m1[3,]),
    after_lower_90 = c(after_m1[4,]),
    after_upper_90 = c(after_m1[5,]),
    after_lower_80 = c(after_m1[6,]),
    after_upper_80 = c(after_m1[7,]),
    # After effect deviations
    after_dev_mean = c(after_dev_m1[1,]),
    after_dev_lower_95 = c(after_dev_m1[2,]),
    after_dev_upper_95 = c(after_dev_m1[3,]),
    after_dev_lower_90 = c(after_dev_m1[4,]),
    after_dev_upper_90 = c(after_dev_m1[5,]),
    after_dev_lower_80 = c(after_dev_m1[6,]),
    after_dev_upper_80 = c(after_dev_m1[7,]),
    # Interaction effects
    int_mean = c(int_m1[1,]),
    int_lower_95 = c(int_m1[2,]),
    int_upper_95 = c(int_m1[3,]),
    int_lower_90 = c(int_m1[4,]),
    int_upper_90 = c(int_m1[5,]),
    int_lower_80 = c(int_m1[6,]),
    int_upper_80 = c(int_m1[7,]),
    # Interaction deviations
    int_dev_mean = c(int_dev_m1[1,]),
    int_dev_lower_95 = c(int_dev_m1[2,]),
    int_dev_upper_95 = c(int_dev_m1[3,]),
    int_dev_lower_90 = c(int_dev_m1[4,]),
    int_dev_upper_90 = c(int_dev_m1[5,]),
    int_dev_lower_80 = c(int_dev_m1[6,]),
    int_dev_upper_80 = c(int_dev_m1[7,])
  )
}
# Function to calculate grand-mean + c1*contrast1 + c2*contrast2
int_eff <- function(s, base, c1, c2, v1, v2) {
  s[[base]] + v1 * s[[c1]] + v2 * s[[c2]]
}

#######################################
####### Load and process data #########
#######################################

# Load datasets for each experiment version and combine
exp_versions <- c("3A", "3B", "4")

choice_df_list <- list()
choice_df_mem_list <- list()

for (exp_version in exp_versions) {
  # Load data for this version
  choice_df_temp <- read.csv(file.path(dpath, paste0("choiceDataFull_", exp_version, ".csv")))
  choice_df_mem_temp <- read.csv(file.path(dpath, paste0("choiceDataMemValue_", exp_version, ".csv")))
  
  # Add exp_version column
  choice_df_temp$exp_version <- exp_version
  choice_df_mem_temp$exp_version <- exp_version
  
  # Clean data
  choice_df_temp <- choice_df_temp[choice_df_temp$rt!=0,]
  choice_df_mem_temp <- choice_df_mem_temp[choice_df_mem_temp$rt!=0,]
  choice_df_temp <- na.omit(choice_df_temp)
  choice_df_mem_temp <- na.omit(choice_df_mem_temp)
  
  # Keep only the columns that are actually used in the analysis
  choice_df_temp <- choice_df_temp[, c("correct","rt", "n_total_memories_shown", "n_remembered_shown", 
                                      "nback_performance", "wid", "exp_version", "multi_time")]
  choice_df_mem_temp <- choice_df_mem_temp[, c("correct","choice", "value", "recalled_mem_value", 
                                              "nback_performance", "wid", "rt", "exp_version", "multi_time")]
  
  # Add to lists
  choice_df_list[[exp_version]] <- choice_df_temp
  choice_df_mem_list[[exp_version]] <- choice_df_mem_temp
}

# Combine all versions
choice_df <- do.call(rbind, choice_df_list)
choice_df_mem <- do.call(rbind, choice_df_mem_list)

# Convert exp_version to factor with meaningful levels
choice_df$exp_version <- factor(choice_df$exp_version, levels = c("3A", "3B", "4"))
choice_df_mem$exp_version <- factor(choice_df_mem$exp_version, levels = c("3A", "3B", "4"))
choice_df$multi_time <- factor(choice_df$multi_time, levels = c("Before", "After"))
choice_df_mem$multi_time <- factor(choice_df_mem$multi_time, levels = c("Before", "After"))

# Convert to factor ― keep the same order of levels
choice_df$exp_version      <- factor(choice_df$exp_version,      levels = c("3A", "3B", "4"))
choice_df_mem$exp_version  <- factor(choice_df_mem$exp_version,  levels = c("3A", "3B", "4"))

# Apply effect (sum-to-zero) coding
sum_contr <- contr.sum(3)
colnames(sum_contr) <- c("exp_c1", "exp_c2")
# assign contrasts to each dataframe
contrasts(choice_df$exp_version)     <- sum_contr
contrasts(choice_df_mem$exp_version) <- sum_contr
before_df <- choice_df_mem %>% filter(multi_time == "Before")
after_df  <- choice_df_mem %>% filter(multi_time == "After")
contrasts(before_df$exp_version) <- sum_contr
contrasts(after_df$exp_version)  <- sum_contr

# Scale predictors within each experiment
choice_df_mem$value_z <- NA
choice_df_mem$recalled_mem_value_z <- NA
choice_df_mem$nback_performance_z <- NA
choice_df$nback_performance_z <- NA

# Separate into Before and After conditions for choice comparison
before_df = choice_df_mem[choice_df_mem$multi_time == "Before",]
after_df = choice_df_mem[choice_df_mem$multi_time == "After",]

for (exp in c("3A", "3B", "4")) {
  # Get indices for this experiment
  exp_indices_mem <- which(choice_df_mem$exp_version == exp)
  exp_indices <- which(choice_df$exp_version == exp)
  
  # Scale within this experiment
  choice_df_mem$value_z[exp_indices_mem] <- scale(choice_df_mem$value[exp_indices_mem])
  choice_df_mem$recalled_mem_value_z[exp_indices_mem] <- scale(choice_df_mem$recalled_mem_value[exp_indices_mem])
  choice_df_mem$nback_performance_z[exp_indices_mem] <- scale(choice_df_mem$nback_performance[exp_indices_mem])
  choice_df$nback_performance_z[exp_indices] <- scale(choice_df$nback_performance[exp_indices])

  before_exp_indices = which(before_df$exp_version == exp)
  before_df$value_z[before_exp_indices] = scale(before_df$value[before_exp_indices])
  before_df$recalled_mem_value_z[before_exp_indices] = scale(before_df$recalled_mem_value[before_exp_indices])
  before_df$nback_performance[before_exp_indices] = scale(before_df$nback_performance[before_exp_indices])
  after_exp_indices = which(after_df$exp_version == exp)
  after_df$value_z[after_exp_indices] = scale(after_df$value[after_exp_indices])
  after_df$recalled_mem_value_z[after_exp_indices] = scale(after_df$recalled_mem_value[after_exp_indices])
  after_df$nback_performance[after_exp_indices] = scale(after_df$nback_performance[after_exp_indices])
}

#######################################
#### Run Choice models and compare ####
#######################################

# Models with experiment effects (for parameter estimation)
fit1_before_kfold <- brm(choice ~ value_z * exp_version + (value_z | wid), 
            family = bernoulli,
            data = before_df,
            chains = N_CHAINS,
            iter = N_ITER)

fit2_before_kfold <- brm(choice ~ recalled_mem_value_z * exp_version + (recalled_mem_value_z | wid), 
            family = bernoulli,
            data = before_df,
            chains = N_CHAINS,
            iter = N_ITER)

fit1_after_kfold <- brm(choice ~ value_z * exp_version + (value_z | wid), 
            family = bernoulli,
            data = after_df,
            chains = N_CHAINS,
            iter = N_ITER)

fit2_after_kfold <- brm(choice ~ recalled_mem_value_z * exp_version + (recalled_mem_value_z | wid), 
            family = bernoulli,
            data = after_df,
            chains = N_CHAINS,
            iter = N_ITER)
 

# Extract and save fixed effects for choice models
choice_effects <- data.frame(
  model = c(rep("before_true", nrow(posterior_summary(fit1_before_kfold))),
            rep("before_memory", nrow(posterior_summary(fit2_before_kfold))),
            rep("after_true", nrow(posterior_summary(fit1_after_kfold))),
            rep("after_memory", nrow(posterior_summary(fit2_after_kfold)))),
  rbind(
    get_credible_intervals(fit1_before_kfold),
    get_credible_intervals(fit2_before_kfold),
    get_credible_intervals(fit1_after_kfold),
    get_credible_intervals(fit2_after_kfold)
  )
)
write.csv(choice_effects, 
          file.path(opath, paste0("choice_effects_manip_pooled.csv")), 
          row.names = FALSE)

# Add k-fold cross-validation
fit1_before_kfold = add_criterion(fit1_before_kfold, "kfold")
fit2_before_kfold = add_criterion(fit2_before_kfold, "kfold")
fit1_after_kfold = add_criterion(fit1_after_kfold, "kfold")
fit2_after_kfold = add_criterion(fit2_after_kfold, "kfold")
  
# Use k-fold models for ELPD comparison
elpd_true_before = fit1_before_kfold$criteria$kfold$pointwise[, 1]
elpd_mem_before = fit2_before_kfold$criteria$kfold$pointwise[, 1]
elpd_true_after = fit1_after_kfold$criteria$kfold$pointwise[, 1]
elpd_mem_after = fit2_after_kfold$criteria$kfold$pointwise[, 1]
  
# Calculate overall ELPD difference
elpd_diff_before = (elpd_mem_before - elpd_true_before)
elpd_se_before = sqrt(length(elpd_diff_before)) * sd(elpd_diff_before)
elpd_diff_after = (elpd_mem_after - elpd_true_after)
elpd_se_after = sqrt(length(elpd_diff_after)) * sd(elpd_diff_after)
  
# Create test data for mixed model
elpd_test_data_before <- data.frame(
    elpd_diff_before = elpd_diff_before,
    experiment = before_df$exp_version,
    wid = before_df$wid,
    multi_time = before_df$multi_time
)

elpd_test_data_after <- data.frame(
    elpd_diff_after = elpd_diff_after,
    experiment = after_df$exp_version,
    wid = after_df$wid,
    multi_time = after_df$multi_time
)
  
# Compile mixed model
elpd_mixed_model_before <- brm(elpd_diff_before ~ 1 + (1 | wid), 
                        data = elpd_test_data_before,
                        chains = N_CHAINS,
                        iter = N_ITER)

elpd_mixed_model_after <- brm(elpd_diff_after ~ 1 + (1 | wid), 
                        data = elpd_test_data_after,
                        chains = N_CHAINS,
                        iter = N_ITER)

# Combine data for before vs after comparison
elpd_test_data_combined <- rbind(
  data.frame(
    elpd_diff = elpd_diff_before,
    experiment = before_df$exp_version,
    timepoint = "before", 
    wid = elpd_test_data_before$wid
  ),
  data.frame(
    elpd_diff = elpd_diff_after,
    experiment = after_df$exp_version,
    timepoint = "after",
    wid = elpd_test_data_after$wid
  )
)

# Compare before vs after
elpd_mixed_model_comparison <- brm(elpd_diff ~ timepoint + (timepoint | wid),
                                 data = elpd_test_data_combined,
                                 chains = N_CHAINS,
                                 iter = N_ITER)

# Check significance using credible intervals
elpd_mixed_summary_before <- posterior_summary(elpd_mixed_model_before)
elpd_mixed_summary_after <- posterior_summary(elpd_mixed_model_after)
elpd_mixed_summary_comparison <- posterior_summary(elpd_mixed_model_comparison)

# Save the final results
elpd_df_out <- data.frame(
  "elpd_diff" = c(sum(elpd_diff_before), sum(elpd_diff_after)),
  "se_diff" = c(elpd_se_before, elpd_se_after),
  "timepoint" = c("Before", "After")
)

elpd_mixed_results_comparison <- data.frame(
  parameter = rownames(elpd_mixed_summary_comparison),
  estimate = elpd_mixed_summary_comparison[, "Estimate"],
  lower = elpd_mixed_summary_comparison[, "Q2.5"],
  upper = elpd_mixed_summary_comparison[, "Q97.5"]
)

write.csv(elpd_mixed_results_comparison, file.path(opath, "choice_elpd_mixed_model_test_manip_pooled.csv"), row.names = FALSE)
write.csv(elpd_df_out, file.path(opath, "choice_elpd_diff_manip_pooled.csv"))

#######################################
###### Run RT models and compare ######
#######################################

# Test number of total memories
fit3 <- brm(rt ~ n_total_memories_shown * exp_version * multi_time + nback_performance_z + (n_total_memories_shown * multi_time | wid),
            family = shifted_lognormal,
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Test number of relevant memories
fit4 <- brm(rt ~ n_remembered_shown * exp_version * multi_time + nback_performance_z + (n_remembered_shown * multi_time | wid),
            family = shifted_lognormal,
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Extract and save fixed effects for RT models
rt_effects <- data.frame(
  model = c(rep("rel_only", nrow(posterior_summary(fit4))),
            rep("memory", nrow(posterior_summary(fit3)))),
  rbind(
    get_credible_intervals(fit4),
    get_credible_intervals(fit3)
  ))

write.csv(rt_effects,
          file.path(opath, paste0("rt_effects_pooled_within.csv")),
          row.names = FALSE)

###################################################
###### Run within-subjs RT models and compare #####
###################################################

# Run RT models with experiment version interactions

# Test number of total memories
choice_df <- choice_df %>%
  group_by(wid) %>%                 
  mutate(n_total_mem_mean = mean(n_total_memories_shown),    # between-participant
         n_total_mem_dev  = n_total_memories_shown - n_total_mem_mean) %>%   # within-participant
  ungroup() %>%
  mutate(n_total_mem_mean = n_total_mem_mean - mean(n_total_mem_mean))  # center at grand mean
fit3 <- brm(rt ~ n_total_mem_dev * exp_version * multi_time + n_total_mem_mean * exp_version * multi_time + nback_performance_z + (n_total_mem_dev * multi_time | wid), 
                 family = shifted_lognormal, 
                 data = choice_df,
                 chains = N_CHAINS,
                 iter = N_ITER)

# Test number of relevant memories
choice_df <- choice_df %>%
  group_by(wid) %>%                 
  mutate(n_rel_mem_mean = mean(n_remembered_shown),    # between-participant
         n_rel_mem_dev  = n_remembered_shown - n_rel_mem_mean) %>%   # within-participant
  ungroup() %>%
  mutate(n_rel_mem_mean = n_rel_mem_mean - mean(n_rel_mem_mean))  # center at grand mean
fit4 <- brm(rt ~ n_rel_mem_dev * exp_version * multi_time + n_rel_mem_mean * exp_version * multi_time + nback_performance_z + (n_rel_mem_dev * multi_time | wid), 
            family = shifted_lognormal, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Extract and save fixed effects for RT models
rt_effects <- data.frame(
  model = c(rep("rel_only", nrow(posterior_summary(fit4))),
            rep("memory", nrow(posterior_summary(fit3)))),
  rbind(
    get_credible_intervals(fit4),
    get_credible_intervals(fit3)
  )
)
write.csv(rt_effects, 
          file.path(opath, paste0("rt_effects_manip_pooled_within.csv")), 
          row.names = FALSE)

# Get posterior samples for RT models
post_samples_fit3 <- posterior_samples(fit3)
post_samples_fit4 <- posterior_samples(fit4)

# Interaction effects (memory × multi_time)
base_f3_int <- "b_n_total_mem_dev:multi_timeAfter"
c1_f3_int   <- "b_n_total_mem_dev:exp_versionexp_c1:multi_timeAfter"
c2_f3_int   <- "b_n_total_mem_dev:exp_versionexp_c2:multi_timeAfter"

base_f4_int <- "b_n_rel_mem_dev:multi_timeAfter"
c1_f4_int   <- "b_n_rel_mem_dev:exp_versionexp_c1:multi_timeAfter"
c2_f4_int   <- "b_n_rel_mem_dev:exp_versionexp_c2:multi_timeAfter"

# Calculate interaction effects for each experiment
eff3A_f3_int <- int_eff(post_samples_fit3, base_f3_int, c1_f3_int, c2_f3_int,  1,  0)
eff3B_f3_int <- int_eff(post_samples_fit3, base_f3_int, c1_f3_int, c2_f3_int,  0,  1)
eff4_f3_int  <- int_eff(post_samples_fit3, base_f3_int, c1_f3_int, c2_f3_int, -1, -1)

eff3A_f4_int <- int_eff(post_samples_fit4, base_f4_int, c1_f4_int, c2_f4_int,  1,  0)
eff3B_f4_int <- int_eff(post_samples_fit4, base_f4_int, c1_f4_int, c2_f4_int,  0,  1)
eff4_f4_int  <- int_eff(post_samples_fit4, base_f4_int, c1_f4_int, c2_f4_int, -1, -1)

# Calculate interaction deviations
dev3A_f3_int <- post_samples_fit3[[c1_f3_int]]
dev3B_f3_int <- post_samples_fit3[[c2_f3_int]]
dev4_f3_int <- post_samples_fit3[[c1_f3_int]] + post_samples_fit3[[c2_f3_int]]

dev3A_f4_int <- post_samples_fit4[[c1_f4_int]]
dev3B_f4_int <- post_samples_fit4[[c2_f4_int]]
dev4_f4_int <- post_samples_fit4[[c1_f4_int]] + post_samples_fit4[[c2_f4_int]]

# Before effects
base_f3_before <- "b_n_total_mem_dev"
c1_f3_before   <- "b_n_total_mem_dev:exp_versionexp_c1"
c2_f3_before   <- "b_n_total_mem_dev:exp_versionexp_c2"

base_f4_before <- "b_n_rel_mem_dev"
c1_f4_before   <- "b_n_rel_mem_dev:exp_versionexp_c1"
c2_f4_before   <- "b_n_rel_mem_dev:exp_versionexp_c2"

# Calculate before effects for each experiment
eff3A_f3_before <- int_eff(post_samples_fit3, base_f3_before, c1_f3_before, c2_f3_before,  1,  0)
eff3B_f3_before <- int_eff(post_samples_fit3, base_f3_before, c1_f3_before, c2_f3_before,  0,  1)
eff4_f3_before  <- int_eff(post_samples_fit3, base_f3_before, c1_f3_before, c2_f3_before, -1, -1)

eff3A_f4_before <- int_eff(post_samples_fit4, base_f4_before, c1_f4_before, c2_f4_before,  1,  0)
eff3B_f4_before <- int_eff(post_samples_fit4, base_f4_before, c1_f4_before, c2_f4_before,  0,  1)
eff4_f4_before  <- int_eff(post_samples_fit4, base_f4_before, c1_f4_before, c2_f4_before, -1, -1)

# After effects (before + interaction)
eff3A_f3_after <- eff3A_f3_before + eff3A_f3_int
eff3B_f3_after <- eff3B_f3_before + eff3B_f3_int
eff4_f3_after  <- eff4_f3_before + eff4_f3_int

eff3A_f4_after <- eff3A_f4_before + eff3A_f4_int
eff3B_f4_after <- eff3B_f4_before + eff3B_f4_int
eff4_f4_after  <- eff4_f4_before + eff4_f4_int

# Now the corrected function call with all required arguments:
rt_effects_by_exp <- summarize_multitime_effects(
  # Interaction effects
  rbind(eff3A_f3_int, eff3A_f4_int),
  rbind(eff3B_f3_int, eff3B_f4_int),
  rbind(eff4_f3_int, eff4_f4_int),
  # Interaction deviations
  rbind(dev3A_f3_int, dev3A_f4_int),
  rbind(dev3B_f3_int, dev3B_f4_int),
  rbind(dev4_f3_int, dev4_f4_int),
  # Before effects
  rbind(eff3A_f3_before, eff3A_f4_before),
  rbind(eff3B_f3_before, eff3B_f4_before),
  rbind(eff4_f3_before, eff4_f4_before),
  # Before effect deviations
  rbind(dev3A_f3_before, dev3A_f4_before),
  rbind(dev3B_f3_before, dev3B_f4_before),
  rbind(dev4_f3_before, dev4_f4_before),
  # After effects
  rbind(eff3A_f3_after, eff3A_f4_after),
  rbind(eff3B_f3_after, eff3B_f4_after),
  rbind(eff4_f3_after, eff4_f4_after),
  # After effect deviations
  rbind(dev3A_f3_after, dev3A_f4_after),
  rbind(dev3B_f3_after, dev3B_f4_after),
  rbind(dev4_f3_after, dev4_f4_after),
  
  model_names = c("memory", "rel_only"))

write.csv(rt_effects_by_exp,
          file.path(opath, "rt_effects_posterior_pooled_manip_within.csv"),
          row.names = FALSE)

#######################################
########### Run acc/rt model ##########
#######################################

choice_df$log_rt <- log(choice_df$rt)
fit5 <- brm(correct ~ log_rt * exp_version * multi_time + nback_performance_z + (log_rt * multi_time | wid), 
            family = bernoulli,
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Extract and save fixed effects for RT models
accrt_effects <- data.frame(
  model = c(rep("speed_acc", nrow(posterior_summary(fit5)))),
  rbind(
    get_credible_intervals(fit5)
  )
)
write.csv(accrt_effects, 
          file.path(opath, paste0("speedacc_effects_pooled_manip.csv")), 
          row.names = FALSE)

# Get posterior samples
post_samples_fit5 <- posterior_samples(fit5)

# Interaction effects (log_rt × multi_time)
base_f5_int <- "b_log_rt:multi_timeAfter"
c1_f5_int   <- "b_log_rt:exp_versionexp_c1:multi_timeAfter"
c2_f5_int   <- "b_log_rt:exp_versionexp_c2:multi_timeAfter"

# Calculate interaction effects for each experiment
eff3A_f5_int <- int_eff(post_samples_fit5, base_f5_int, c1_f5_int, c2_f5_int,  1,  0)
eff3B_f5_int <- int_eff(post_samples_fit5, base_f5_int, c1_f5_int, c2_f5_int,  0,  1)
eff4_f5_int  <- int_eff(post_samples_fit5, base_f5_int, c1_f5_int, c2_f5_int, -1, -1)

# Calculate interaction deviations
dev3A_f5_int <- post_samples_fit5[[c1_f5_int]]
dev3B_f5_int <- post_samples_fit5[[c2_f5_int]]
dev4_f5_int <- post_samples_fit5[[c1_f5_int]] + post_samples_fit5[[c2_f5_int]]

# Before effects (main effect of log_rt)
base_f5_before <- "b_log_rt"
c1_f5_before   <- "b_log_rt:exp_versionexp_c1"
c2_f5_before   <- "b_log_rt:exp_versionexp_c2"

# Calculate before effects for each experiment
eff3A_f5_before <- int_eff(post_samples_fit5, base_f5_before, c1_f5_before, c2_f5_before,  1,  0)
eff3B_f5_before <- int_eff(post_samples_fit5, base_f5_before, c1_f5_before, c2_f5_before,  0,  1)
eff4_f5_before  <- int_eff(post_samples_fit5, base_f5_before, c1_f5_before, c2_f5_before, -1, -1)

# After effects (before + interaction)
eff3A_f5_after <- eff3A_f5_before + eff3A_f5_int
eff3B_f5_after <- eff3B_f5_before + eff3B_f5_int
eff4_f5_after  <- eff4_f5_before + eff4_f5_int

# Summarize
accrt_effects_by_exp <- summarize_multitime_effects(
  eff3A_f5_int, eff3B_f5_int, eff4_f5_int,          # interaction effects
  dev3A_f5_int, dev3B_f5_int,                       # interaction deviations
  eff3A_f5_before, eff3B_f5_before, eff4_f5_before, # before effects
  eff3A_f5_after, eff3B_f5_after, eff4_f5_after,    # after effects
  model_names = "speed_acc"
)

write.csv(accrt_effects_by_exp,
          file.path(opath, "speedacc_posterior_pooled_manip.csv"),
          row.names = FALSE)
