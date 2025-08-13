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
# Function to do the same from posterior samples
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
# Function to calculate grand-mean + c1*contrast1 + c2*contrast2
int_eff <- function(s, base, c1, c2, v1, v2) {
  s[[base]] + v1 * s[[c1]] + v2 * s[[c2]]
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
    experiment = rep(c("1A", "1B", "2"), n_models),
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

#######################################
####### Load and process data #########
#######################################

# Load datasets for each experiment version and combine
exp_versions <- c("1A", "1B", "2")

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
                                      "nback_performance", "wid", "exp_version")]
  choice_df_mem_temp <- choice_df_mem_temp[, c("correct","choice", "value", "recalled_mem_value", 
                                              "nback_performance", "wid", "rt", "exp_version")]
  
  # Add to lists
  choice_df_list[[exp_version]] <- choice_df_temp
  choice_df_mem_list[[exp_version]] <- choice_df_mem_temp
}

# Combine all versions
choice_df <- do.call(rbind, choice_df_list)
choice_df_mem <- do.call(rbind, choice_df_mem_list)

# Convert exp_version to factor with meaningful levels
choice_df$exp_version <- factor(choice_df$exp_version, levels = c("1A", "1B", "2"))
choice_df_mem$exp_version <- factor(choice_df_mem$exp_version, levels = c("1A", "1B", "2"))

# Scale predictors within each experiment
choice_df_mem$value_z <- NA
choice_df_mem$recalled_mem_value_z <- NA
choice_df_mem$nback_performance_z <- NA
choice_df$nback_performance_z <- NA

for (exp in c("1A", "1B", "2")) {
  # Get indices for this experiment
  exp_indices_mem <- which(choice_df_mem$exp_version == exp)
  exp_indices <- which(choice_df$exp_version == exp)
  
  # Scale within this experiment
  choice_df_mem$value_z[exp_indices_mem] <- scale(choice_df_mem$value[exp_indices_mem])
  choice_df_mem$recalled_mem_value_z[exp_indices_mem] <- scale(choice_df_mem$recalled_mem_value[exp_indices_mem])
  choice_df_mem$nback_performance_z[exp_indices_mem] <- scale(choice_df_mem$nback_performance[exp_indices_mem])
  choice_df$nback_performance_z[exp_indices] <- scale(choice_df$nback_performance[exp_indices])
}

# Apply effect (sum-to-zero) coding
sum_contr <- contr.sum(3)
colnames(sum_contr) <- c("exp_c1", "exp_c2")
# assign contrasts to each dataframe
contrasts(choice_df$exp_version)     <- sum_contr
contrasts(choice_df_mem$exp_version) <- sum_contr

#######################################
#### Run choice models and compare ####
#######################################

# Models with experiment effects (for parameter estimation)
fit1 <- brm(choice ~ value_z * exp_version + (value_z | wid), 
             family = bernoulli,
             data = choice_df_mem,
             chains = N_CHAINS,
             iter = N_ITER)

fit2 <- brm(choice ~ recalled_mem_value_z * exp_version + (recalled_mem_value_z | wid), 
             family = bernoulli,
             data = choice_df_mem,
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
         file.path(opath, paste0("choice_effects_pooled.csv")), 
         row.names = FALSE)
  
# Add k-fold cross validation
fit1 = add_criterion(fit1, "kfold")
fit2 = add_criterion(fit2, "kfold")

# Use k-fold models for ELPD comparison
elpd_true = fit1$criteria$kfold$pointwise[,1]
elpd_mem = fit2$criteria$kfold$pointwise[,1]

# Calculate overall ELPD difference 
elpd_diff = (elpd_mem - elpd_true)
elpd_se = sqrt(length(elpd_diff)) * sd(elpd_diff)

# Create test data for mixed model
elpd_test_data <- data.frame(
  elpd_diff = elpd_diff,
  experiment = choice_df_mem$exp_version,
  wid = choice_df_mem$wid
)

# Assess difference from 0
elpd_mixed_model <- brm(elpd_diff ~ 1 + (1 | wid), 
                        data = elpd_test_data,
                        chains = N_CHAINS,
                        iter = N_ITER)

# Get model summary with both intercept (difference from 0) and experiment effects
elpd_mixed_summary <- posterior_summary(elpd_mixed_model)
  
# Save the final results
elpd_df_out <- data.frame("elpd_diff" = c(sum(elpd_diff)),
                          "se_diff" = c(elpd_se))
elpd_raw <- data.frame(elpd_mem = elpd_mem, elpd_true = elpd_true, elpd_diff = elpd_diff)

elpd_mixed_results <- data.frame(
  parameter = rownames(elpd_mixed_summary),
  estimate = elpd_mixed_summary[, "Estimate"],
  lower = elpd_mixed_summary[, "Q2.5"],
  upper = elpd_mixed_summary[, "Q97.5"]
)

write.csv(elpd_mixed_results, file.path(opath, "choice_elpd_mixed_model_test_pooled.csv"), row.names = FALSE)
write.csv(elpd_df_out, file.path(opath, "choice_elpd_diff_pooled.csv"))
write.csv(elpd_raw, file.path(opath, "choice_elpd_diff_raw_pooled.csv"))

#######################################
###### Run RT models and compare ######
#######################################

# Test number of total memories

fit3 <- brm(rt ~ n_total_memories_shown * exp_version + nback_performance_z + (n_total_memories_shown | wid), 
            family = shifted_lognormal, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Test number of relevant memories
fit4 <- brm(rt ~ n_remembered_shown * exp_version + nback_performance_z + (n_remembered_shown | wid), 
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
          file.path(opath, paste0("rt_effects_pooled.csv")), 
          row.names = FALSE)

###################################################
###### Run within-subjs RT models and compare #####
###################################################

# Test number of total memories
choice_df <- choice_df %>%
  group_by(wid) %>%                 
  mutate(n_total_mem_mean = mean(n_total_memories_shown),    # between-participant
         n_total_mem_dev  = n_total_memories_shown - n_total_mem_mean) %>%   # within-participant
  ungroup() %>%
  mutate(n_total_mem_mean = n_total_mem_mean - mean(n_total_mem_mean))  # center at grand mean
fit3 <- brm(rt ~ n_total_mem_dev * exp_version + n_total_mem_mean * exp_version + nback_performance_z + (n_total_mem_dev | wid), 
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
fit4 <- brm(rt ~ n_rel_mem_dev * exp_version + n_rel_mem_mean * exp_version + nback_performance_z + (n_rel_mem_dev | wid), 
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
          file.path(opath, paste0("rt_effects_pooled_within.csv")), 
          row.names = FALSE)

# Get posterior samples for RT models
post_samples_fit3 <- posterior_samples(fit3)
post_samples_fit4 <- posterior_samples(fit4)

# Calculate effects for each experiment
base_f3 <- "b_n_total_mem_dev"
c1_f3   <- "b_n_total_mem_dev:exp_versionexp_c1"
c2_f3   <- "b_n_total_mem_dev:exp_versionexp_c2"

base_f4 <- "b_n_rel_mem_dev"
c1_f4   <- "b_n_rel_mem_dev:exp_versionexp_c1"
c2_f4   <- "b_n_rel_mem_dev:exp_versionexp_c2"

eff1A_f3 <- int_eff(post_samples_fit3, base_f3, c1_f3, c2_f3,  1,  0)   # 1A = (1,0)
eff1B_f3 <- int_eff(post_samples_fit3, base_f3, c1_f3, c2_f3,  0,  1)   # 1B = (0,1)
eff2_f3  <- int_eff(post_samples_fit3, base_f3, c1_f3, c2_f3, -1, -1)   # 2  = (-1,-1)

eff1A_f4 <- int_eff(post_samples_fit4, base_f4, c1_f4, c2_f4,  1,  0)
eff1B_f4 <- int_eff(post_samples_fit4, base_f4, c1_f4, c2_f4,  0,  1)
eff2_f4  <- int_eff(post_samples_fit4, base_f4, c1_f4, c2_f4, -1, -1)

dev1A_f3 <- post_samples_fit3[[c1_f3]]
dev1B_f3 <- post_samples_fit3[[c2_f3]]
dev2_f3 <- post_samples_fit3[[c1_f3]] + post_samples_fit3[[c2_f3]]

dev1A_f4 <- post_samples_fit4[[c1_f4]]
dev1B_f4 <- post_samples_fit4[[c2_f4]]
dev2_f4 <- post_samples_fit4[[c1_f4]] + post_samples_fit4[[c2_f4]]

effects_A <- rbind(eff1A_f3, eff1A_f4)
effects_B <- rbind(eff1B_f3, eff1B_f4)
effects_2 <- rbind(eff2_f3, eff2_f4)
dev_A <- rbind(dev1A_f3, dev1A_f4)
dev_B <- rbind(dev1B_f3, dev1B_f4)
rt_effects_by_exp <- summarize_effects(
  effects_A, effects_B, effects_2,
  dev_A, dev_B,
  model_names = c("memory", "rel_only")
)

write.csv(rt_effects_by_exp,
          file.path(opath, "rt_effects_posterior_pooled_within.csv"),
          row.names = FALSE)

#######################################
########### Run acc/rt model ##########
#######################################

choice_df$log_rt <- log(choice_df$rt)
fit5 <- brm(correct ~ log_rt * exp_version + nback_performance_z + (log_rt | wid), 
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
post_samples_fit5 <- posterior_samples(fit5)

# Calculate effects for each experiment
base_f5 <- "b_log_rt"
c1_f5   <- "b_log_rt:exp_versionexp_c1"
c2_f5   <- "b_log_rt:exp_versionexp_c2"

# Calculate total effects
eff1A_f5 <- int_eff(post_samples_fit5, base_f5, c1_f5, c2_f5,  1,  0)   # 1A = (1,0)
eff1B_f5 <- int_eff(post_samples_fit5, base_f5, c1_f5, c2_f5,  0,  1)   # 1B = (0,1)
eff2_f5  <- int_eff(post_samples_fit5, base_f5, c1_f5, c2_f5, -1, -1)   # 2  = (-1,-1)

# Calculate deviations
dev1A_f5 <- post_samples_fit5[[c1_f5]]
dev1B_f5 <- post_samples_fit5[[c2_f5]]
dev2_f5 <- post_samples_fit5[[c1_f5]] + post_samples_fit5[[c2_f5]]

# Extract and save fixed effects
effects_A <- rbind(eff1A_f5)
effects_B <- rbind(eff1B_f5)
effects_2 <- rbind(eff2_f5)
dev_A <- rbind(dev1A_f5)
dev_B <- rbind(dev1B_f5)
accrt_effects_by_exp <- summarize_effects(
  eff1A_f5, eff1B_f5, eff2_f5,
  dev1A_f5, dev1B_f5,
  model_names = c("speed_acc")
)

write.csv(accrt_effects_by_exp,
          file.path(opath, "speedacc_posterior_pooled.csv"),
          row.names = FALSE)
write.csv(accrt_effects, 
          file.path(opath, paste0("speedacc_effects_pooled.csv")), 
          row.names = FALSE)
