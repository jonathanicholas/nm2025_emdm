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

# Load and process data
choice_df <- read.csv(file.path(dpath, paste0("choiceDataFull_", exp_version, ".csv")))
choice_df_mem <- read.csv(file.path(dpath, paste0("choiceDataMemValue_", exp_version, ".csv")))
choice_df = choice_df[choice_df$rt!=0,]
choice_df_mem = choice_df_mem[choice_df_mem$rt!=0,]
choice_df = na.omit(choice_df)
choice_df_mem = na.omit(choice_df_mem)
choice_df$nback_performance_z = scale(choice_df$nback_performance)
choice_df <- choice_df %>%
  group_by(wid) %>%
  mutate(participant_accuracy = mean(correct, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(
    median_accuracy = median(participant_accuracy),
    high_accuracy = ifelse(participant_accuracy > median_accuracy, 1, 0),
    participant_accuracy_z = as.numeric(scale(participant_accuracy))
  )
choice_df$n_irrelevant_recalled = choice_df$n_total_memories - choice_df$recalled_n_images_with_option

#######################################
#### Run choice models and compare ####
#######################################

# Scale predictors if needed
if (!"value_z" %in% names(choice_df_mem)) {
  choice_df_mem$value_z = scale(choice_df_mem$true_offer_value)
}
if (!"recalled_mem_value_z" %in% names(choice_df_mem)) {
  choice_df_mem$recalled_mem_value_z = scale(choice_df_mem$recalled_offer_value)
}

# Run choice models with specified chains and iterations
fit1 <- brm(choice ~ value_z + (value_z | wid),
                   family = bernoulli,
                   data = choice_df_mem,
                   chains = N_CHAINS,
                   iter = N_ITER)

fit2 <- brm(choice ~ recalled_mem_value_z + (recalled_mem_value_z | wid),
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
          file.path(opath, paste0("choice_effects_", exp_version, ".csv")),
          row.names = FALSE)

# Create folds and run kfold cross validation (10-fold)
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

# Write output files with version in filename
write.csv(elpd_df_out, file.path(opath, paste0("choice_elpd_diff_", exp_version, ".csv")))
write.csv(elpd_raw, file.path(opath, paste0("choice_elpd_diff_raw_", exp_version, ".csv")))

#######################################
###### Run RT models and compare ######
#######################################

# RT models for total and relevant

# Run RT models with specified chains and iterations
fit3 <- brm(rt ~ n_total_memories + nback_performance_z + (1 | wid), 
            family = shifted_lognormal, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

fit4 <- brm(rt ~ recalled_n_images_with_option + nback_performance_z + (recalled_n_images_with_option | wid), 
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
          file.path(opath, paste0("rt_effects_", exp_version, ".csv")), 
          row.names = FALSE)

# RT models with accuracy included

# Run RT models with specified chains and iterations
fit3a <- brm(rt ~ n_irrelevant_recalled*participant_accuracy_z + nback_performance_z + (n_irrelevant_recalled | wid), 
             family = shifted_lognormal, 
             data = choice_df,
             chains = N_CHAINS,
             iter = N_ITER)

# Get posterior draws for coefficients
draws_irrelevant <- as_draws_df(fit3a)
write.csv(draws_irrelevant, "posterior_draws_irrelevant.csv", row.names = FALSE)

# Split data and do model comparison
median_acc <- median(choice_df$participant_accuracy)
choice_df$accuracy_group <- ifelse(choice_df$participant_accuracy >= median_acc, "high", "low")

low_acc_df <- subset(choice_df, accuracy_group == "low")  
high_acc_df <- subset(choice_df, accuracy_group == "high")

# Low accuracy group models
fit_low_total <- brm(rt ~ n_total_memories + nback_performance_z + 
                       (1 | wid),
                     family = shifted_lognormal,
                     data = low_acc_df,
                     chains = N_CHAINS,
                     iter = N_ITER)

fit_low_relevant <- brm(rt ~ recalled_n_images_with_option + nback_performance_z + 
                          (recalled_n_images_with_option | wid),
                        family = shifted_lognormal,
                        data = low_acc_df,
                        chains = N_CHAINS,
                        iter = N_ITER)

# High accuracy group models  
fit_high_total <- brm(rt ~ n_total_memories + nback_performance_z + 
                        (1 | wid),
                      family = shifted_lognormal,
                      data = high_acc_df,
                      chains = N_CHAINS,
                      iter = N_ITER)

fit_high_relevant <- brm(rt ~ recalled_n_images_with_option + nback_performance_z + 
                           (recalled_n_images_with_option | wid),
                         family = shifted_lognormal,
                         data = high_acc_df,
                         chains = N_CHAINS,
                         iter = N_ITER)

# Add kfold criterion to all models
fit_low_total <- add_criterion(fit_low_total, "kfold")
fit_low_relevant <- add_criterion(fit_low_relevant, "kfold")
fit_high_total <- add_criterion(fit_high_total, "kfold")
fit_high_relevant <- add_criterion(fit_high_relevant, "kfold")

# LOW ACCURACY GROUP COMPARISON
# Extract ELPD estimates
elpd_low_total <- fit_low_total$criteria$kfold$pointwise[,1]
elpd_low_relevant <- fit_low_relevant$criteria$kfold$pointwise[,1]
elpd_diff_low <- (elpd_low_relevant - elpd_low_total)
elpd_se_low <- sqrt(length(elpd_diff_low)) * sd(elpd_diff_low)

elpd_df_low <- data.frame("elpd_diff" = c(sum(elpd_diff_low)),
                          "se_diff" = c(elpd_se_low),
                          "group" = "low_accuracy")
elpd_raw_low <- data.frame(elpd_total = elpd_low_total, 
                           elpd_relevant = elpd_low_relevant, 
                           elpd_diff = elpd_diff_low)

# HIGH ACCURACY GROUP COMPARISON  
# Extract ELPD estimates
elpd_high_total <- fit_high_total$criteria$kfold$pointwise[,1]
elpd_high_relevant <- fit_high_relevant$criteria$kfold$pointwise[,1]
elpd_diff_high <- (elpd_high_relevant - elpd_high_total)
elpd_se_high <- sqrt(length(elpd_diff_high)) * sd(elpd_diff_high)

elpd_df_high <- data.frame("elpd_diff" = c(sum(elpd_diff_high)),
                           "se_diff" = c(elpd_se_high),
                           "group" = "high_accuracy")
elpd_raw_high <- data.frame(elpd_total = elpd_high_total, 
                            elpd_relevant = elpd_high_relevant, 
                            elpd_diff = elpd_diff_high)

# Combine results
elpd_df_combined <- rbind(elpd_df_low, elpd_df_high)
print(elpd_df_combined)

# Get coefficient summaries
coef_low_total <- fixef(fit_low_total)["n_total_memories", ]
coef_low_relevant <- fixef(fit_low_relevant)["recalled_n_images_with_option", ]
coef_high_total <- fixef(fit_high_total)["n_total_memories", ]
coef_high_relevant <- fixef(fit_high_relevant)["recalled_n_images_with_option", ]

# Create a summary table of all coefficients
coef_summary <- data.frame(
  group = rep(c("low_accuracy", "high_accuracy"), each = 2),
  model = rep(c("total", "relevant"), 2),
  variable = c("n_total_memories", "recalled_n_images_with_option",
               "n_total_memories", "recalled_n_images_with_option"),
  estimate = c(coef_low_total["Estimate"], coef_low_relevant["Estimate"],
               coef_high_total["Estimate"], coef_high_relevant["Estimate"]),
  se = c(coef_low_total["Est.Error"], coef_low_relevant["Est.Error"],
         coef_high_total["Est.Error"], coef_high_relevant["Est.Error"]),
  lower_ci = c(coef_low_total["Q2.5"], coef_low_relevant["Q2.5"],
               coef_high_total["Q2.5"], coef_high_relevant["Q2.5"]),
  upper_ci = c(coef_low_total["Q97.5"], coef_low_relevant["Q97.5"],
               coef_high_total["Q97.5"], coef_high_relevant["Q97.5"]))
print(coef_summary)

###################################################
###### Run within-subjs RT models and compare #####
###################################################

# Test number of relevant memories
choice_df <- choice_df %>%
  group_by(wid) %>%                 
  mutate(n_rel_mem_mean = mean(recalled_n_images_with_option),    # between-participant
         n_rel_mem_dev  = recalled_n_images_with_option - n_rel_mem_mean) %>%   # within-participant
  ungroup() %>%
  mutate(n_rel_mem_mean = n_rel_mem_mean - mean(n_rel_mem_mean))  # center at grand mean

fit4 <- brm(rt ~ n_rel_mem_mean + n_rel_mem_dev + nback_performance_z + (n_rel_mem_dev | wid), 
            family = shifted_lognormal, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Extract and save fixed effects for RT models
rt_effects <- data.frame(
  model = c(rep("rel_only", nrow(posterior_summary(fit4)))),
  rbind(
    get_credible_intervals(fit4),
  )
)
write.csv(rt_effects, 
          file.path(opath, paste0("rt_effects_", exp_version, "_within.csv")), 
          row.names = FALSE)

#######################################
########### Run acc/rt model ##########
#######################################

choice_df$log_rt <- log(choice_df$rt)
fit5 <- brm(correct ~ log_rt + nback_performance_z + (log_rt | wid), 
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
          file.path(opath, paste0("speedacc_effects_", exp_version, ".csv")), 
          row.names = FALSE)

# SAVE EVERYTHING

# Extract and save fixed effects for all RT models
rt_effects_fit3 <- data.frame(
  model = rep("fit3_total", nrow(posterior_summary(fit3))),
  get_credible_intervals(fit3)
)

rt_effects_fit3a <- data.frame(
  model = rep("fit3_irrelevant", nrow(posterior_summary(fit3a))),
  get_credible_intervals(fit3a)
)

rt_effects_fit4 <- data.frame(
  model = rep("fit4_relevant", nrow(posterior_summary(fit4))),
  get_credible_intervals(fit4)
)

rt_effects_low_total <- data.frame(
  model = rep("fit_low_total", nrow(posterior_summary(fit_low_total))),
  get_credible_intervals(fit_low_total)
)

rt_effects_high_total <- data.frame(
  model = rep("fit_high_total", nrow(posterior_summary(fit_high_total))),
  get_credible_intervals(fit_high_total)
)

rt_effects_low_relevant <- data.frame(
  model = rep("fit_low_relevant", nrow(posterior_summary(fit_low_relevant))),
  get_credible_intervals(fit_low_relevant)
)

rt_effects_high_relevant <- data.frame(
  model = rep("fit_high_relevant", nrow(posterior_summary(fit_high_relevant))),
  get_credible_intervals(fit_high_relevant)
)

# Combine all effects
all_rt_effects <- rbind(
  rt_effects_fit3,
  rt_effects_fit3a,
  rt_effects_fit4,
  rt_effects_low_total,
  rt_effects_high_total,
  rt_effects_low_relevant,
  rt_effects_high_relevant
)

# Write combined effects file
write.csv(all_rt_effects, 
          file.path(opath, paste0("rt_effects_indivdiff_models_", exp_version, ".csv")), 
          row.names = FALSE)

# Save ELPD differences for the accuracy group comparisons
# Write ELPD difference files
write.csv(elpd_df_low, 
          file.path(opath, paste0("rt_elpd_diff_low_accuracy_", exp_version, ".csv")), 
          row.names = FALSE)

write.csv(elpd_df_high, 
          file.path(opath, paste0("rt_elpd_diff_high_accuracy_", exp_version, ".csv")), 
          row.names = FALSE)

# Save raw ELPD data
write.csv(elpd_raw_low, 
          file.path(opath, paste0("rt_elpd_raw_low_accuracy_", exp_version, ".csv")), 
          row.names = FALSE)

write.csv(elpd_raw_high, 
          file.path(opath, paste0("rt_elpd_raw_high_accuracy_", exp_version, ".csv")), 
          row.names = FALSE)
