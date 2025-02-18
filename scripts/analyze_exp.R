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

# Scale predictors if needed
if (!"value_z" %in% names(choice_df_mem)) {
  choice_df_mem$value_z = scale(choice_df_mem$value)
}
if (!"recalled_mem_value_z" %in% names(choice_df_mem)) {
  choice_df_mem$recalled_mem_value_z = scale(choice_df_mem$recalled_mem_value)
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

# Fit to full data bc possible for beta difference comparison analysis
choice_df$value_z = scale(choice_df$value)
fit1a <- brm(choice ~ value_z + (value_z | wid), 
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
          file.path(opath, paste0("choice_effects_", exp_version, ".csv")), 
          row.names = FALSE)


# Create folds and run kfold cross validation (10-fold)
fit1 = add_criterion(fit1, "kfold")
fit2 = add_criterion(fit2, "kfold")
# Extract ELPD estimates
elpd_true = fit1$criteria$kfold$pointwise[,1]
elpd_mem = fit2$criteria$kfold$pointwise[,1]

elpd_diff = (elpd_mem - elpd_true) # * -1
elpd_se = sqrt(length(elpd_diff)) * sd(elpd_diff) # * -1

elpd_df_out <- data.frame("elpd_diff" = c(sum(elpd_diff)),
                          "se_diff" = c(elpd_se))
elpd_raw <- data.frame(elpd_mem = elpd_mem, elpd_true = elpd_true, elpd_diff = elpd_diff)

# Write output files with version in filename
write.csv(elpd_df_out, file.path(opath, paste0("choice_elpd_diff_", exp_version, ".csv")))
write.csv(elpd_raw, file.path(opath, paste0("choice_elpd_diff_raw_", exp_version, ".csv")))

#######################################
###### Run RT models and compare ######
#######################################

# Run RT models with specified chains and iterations
fit3 <- brm(rt ~ n_total_memories_shown + (n_total_memories_shown | wid), 
                   family = shifted_lognormal, 
                   data = choice_df,
                   chains = N_CHAINS,
                   iter = N_ITER)

fit4 <- brm(rt ~ n_remembered_shown + (n_total_memories_shown | wid), 
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
