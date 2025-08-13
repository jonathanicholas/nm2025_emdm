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

# Define helper function(s)
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

# Load value recall dataset and fit value recall model
value_df <- read.csv(file.path(dpath, paste0("valueRecallData_", exp_version, ".csv")))
value_df$correct <- ifelse(value_df$correct, 1, 0)

fit1 <- brm(correct ~ 1 + (1|wid), 
            family = gaussian, 
            data = value_df, 
            chains = N_CHAINS, 
            iter = N_ITER)

# Extract and save fixed effects for value memory model
value_effects <- data.frame(
  model = c(rep("value_recall", nrow(posterior_summary(fit1)))),
  rbind(get_credible_intervals(fit1))
)
write.csv(value_effects, 
          file.path(opath, paste0("value_effects_", exp_version, ".csv")), 
          row.names = FALSE)
