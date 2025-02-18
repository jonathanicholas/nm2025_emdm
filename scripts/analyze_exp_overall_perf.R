# Set parameters
N_CHAINS = 4  # Number of chains for MCMC sampling
N_ITER = 2000 # Number of iterations per chain

args = commandArgs(trailingOnly = TRUE)
dpath = args[1]
opath = args[2]
exp_version = args[3]

library(brms)
library(rstan)
library(dplyr)
options(mc.cores = parallel::detectCores())

# Define helper function for credible intervals
get_credible_intervals <- function(fit) {
  post_summary <- posterior_summary(fit, probs = c(0.25, 0.75, 0.1, 0.9, 0.025, 0.975))
  
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

# Load and prepare data
choice_df <- read.csv(file.path(dpath, paste0("choiceDataFull_", exp_version, ".csv")))

chance_level <- 0.5

fit_before <- brm(correct ~ 1 + (1 | wid), 
            family = bernoulli, 
            data = choice_df,
            chains = N_CHAINS,
            iter = N_ITER)

# Extract and save effects
effects <- data.frame(
  model = c(rep("before", nrow(posterior_summary(fit_before)))),
  rbind(get_credible_intervals(fit_before))
)

# Save results
write.csv(effects, 
          file.path(opath, paste0("performance_effects_", exp_version, ".csv")), 
          row.names = FALSE)