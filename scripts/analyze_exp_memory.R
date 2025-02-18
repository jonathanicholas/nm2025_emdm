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

# Load recall dataset and fit recall model
recall_df <- read.csv(file.path(dpath, paste0("recallData_", exp_version, ".csv")))
chance_recall <- 0.375
# Create the summary dataframe
recall_summary <- recall_df %>%
  mutate(recalled_numeric = ifelse(recalled == "True", 1, 0)) %>%
  group_by(wid) %>%
  summarize(mean_recalled = mean(recalled_numeric)) %>%
  mutate(diff_from_chance = mean_recalled - chance_recall)

# Run recall model
fit1 <- brm(diff_from_chance ~ 1,
            family = gaussian,
            data = recall_summary,
            chains = N_CHAINS,
            iter = N_ITER)
# Extract and save fixed effects for recall model
recall_effects <- data.frame(
  model = c(rep("recall", nrow(posterior_summary(fit1)))),
  rbind(get_credible_intervals(fit1))
)

write.csv(recall_effects, 
          file.path(opath, paste0("recall_effects_", exp_version, ".csv")), 
          row.names = FALSE)

# Fit value memory model

# First filter to remove non-recalled items
filtered_recall_df <- recall_df %>%
  filter(recalled == "True")

normalize_preserve_zero <- function(df, columns) {
  # Get maximum absolute value ignoring NAs
  max_abs <- max(abs(na.omit(unlist(df[, columns]))))
  df_norm <- df
  for (col in columns) {
    df_norm[[col]] <- df[[col]] / max_abs
  }
  return(df_norm)
}
recall_df_norm <- normalize_preserve_zero(filtered_recall_df, c("outcome", "remembered_outcome"))


print(head(recall_df_norm))

fit2 <- brm(remembered_outcome ~ outcome + (outcome|wid), 
            family = gaussian, 
            data = recall_df_norm, 
            chains = N_CHAINS, 
            iter = N_ITER)

# Extract and save fixed effects for value memory model
value_effects <- data.frame(
  model = c(rep("value_recall", nrow(posterior_summary(fit2)))),
  rbind(get_credible_intervals(fit2))
)
write.csv(value_effects, 
          file.path(opath, paste0("value_effects_", exp_version, ".csv")), 
          row.names = FALSE)
