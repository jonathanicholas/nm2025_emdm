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
before_df <- recall_df[recall_df$multi_time == "Before",]
after_df <- recall_df[recall_df$multi_time == "After",]
chance_recall <- 0.375
# Create the summary dataframe
before_recall_summary <- before_df %>%
  mutate(recalled_numeric = ifelse(recalled == "True", 1, 0)) %>%
  group_by(wid) %>%
  summarize(mean_recalled = mean(recalled_numeric)) %>%
  mutate(diff_from_chance = mean_recalled - chance_recall)
after_recall_summary <- after_df %>%
  mutate(recalled_numeric = ifelse(recalled == "True", 1, 0)) %>%
  group_by(wid) %>%
  summarize(mean_recalled = mean(recalled_numeric)) %>%
  mutate(diff_from_chance = mean_recalled - chance_recall)

# Run recall model
fit1 <- brm(diff_from_chance ~ 1, family = gaussian, data = before_recall_summary, chains = N_CHAINS, iter = N_ITER)
fit2 <- brm(diff_from_chance ~ 1, family = gaussian, data = after_recall_summary, chains = N_CHAINS, iter = N_ITER)

# Extract and save fixed effects for recall model
recall_effects <- data.frame(
  model = c(rep("before_recall", nrow(posterior_summary(fit1))),
            rep("after_recall", nrow(posterior_summary(fit2)))),
  rbind(get_credible_intervals(fit1), get_credible_intervals(fit2))
)

# Compare before vs after recall (intercepts)
recall_samples <- data.frame(
  before = as.vector(posterior_samples(fit1)$b_Intercept),
  after = as.vector(posterior_samples(fit2)$b_Intercept)
)
recall_diff <- with(recall_samples, after - before)
recall_comparison <- data.frame(
  comparison = "after_vs_before_recall",
  estimate = mean(recall_diff),
  ci_95_lower = quantile(recall_diff, 0.025),
  ci_95_upper = quantile(recall_diff, 0.975),
  prob_positive = mean(recall_diff > 0)
)

write.csv(recall_effects, 
          file.path(opath, paste0("recall_effects_", exp_version, ".csv")), 
          row.names = FALSE)
write.csv(recall_comparison,
          file.path(opath, paste0("recall_comparison_", exp_version, ".csv")),
          row.names = FALSE)

# Fit value memory model

# First filter to remove non-recalled items
filtered_before_df <- before_df %>%
  filter(recalled == "True")
filtered_after_df <- after_df %>%
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
before_value_df_norm <- normalize_preserve_zero(filtered_before_df, c("outcome", "remembered_outcome"))
after_value_df_norm <- normalize_preserve_zero(filtered_after_df, c("outcome", "remembered_outcome"))

fit3 <- brm(remembered_outcome ~ outcome + (outcome|wid), 
            family = gaussian, 
            data = before_value_df_norm, 
            chains = N_CHAINS, 
            iter = N_ITER)
fit4 <- brm(remembered_outcome ~ outcome + (outcome|wid), 
            family = gaussian, 
            data = after_value_df_norm, 
            chains = N_CHAINS, 
            iter = N_ITER)

# Extract and save fixed effects for value memory model
value_effects <- data.frame(
  model = c(rep("before_value_recall", nrow(posterior_summary(fit3))),
            rep("after_value_recall", nrow(posterior_summary(fit4)))),
  rbind(get_credible_intervals(fit3), get_credible_intervals(fit4))
)

# Compare before vs after value memory (outcome slopes)
value_samples <- data.frame(
  before = as.vector(posterior_samples(fit3)$b_outcome),
  after = as.vector(posterior_samples(fit4)$b_outcome)
)
value_diff <- with(value_samples, after - before)
value_comparison <- data.frame(
  comparison = "after_vs_before_value_memory",
  estimate = mean(value_diff),
  ci_95_lower = quantile(value_diff, 0.025),
  ci_95_upper = quantile(value_diff, 0.975),
  prob_positive = mean(value_diff > 0)
)

write.csv(value_effects, 
          file.path(opath, paste0("value_effects_", exp_version, ".csv")), 
          row.names = FALSE)
write.csv(value_comparison,
          file.path(opath, paste0("value_comparison_", exp_version, ".csv")),
          row.names = FALSE)
