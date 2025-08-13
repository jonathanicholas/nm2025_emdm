library(jsonlite)
library(rstan)
library(loo)
library(dplyr)

options(mc.cores = parallel::detectCores())

n_iter = 2000
exp = "1B"
dpath = file.path(getwd(), "data")
opath = file.path(getwd(), "output")
mpath = file.path(getwd(), "scripts/models")

# Load the data
stan_data <- fromJSON(file.path(dpath, sprintf("stan_data_%s.json", exp)))

# Get stan code for the Q-learning model
sm = sprintf("%s/q_learning_model.stan", mpath)
fit_q <- stan(file = sm, data = stan_data, iter=n_iter)

# Get stan code for the recency-weighting model
sm = sprintf("%s/recency_weighted_sum_model.stan", mpath)
fit_recency <- stan(file = sm, data = stan_data, iter=n_iter)

# Get stan code for the true offer value model
sm = sprintf("%s/true_offer_value_model.stan", mpath)
fit_true_value <- stan(file = sm, data = stan_data, iter=n_iter)

# Extract log-likelihood matrices
log_lik_true <- extract_log_lik(fit_true_value)
log_lik_rec <- extract_log_lik(fit_recency)
log_lik_q <- extract_log_lik(fit_q)

# Use LOO to save some time
loo_true <- loo(log_lik_true)
loo_rec <- loo(log_lik_rec)
loo_q <- loo(log_lik_q)

# Extract pointwise ELPD values
elpd_true <- loo_true$pointwise[,1]
elpd_rec <- loo_rec$pointwise[,1]
elpd_q <- loo_q$pointwise[,1]

# Calculate ELPD differences - Recency vs True
elpd_diff_rec_true <- (elpd_rec - elpd_true)
elpd_se_rec_true <- sqrt(length(elpd_diff_rec_true)) * sd(elpd_diff_rec_true)

# Calculate ELPD differences - Q-learning vs True
elpd_diff_q_true <- (elpd_q - elpd_true)
elpd_se_q_true <- sqrt(length(elpd_diff_q_true)) * sd(elpd_diff_q_true)

# Calculate ELPD differences - Q-learning vs Recency
elpd_diff_q_rec <- (elpd_q - elpd_rec)
elpd_se_q_rec <- sqrt(length(elpd_diff_q_rec)) * sd(elpd_diff_q_rec)

# Save detailed ELPD comparisons
elpd_results <- data.frame(
  comparison = c("recency_vs_true", "qlearning_vs_true", "qlearning_vs_recency"),
  elpd_difference = c(sum(elpd_diff_rec_true), sum(elpd_diff_q_true), sum(elpd_diff_q_rec)),
  standard_error = c(elpd_se_rec_true, elpd_se_q_true, elpd_se_q_rec)
)

write.csv(elpd_results, file.path(opath, sprintf("stan_model_comparison_elpd_%s.csv", exp)), row.names = FALSE)

# Save individual model summaries
recency_summary <- summary(fit_recency)$summary
write.csv(recency_summary, file.path(opath, sprintf("recency_model_summary_%s.csv", exp)), row.names = TRUE)

true_summary <- summary(fit_true_value)$summary
write.csv(true_summary, file.path(opath, sprintf("true_sum_model_summary_%s.csv", exp)), row.names = TRUE)

q_summary <- summary(fit_q)$summary
write.csv(q_summary, file.path(opath, sprintf("qlearning_model_summary_%s.csv", exp)), row.names = TRUE)

# Print comparison results
print("Model Comparison Results:")
print(elpd_results)

# Optional: Use loo_compare for formal comparison
loo_comparison <- loo_compare(loo_true, loo_rec, loo_q)
print("LOO Comparison (higher ELPD is better):")
print(loo_comparison)