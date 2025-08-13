data {
    int<lower=1> n_learning_trials;
    int<lower=1> n_choice_trials;
    int<lower=1> n_subjects;
    int<lower=1> n_features;
    int<lower=1> n_betas;
    int<lower=1> max_games;
    
    // Learning phase
    array[n_learning_trials] int<lower=1,upper=n_subjects> subject_learn;
    array[n_learning_trials] int game_learn;
    array[n_learning_trials] int<lower=1,upper=n_features> feature1;
    array[n_learning_trials] int<lower=1,upper=n_features> feature2;
    array[n_learning_trials] real outcome;

    // Choice phase
    array[n_choice_trials] int<lower=1,upper=n_subjects> subject_choice;
    array[n_choice_trials] int game_choice;
    array[n_choice_trials] int<lower=1,upper=n_features> chosen_feature;
    array[n_choice_trials] int<lower=0,upper=1> choice;
}

parameters {
    vector[n_betas] beta_mu;
    vector<lower=0,upper=pi()/2>[n_betas] tau_unif;
    cholesky_factor_corr[n_betas] Lcorr;
    matrix[n_betas,n_subjects] z;
}

transformed parameters {
    vector<lower=0>[n_betas] tau;
    matrix[n_subjects,n_betas] u;
    
    // Store feature sums for each subject-game-feature combination
    array[n_subjects, max_games, n_features] real feature_sums;
    
    // Initialize all sums to zero
    for (s in 1:n_subjects) {
        for (g in 1:max_games) {
            for (f in 1:n_features) {
                feature_sums[s,g,f] = 0.0;
            }
        }
    }
    
    // Process all learning trials to compute feature sums (NO WEIGHTING)
    for (t in 1:n_learning_trials) {
        int s = subject_learn[t];
        int g = game_learn[t];
        int f1 = feature1[t];
        int f2 = feature2[t];
        // Just use the raw outcome, no position weighting
        real raw_outcome = outcome[t];
        
        feature_sums[s,g,f1] += raw_outcome;
        feature_sums[s,g,f2] += raw_outcome;
    }
    
    // Reparameterize
    for (k in 1:n_betas) {
        tau[k] = 2.5 * tan(tau_unif[k]);
    }
    u = (diag_pre_multiply(tau,Lcorr)*z)';
}

model {
    beta_mu ~ normal(0, 5);
    tau_unif ~ uniform(0,pi()/2);
    Lcorr ~ lkj_corr_cholesky(2);
    to_vector(z) ~ normal(0, 1);
    
    for (c in 1:n_choice_trials) {
        int s = subject_choice[c];
        int g = game_choice[c];
        int f = chosen_feature[c];
        real feature_value = feature_sums[s,g,f];
        real mu = (beta_mu[1] + u[s, 1]) + (beta_mu[2] + u[s, 2]) * feature_value;
        choice[c] ~ bernoulli_logit(mu);
    }
}

generated quantities {
    vector[n_choice_trials] log_lik;
    
    for (c in 1:n_choice_trials) {
        int s = subject_choice[c];
        int g = game_choice[c];
        int f = chosen_feature[c];
        real feature_value = feature_sums[s,g,f];
        real mu = (beta_mu[1] + u[s, 1]) + (beta_mu[2] + u[s, 2]) * feature_value;
        log_lik[c] = bernoulli_logit_lpmf(choice[c] | mu);
    }
}