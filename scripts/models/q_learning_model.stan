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
    array[n_learning_trials] int<lower=1,upper=6> trial_position;        
    
    // Choice phase    
    array[n_choice_trials] int<lower=1,upper=n_subjects> subject_choice;    
    array[n_choice_trials] int game_choice;    
    array[n_choice_trials] int<lower=1,upper=n_features> chosen_feature;    
    array[n_choice_trials] int<lower=0,upper=1> choice;
}

parameters {
    // Hierarchical learning rate
    real<lower=0> a1;
    real<lower=0> a2;
    real<lower=0,upper=1> learning_rate[n_subjects];
        
    vector[n_betas] beta_mu;
    vector<lower=0,upper=pi()/2>[n_betas] tau_unif;
    cholesky_factor_corr[n_betas] Lcorr;
    matrix[n_betas,n_subjects] z;
}

transformed parameters {
    vector<lower=0>[n_betas] tau;
    matrix[n_subjects,n_betas] u;
        
    // Store Q-values and counts for each subject-game-feature combination
    array[n_subjects, max_games, n_features] real q_values;
    array[n_subjects, max_games, n_features] real feature_counts;
        
    // Initialize all Q-values and counts to zero
    for (s in 1:n_subjects) {
        for (g in 1:max_games) {
            for (f in 1:n_features) {
                q_values[s,g,f] = 0.0;
                feature_counts[s,g,f] = 0;
            }
        }
    }
        
    // Process all learning trials to compute Q-values with Q-learning
    for (t in 1:n_learning_trials) {
        int s = subject_learn[t];
        int g = game_learn[t];
        int f1 = feature1[t];
        int f2 = feature2[t];
        real alpha = learning_rate[s];
                
        // Q-learning updates for both features
        q_values[s,g,f1] = q_values[s,g,f1] + alpha * (outcome[t] - q_values[s,g,f1]);
        q_values[s,g,f2] = q_values[s,g,f2] + alpha * (outcome[t] - q_values[s,g,f2]);
        
        // Update counts
        feature_counts[s,g,f1] += 1;
        feature_counts[s,g,f2] += 1;
    }
        
    // Reparameterize betas
    for (k in 1:n_betas) {
        tau[k] = 2.5 * tan(tau_unif[k]);
    }
    u = (diag_pre_multiply(tau,Lcorr)*z)';
}

model {
    // Hierarchical learning rate priors
    a1 ~ normal(0,5);
    a2 ~ normal(0,5);
    learning_rate ~ beta(a1,a2);
        
    beta_mu ~ normal(0, 5);
    tau_unif ~ uniform(0,pi()/2);
    Lcorr ~ lkj_corr_cholesky(2);
    to_vector(z) ~ normal(0, 1);
        
    for (c in 1:n_choice_trials) {
        int s = subject_choice[c];
        int g = game_choice[c];
        int f = chosen_feature[c];
        real feature_value = q_values[s,g,f] * feature_counts[s,g,f];  // Q-value × count to get sum
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
        real feature_value = q_values[s,g,f] * feature_counts[s,g,f];  // Q-value × count to get sum
        real mu = (beta_mu[1] + u[s, 1]) + (beta_mu[2] + u[s, 2]) * feature_value;
        log_lik[c] = bernoulli_logit_lpmf(choice[c] | mu);
    }
}