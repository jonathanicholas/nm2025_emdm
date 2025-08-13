# Episodic memory facilitates flexible decision making via access to detailed events
### Jonathan Nicholas, Marcelo G. Mattar
Department of Psychology, New York University

```sh
├── plot_and_summarize.ipynb                                 # notebook containing code to produce figures and summary statistics reported in the paper
├── run_analyses.R                                           # master script to run all primary analyses reported in the paper
├── main.tex                                                 # tex file to produce manuscript
├── figures                                                  # directory containing all figures
├── scripts                                                  # directory containing all scripts
  ├── models                                                 # directory containing process models + simulation code
    ├── simulate_exp1.py                                     # script to simulate experiment 1
    ├── simulate_exp5.py                                     # script to simulate experiment 5
    ├── true_offer_value_model.stan                          # stan code for true offer value model
    ├── recency_weighted_sum_model.stan                      # stan code for recency weighted summing model
    ├── q_learning_model.stan                                # stan code for q learning model
  ├── analyze_pilot.R                                        # script to analyze pilot experiment
  ├── analyze_exp.R                                          # script to analyze choice and rt for exps 1 and 2
  ├── analyze_exp_memory.R                                   # script to analyze recall for exps 1 and 2
  ├── analyze_exp_overall_perf.R                             # script to analyze overall performance for exps 1 and 2
  ├── analyze_manipulation_exp.R                             # script to analyze choice and rt for exps 3 and 4
  ├── analyze_manipulation_exp_memory.R                      # script to analyze recall for exps 3 and 4
  ├── analyze_manipulation_exp_overall_perf.R                # script to analyze overall performance for exps 3 and 4
  ├── analyze_final_round.R                                  # script to analyze final round performance for exp 4
  ├── analyze_exp5.R                                         # script to analyze choice and rt for exp 5
  ├── analyze_exp5_memory.R                                  # script to analyze recall for exp 5
  ├── analyze_pooled_exps.R                                  # script to analyze pooled choice and rt for exps 1 and 2
  ├── analyze_pooled_manipulation_exps.R                     # script to analyze pooled choice and rt for exps 3 and 4
  ├── analyze_exp5_memory.R                                  # script to analyze recall for exp 5
  ├── run_stan_models.R                                      # script to run stan models assessing feature-based strategy
├── output                                                   # directory with all analysis output
├── data                                                     # directory with data files for each experiment + simulated data
```    

## Contact
Jonathan Nicholas (jdn316@nyu.edu)
