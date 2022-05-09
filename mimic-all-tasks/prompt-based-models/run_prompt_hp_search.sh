#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_triage)
n_trials=(30)
for task in "${tasks[@]}"
    do
    python prompt_hyperparameter_search_optuna.py --num_epochs 10 --n_trials "$n_trials" --tune_plm --dataset "$task" --gpu_num 6 &&    
    # frozen plm i.e. no tune_plm argument
    python prompt_hyperparameter_search_optuna.py --num_epochs 10 --n_trials 30 --dataset "$task" --gpu_num 6
done
