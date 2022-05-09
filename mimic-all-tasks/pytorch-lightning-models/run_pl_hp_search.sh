#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_triage)
n_trials=(50)
for task in "${tasks[@]}"
    do
    python hp_search_optuna.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT \
        --num_epochs 10 --n_trials "$n_trials" --dataset "$task" --training_size fewshot --few_shot_n 128 --gpus 6 &&    
    # frozen plm i.e. no tune_plm argument
    python hp_search_optuna.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT \
        --num_epochs 10 --nr_frozen_epochs 10 --n_trials "$n_trials" --dataset "$task" --training_size fewshot --few_shot_n 128 --gpus 6
done