#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_50 icd9_triage mortality)
for task in "${tasks[@]}"
    do
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type mixed \
                --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6 &&
    # frozen plm i.e. no tune_plm argument
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type mixed \
                --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --dataset "$task" --zero_shot --run_evaluation --gpu_num 6
done




