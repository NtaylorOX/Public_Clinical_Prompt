#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_triage)
temp_ids=(0 1 2 3)
prompt_lr=(0.012125)
grad_accum_steps=(3)
for temp_id in "${temp_ids[@]}"
    do
    # frozen plm with mixed template and soft verbalizer
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id "$temp_id" --template_type mixed \
                --prompt_lr "$prompt_lr" --gradient_accum_steps "$grad_accum_steps" \
                --verbalizer_type soft --verbalizer_id 0 --training_size fewshot --few_shot_n 128 --dataset "$tasks" --sensitivity True \
                --run_evaluation --gpu_num 6 &&
    # frozen plm with mixed template and manual verbalizer
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id "$temp_id" --template_type mixed \
                --prompt_lr "$prompt_lr" --gradient_accum_steps "$grad_accum_steps" \
                --verbalizer_type manual --verbalizer_id 0 --training_size fewshot --few_shot_n 128 --dataset "$tasks" --sensitivity True \
                --run_evaluation --gpu_num 6
done