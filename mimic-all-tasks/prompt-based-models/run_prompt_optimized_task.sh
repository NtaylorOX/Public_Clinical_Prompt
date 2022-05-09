#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_triage)
prompt_lr=(0.012125)
grad_accum_steps=(3)
for task in "${tasks[@]}"
    do

    # frozen plm with mixed template and manual verbalizer
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 50 --template_id 2 --template_type mixed \
                --prompt_lr "$prompt_lr" --gradient_accum_steps "$grad_accum_steps" \
                --verbalizer_type soft --verbalizer_id 0 --training_size fewshot --few_shot_n 128 --dataset "$tasks" --optimized_run True \
                --run_evaluation --gpu_num 6
done