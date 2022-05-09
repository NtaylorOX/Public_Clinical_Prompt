#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_triage)
# do all these with finetune
for task in "${tasks[@]}"
    do
    # manual template manual verb
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type manual \
                --verbalizer_type manual --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6 &&
    # manual template soft verb            
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type manual \
                --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6 &&
    # soft template manual verb
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type soft \
                --verbalizer_type manual --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6 &&
    # soft template soft verb
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type soft \
                --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6 &&
    # mixed template manual verb
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type mixed \
                --verbalizer_type manual --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6 &&   
   # mixed template soft verb
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type mixed \
                --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 6
done
