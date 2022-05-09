#!/usr/bin/env bash
# tasks will be array of datasets you want to run experiments on separated by a space
tasks=(icd9_triage)
# do all these with finetune
for task in "${tasks[@]}"
    do    
   # mixed template soft verb
    python prompt_experiment_runner.py --model bert \
                --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
                --num_epochs 15 --template_id 2 --template_type mixed \
                --verbalizer_type soft --freeze_verbalizer_plm --verbalizer_id 0 \
                --max_steps 25000 --training_size fewshot --few_shot_n 128 --dataset "$task" --run_evaluation --gpu_num 7
done
