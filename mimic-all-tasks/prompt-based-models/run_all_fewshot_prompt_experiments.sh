#!/usr/bin/env bash
# all tasks 50, 200, 500 samples - put tasks you wanna do in the tasks array separated by space
tasks=(icd9_triage icd9_50)
model=(microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) 
# e.g microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract | emilyalsentzer/Bio_ClinicalBERT
num_sample=(16 32 64 128)
for task in "${tasks[@]}"
    do
    for num in "${num_sample[@]}"
        do  # finetune plm
            echo "$task $num $model"
            # python prompt_experiment_runner.py --model bert --model_name_or_path "$model" \
            #     --num_epochs 15 --template_id 2 --template_type mixed \
            #     --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --tune_plm --dataset "$task" --zero_shot --run_evaluation --gpu_num 7 \
            #     --training_size fewshot --few_shot_n "$num" --no_ckpt &&
            # frozen plm
            python prompt_experiment_runner.py \
                --model bert \
                --model_name_or_path "$model" \
                --num_epochs 15 --template_id 2 --template_type mixed \
                --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --dataset "$task" --zero_shot --run_evaluation --gpu_num 7 \
                --training_size fewshot --few_shot_n "$num"
    done
done    