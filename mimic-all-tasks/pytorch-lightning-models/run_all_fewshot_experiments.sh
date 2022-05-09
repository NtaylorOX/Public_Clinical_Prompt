#!/usr/bin/env bash
# all tasks 50, 200, 500 samples
tasks=(icd9_triage icd9_50)
num_sample=(16 32 64 128)
model=(microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) 
# e.g microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract | emilyalsentzer/Bio_ClinicalBERT
for task in "${tasks[@]}"
    do
    for num in "${num_sample[@]}"
        do  # finetune plm
            python train_bert_classifier.py --transformer_type bert --encoder_model "$model" \
                        --batch_size 4 \
                        --gpus 6 --max_epochs 15 \
                        --dataset "$task" \
                        --training_size fewshot \
                        --few_shot_n "$num" &&    
            # frozen plm below
            python train_bert_classifier.py --transformer_type bert --encoder_model "$model" \
                        --batch_size 4 \
                        --gpus 6 --max_epochs 15 \
                        --nr_frozen_epochs 15 \
                        --dataset "$task" \
                        --training_size fewshot \
                        --few_shot_n "$num"
                        
    done    
done