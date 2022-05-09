#!/usr/bin/env bash
# run optimized sensitivity analysis
tasks=(icd9_triage)
hidden_size=(2 5 100 768 2000)
num_sample=(128)
classifier_lr=(0.000302)
dropout=(0.382)
grad_accum_steps=(4)
for hidden in "${hidden_size[@]}" 

    do  # frozen plm below
        python train_bert_classifier.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT \
                    --batch_size 4 \
                    --gpus 7 --max_epochs 20 \
                    --nr_frozen_epochs 20 \
                    --dataset "$tasks" \
                    --training_size fewshot \
                    --few_shot_n "$num_sample" \
                    --classifier_hidden_dim "$hidden" \
                    --accumulate_grad_batches "$grad_accum_steps" \
                    --classifier_learning_rate "$classifier_lr" \
                    --dropout "$dropout" \
                    --sensitivity True   
done    
