#!/usr/bin/env bash
# based on optuna hyperparam random search results
tasks=(icd9_triage)
classifier_lr=(0.000302)
dropout=(0.382)
num_sample=(128)
grad_accum_steps=(4)
for task in "${tasks[@]}" 

    do  # frozen plm below
        python train_bert_classifier.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT \
                    --batch_size 4 \
                    --gpus 7 --max_epochs 50 \
                    --nr_frozen_epochs 50 \
                    --dataset "$task" \
                    --training_size fewshot \
                    --few_shot_n "$num_sample" \
                    --accumulate_grad_batches "$grad_accum_steps" \
                    --classifier_learning_rate "$classifier_lr" \
                    --dropout "$dropout" \
                    --optimized_run True
done    