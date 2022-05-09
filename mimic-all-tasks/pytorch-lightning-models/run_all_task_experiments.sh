tasks=(icd9_triage icd9_50 mortality)
for task in "${tasks[@]}"
do
    python train_bert_classifier.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT \
                --batch_size 4 \
                --gpus 7 --max_epochs 15 \
                --dataset "$task" \
                --training_size full &&
    # frozen plm below
    python train_bert_classifier.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT \
                --batch_size 4 \
                --gpus 7 --max_epochs 15 \
                --nr_frozen_epochs 15 \
                --dataset "$task" \
                --training_size full
done