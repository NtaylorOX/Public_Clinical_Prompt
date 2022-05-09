#!/usr/bin/env bash

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model bert-base-uncased \
    --gpus 1

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model emilyalsentzer/Bio_ClinicalBERT \
    --gpus 1

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta \
    --encoder_model allenai/biomed_roberta_base \
    --gpus 1

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta-long \
    --encoder_model simonlevine/bioclinical-roberta-long \
    --batch_size 2 \
    --gpus 1
    # --fast_dev_run True