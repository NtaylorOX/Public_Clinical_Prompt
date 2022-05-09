
from loguru import logger
import os
import copy
import math
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizer, DataCollatorForLanguageModeling, Trainer, TextDataset
from transformers import TrainingArguments, HfArgumentParser

from torch.utils.data import Dataset

from transformers.modeling_longformer import LongformerSelfAttention

import logging
import warnings

import torch
import torch.nn as nn

import pandas as pd


# Format: each document should be separated by an empty line
TRAIN_FPATH = 'data/filtered_all_notes_train.txt'
VAL_FPATH = 'data/filtered_all_notes_val.txt'
SAMPLE_FPATH = 'data/filtered_all_notes_SAMPLE.txt'

MODEL_OUT_DIR = './roberta_gen'

FAST_DEV_RUN = False

if FAST_DEV_RUN == True:

    pd.read_csv(VAL_FPATH,sep='\t', header=None).sample(100).to_csv(SAMPLE_FPATH,header=None,index=None,sep='\t')

    TRAIN_FPATH = SAMPLE_FPATH
    VAL_FPATH = SAMPLE_FPATH


def main():

    if FAST_DEV_RUN == True:

        training_args = TrainingArguments(
            output_dir="./roberta_gen/checkpoints",
            overwrite_output_dir=True,
            max_steps=1,
            warmup_steps= 0,
            logging_steps=1,
            save_steps=1,
            max_grad_norm= 5.0,
            per_device_eval_batch_size=8, #4,4 WORKS with FP32
            per_device_train_batch_size=8,
            gradient_accumulation_steps= 32,
            learning_rate = 0.00003,
            adam_epsilon= 1e-6,
            weight_decay= 0.01,
            do_eval= True,
            do_train=True,
            fp16=True
            )
    
    elif FAST_DEV_RUN == False:

        training_args = TrainingArguments(
        output_dir="./roberta_gen/checkpoints",
        overwrite_output_dir=True,
        warmup_steps= 500,
        logging_steps=500,
        max_steps = 5000,
        save_steps=500,
        max_grad_norm= 5.0,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps= 32,
        learning_rate = 0.00003,
        adam_epsilon= 1e-6,
        weight_decay= 0.01,
        do_eval= True,
        do_train=True,
        fp16=True
        )

    base_model_name_HF = 'allenai/biomed_roberta_base' #params['base_model_name']

    model_path = f'{MODEL_OUT_DIR}/bioclinical-roberta'

    unpretrained_model_path = base_model_name_HF

    logger.info(f'Loading the model from {unpretrained_model_path}')

    tokenizer = RobertaTokenizer.from_pretrained(unpretrained_model_path)
    model = RobertaForMaskedLM.from_pretrained(unpretrained_model_path, gradient_checkpointing=True)

    logger.warning(f'Tokenizer {tokenizer} parameterized with model_max_len as {tokenizer.model_max_length}')

    model.config.gradient_checkpointing = True
    
    logger.critical(f'Pre-Training {model.num_parameters()}-parameter model. This could take days!!!!')
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path_out=training_args.output_dir)

    logger.warning(f'Saving model to {model_path}/final')
    model.save_pretrained(f'{model_path}/final') # --> "./longformer_gen/bioclinicaLongformer/final"
    logger.critical('Final pre-trained model, tokenizer,and config saved!')


def pretrain_and_evaluate(training_args, model, tokenizer, eval_only, model_path_out):
    logger.info(f'Loading and tokenizing data is usually slow: {VAL_FPATH}')

    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=VAL_FPATH,
                              block_size=tokenizer.max_len)

    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {TRAIN_FPATH}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=TRAIN_FPATH,
                                    block_size= tokenizer.max_len)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    logger.warning(f'Model Params set to {training_args}')

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path_out)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')

def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model

if __name__=="__main__":
    main()