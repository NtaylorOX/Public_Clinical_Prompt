
from loguru import logger
import os
import copy
import math
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, Trainer, TextDataset
from transformers import TrainingArguments, HfArgumentParser

from torch.utils.data import Dataset

from transformers.modeling_longformer import LongformerSelfAttention

import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F


import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock
import pandas as pd


# Format: each document should be separated by an empty line
TRAIN_FPATH = 'data/filtered_all_notes_train.txt'
VAL_FPATH = 'data/filtered_all_notes_val.txt'
SAMPLE_FPATH = 'data/filtered_all_notes_SAMPLE.txt'

MODEL_OUT_DIR = './longformer_gen'
LOCAL_ATTN_WINDOW = 512 #params['local_attention_window']
GLOBAL_MAX_POS = 4096 #params['global_attention_window']

FAST_DEV_RUN = True

if FAST_DEV_RUN == True:

    pd.read_csv(VAL_FPATH,sep='\t', header=None).sample(100).to_csv(SAMPLE_FPATH,header=None,index=None,sep='\t')

    TRAIN_FPATH = SAMPLE_FPATH
    VAL_FPATH = SAMPLE_FPATH


def main():

    if FAST_DEV_RUN == True:

        training_args = TrainingArguments(
            output_dir="./longformer_gen/checkpoints",
            overwrite_output_dir=True,
            max_steps=1,
            warmup_steps= 0, #-->3000
            logging_steps=1,
            save_steps=1,
            max_grad_norm= 5.0,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=2,
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
        output_dir=f"./longformer_gen/checkpoints/bioclinicaLongformer",
        overwrite_output_dir=True,
        warmup_steps= 500,
        logging_steps=500,
        max_steps = 3000,
        save_steps=500,
        max_grad_norm= 5.0,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps= 32,
        learning_rate = 0.00003,
        adam_epsilon= 1e-6,
        weight_decay= 0.01,
        do_eval= True,
        do_train=True,
        fp16=True
        )

    base_model_name_HF = 'allenai/biomed_roberta_base' #params['base_model_name']

    base_model_name = base_model_name_HF.split('/')[-1]
    model_path = f'{MODEL_OUT_DIR}/bioclinical-longformer' #includes speedfix
    unpretrained_model_path = f'{MODEL_OUT_DIR}/{base_model_name}-{GLOBAL_MAX_POS}' #includes speedfix

    logger.info(f'Loading the model from {unpretrained_model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(unpretrained_model_path,model_max_length=GLOBAL_MAX_POS)
    model = RobertaLongForMaskedLM.from_pretrained(unpretrained_model_path,gradient_checkpointing=True)

    logger.warning(f'Tokenizer {tokenizer} parameterized with model_max_len as {tokenizer.model_max_length}')

    # model.config.gradient_checkpointing = True #set this to ensure GPU memory constraints are OK.
    
    logger.critical(f'Pre-Training {model.num_parameters()}-parameter model. This could take ~ 2-3 days!!!!')
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path_out=training_args.output_dir)

    logger.warning(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.warning(f'Saving model to {model_path}/final')
    model.save_pretrained(f'{model_path}/final') # --> "./longformer_gen/bioclinicaLongformer/final"
    logger.critical('Final pre-trained model, tokenizer,and config saved!')


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):


        logger.warning(f'Block size in dataset set as {block_size}')
        # warnings.warn(DEPRECATION_WARNING, FutureWarning)
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f'Creating features from dataset file at {file_path}')

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 10 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size, padding='max_length')
        
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]



def pretrain_and_evaluate(training_args, model, tokenizer, eval_only, model_path_out):
    logger.info(f'Loading and tokenizing data is usually slow: {VAL_FPATH}')

    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=VAL_FPATH,
                              block_size=GLOBAL_MAX_POS)

    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {TRAIN_FPATH}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=TRAIN_FPATH,
                                    block_size= GLOBAL_MAX_POS)


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