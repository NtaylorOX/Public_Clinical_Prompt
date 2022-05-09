import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='7,8,9 '
# # Kill all processes on GPU 6 and 7
# os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 6 && $2 <= 7 {print $5}')""")
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl # current works with version 1.5.10
from torchnlp.encoders import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loguru import logger

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizerFast as RobertaTokenizer
from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from bert_classifier import MimicBertModel, MimicDataset, MimicDataModule
import argparse
from datetime import datetime
import warnings

from data_utils import FewShotSampler, Mimic_ICD9_Processor, Mimic_ICD9_Triage_Processor, Mimic_Mortality_Processor, Mimic_LOS_Processor, Mimic_Readmission_Processor

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

'''
Script to run training with a argument specified BERT model as the pre-trained encoder for instance classification.


Example cmd usage:

python train_binary_classifier.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT --batch_size 4 --gpus 0 --max_epochs 10 --dataset mortality --fast_dev_run True
'''


#TODO - edit below to handle all mimic tasks

# classes are imbalanced - lets calculate class weights for loss

def get_class_weights(train_df, label_col):
    classes = list(train_df[label_col].unique())
    class_dict = {}
    nSamples = []
    for c in classes:
        class_dict[c] = len(train_df[train_df[label_col] == c])
        nSamples.append(class_dict[c])

    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    return torch.FloatTensor(normedWeights)

def read_csv(data_dir,filename):
    return pd.read_csv(f"{data_dir}{filename}", index_col=None)

def main():
    parser = argparse.ArgumentParser()

    #TODO - add an argument to specify whether using balanced data then update directories based on that

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "/mnt/sdd/yizhang/projects/data",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--training_file",
                        default = "train.csv",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")
    parser.add_argument("--validation_file",
                        default = "valid.csv",
                        type=str,
                        help = "The default name of the training file")
    parser.add_argument("--test_file",
                        default = "test.csv",
                        type=str,
                        help = "The default name of hte test file")

    parser.add_argument("--pretrained_models_dir",
                        default="",
                        type=str,
                        help="The data path to the directory containing local pretrained models from huggingface")


    parser.add_argument("--text_col",
                        default = "Clinical_Note_Text",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default = "/mnt/sdd/yizhang/projects/prompt_mimic/mimic-all-tasks/pytorch-lightning-models/logs",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files"
                        )
    parser.add_argument("--reinit_n_layers",
                        default = 0,
                        type=int,
                        help = "number of pretrained final bert encoder layers to reinitialize for stabilisation"
                        )
    parser.add_argument("--max_tokens",
                        default = 512,
                        type=int,
                        help = "Max tokens to be used in modelling"
                        )
    parser.add_argument("--max_epochs",
                        default = 30,
                        type=int,
                        help = "Number of epochs to train"
                        )
    parser.add_argument("--batch_size",
                        default = 4,
                        type=int,
                        help = "batch size for training"
                        )
    parser.add_argument("--accumulate_grad_batches",
                        default = 10,
                        type=int,
                        help = "number of batches to accumlate before optimization step"
                        )
    parser.add_argument("--balance_data",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")
    parser.add_argument("--class_weights",
                        action = 'store_true',
                        help="Whether not to apply ce_class_weights for cross entropy loss function")

    parser.add_argument("--gpus", type=int, default=1, help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

    parser.add_argument(
            "--encoder_model",
            default= 'emilyalsentzer/Bio_ClinicalBERT',# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased',
            type=str,
            help="Encoder model to be used.",
        )

    parser.add_argument(
        "--transformer_type",
        default='bert', #'longformer', roberta-long
        type=str,
        help="Encoder model /tokenizer to be used (has consequences for tokenization and encoding; default = longformer).",
    )   
    
    parser.add_argument(
        "--max_tokens_longformer",
        default=4096,
        type=int,
        help="Max tokens to be considered per instance..",
    )

    parser.add_argument(
        "--encoder_learning_rate",
        default=1e-05,
        type=float,
        help="Encoder specific learning rate.",
    )
    parser.add_argument(
        "--classifier_learning_rate",
        default=1e-05,
        type=float,
        help="Classification head learning rate.",
    )
    parser.add_argument(
        "--nr_frozen_epochs",
        default=0,
        type=int,
        help="Number of epochs we want to keep the encoder model frozen.",
    )

    parser.add_argument(
        "--dataset",
        default="mortality", #or: icd9_triage
        type=str,
        help="name of dataset",
    )

    parser.add_argument(
        "--label_col",
        default="label", # label column of dataframes provided - should be label if using the dataprocessors from utils
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=24,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="monitor_roc_auc", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        help="Optimization algorithm to use e.g. adamw, adafactor"
    )

    parser.add_argument(
        "--training_size",
        default="full",
        type=str,
        help="full training used, fewshot, or zero"
    )

    parser.add_argument(
        "--few_shot_n",
        type=int,
        default = 100
    )

    # TODO - add an argument to specify whether using balanced data then update directories based on that
    args = parser.parse_args()

    print(f"arguments provided are: {args}")
    # set up parameters
    data_dir = args.data_dir
    save_dir = args.save_dir
    pretrained_dir = args.pretrained_models_dir
    pretrained_model_name = args.encoder_model
    max_tokens = args.max_tokens
    n_epochs = args.max_epochs
    batch_size = args.batch_size
    reinit_n_layers = args.reinit_n_layers
    accumulate_grad_batches = args.accumulate_grad_batches

    time_now = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    # set up the ckpt and logging dirs



    # update ckpt and logs dir based on the dataset
    
    ckpt_dir = f"./ckpts/{args.dataset}/{args.encoder_model}/version_{time_now}"
    # log_dir = f"./logs/{args.dataset}/"
    log_dir = f"{args.save_dir}/{args.dataset}/{args.training_size}_{args.few_shot_n}/full"

    # update ckpt and logs dir based on whether plm (encoder) was frozen during training

    if args.nr_frozen_epochs > 0:
        logger.warning(f"Freezing the encoder/plm for {args.nr_frozen_epochs} epochs")
        ckpt_dir = f"./ckpts/{args.dataset}/frozen_plm/{args.encoder_model}/version_{time_now}"
        # log_dir = f"./logs/{args.dataset}/frozen_plm/"
        log_dir = f"{log_dir}/frozen_plm"

    # load tokenizer
    print(f"loading tokenizer : {pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_model_name}")
    
    # TODO update the dataloading to use the custom dataprocessors from data_utils in this folder

    if args.dataset == "icd9_50":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        Processor = Mimic_ICD9_Processor
        # update data_dir
        data_dir = f"{args.data_dir}/mimic3-icd9-data/intermediary-data/top_50_icd9/"

        # are we doing any downsampling or balancing etc
        class_weights = args.class_weights
        balance_data = args.balance_data

        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)


    elif args.dataset == "icd9_triage":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        Processor = Mimic_ICD9_Triage_Processor
        # update data_dir
        data_dir = f"{args.data_dir}/mimic3-icd9-data/intermediary-data/triage/"

        # are we doing any downsampling or balancing etc
        class_weights = args.class_weights
        balance_data = args.balance_data
        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)

    elif args.dataset == "mortality":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        Processor = Mimic_Mortality_Processor
        # update data_dir
        data_dir = f"{args.data_dir}/clinical-outcomes-data/mimic3-clinical-outcomes/mp/"

        # are we doing any downsampling or balancing etc
        class_weights = args.class_weights
        balance_data = args.balance_data

        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)
    elif args.dataset == "los":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        Processor = Mimic_LOS_Processor
        # update data_dir
        data_dir = f"{args.data_dir}/mimic_los/"

        # are we doing any downsampling or balancing etc
        class_weights = args.class_weights
        balance_data = args.balance_data

        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)
    elif args.dataset == "readmission":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        Processor = Mimic_Readmission_Processor
        # update data_dir
        data_dir = f"{args.data_dir}/discharge/"

        # are we doing any downsampling or balancing etc
        class_weights = args.class_weights
        balance_data = args.balance_data

        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)

    else:
        #TODO implement los and mimic readmission
        raise NotImplementedError
    # if doing few shot sampling - apply few shot sampler

    if args.training_size =="fewshot":
        logger.warning("Will be performing few shot learning!")
        # initialise the sampler
        support_sampler = FewShotSampler(num_examples_per_label = args.few_shot_n, also_sample_dev=False, label_col = args.label_col)
        # now apply to each dataframe but convert to dictionary in records form first
        train_df = support_sampler(train_df.to_dict(orient="records"), seed = 1)
        #maybe val can 
        val_df = support_sampler(val_df.to_dict(orient="records"), seed = 1)
        test_df = support_sampler(test_df.to_dict(orient="records"), seed = 1)

    logger.warning(f"train_df shape: {train_df.shape} and train_df cols:{train_df.columns}")

    # get number labels or classes as length of class_labels
    n_labels = len(class_labels)
    # push data through pipeline
    # instantiate datamodule
    data_module = MimicDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=batch_size,
        max_token_len=max_tokens,
        label_col = args.label_col,
        num_workers=args.loader_workers,
    )
    # set up some parameters
    steps_per_epoch = len(train_df) // batch_size
    total_training_steps = steps_per_epoch * n_epochs
    # warmup_steps = total_training_steps // 5
    warmup_steps = 100
    warmup_steps, total_training_steps

    # get some class specific loss weights - only needed if doing some form of weighted cross entropy with ubalanced classes
    ce_class_weights = get_class_weights(data_module.train_df, args.label_col)

    #set up model
    model = MimicBertModel(bert_model=pretrained_model_name,
                                 num_labels=n_labels,
                                 n_warmup_steps=warmup_steps,
                                 n_training_steps=total_training_steps,
                                 nr_frozen_epochs = args.nr_frozen_epochs,
                                 ce_class_weights=ce_class_weights,
                                 weight_classes=args.class_weights,
                                 reinit_n_layers=reinit_n_layers,
                                 class_labels = class_labels,
                                 encoder_learning_rate = args.encoder_learning_rate,
                                 classifier_learning_rate = args.classifier_learning_rate,
                                 optimizer= args.optimizer                            
                                 )

    #setup checkpoint and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ckpt_dir}",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor=args.monitor,
        mode=args.metric_mode,
        save_last = True
    )

    tb_logger = TensorBoardLogger(
        save_dir=f"{log_dir}",
        version="version_" + time_now,
        name=f'{args.encoder_model}',
    )

    # early stopping based on validation metric of choice
    early_stopping_callback = EarlyStopping(monitor=args.monitor, mode = args.metric_mode, patience=args.patience)

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        gpus=[args.gpus],
        log_gpu_memory="all",
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.accumulate_grad_batches,
        checkpoint_callback = True,
        callbacks = [checkpoint_callback,early_stopping_callback],
        max_epochs=args.max_epochs,
        default_root_dir=f'./'        
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------

    # datamodule = MimicDataModule
    trainer.fit(model, data_module)

    # test
    # trainer.test(ckpt_path=f"ckpt_dir")

# run script
if __name__ == "__main__":
    main()