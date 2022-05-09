import os
from sched import scheduler
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='6,7 '

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import tempfile

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders import LabelEncoder

import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loguru import logger

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup 
from transformers.optimization import Adafactor, AdafactorSchedule 
from transformers import RobertaTokenizerFast as RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score 


from data_utils import FewShotSampler, Mimic_ICD9_Processor, Mimic_ICD9_Triage_Processor, Mimic_Mortality_Processor

from bert_classifier import MimicBertModel, MimicDataset, MimicDataModule

import numpy as np
import pandas as pd

import argparse

import joblib

parser = argparse.ArgumentParser()

#TODO - add an argument to specify whether using balanced data then update directories based on that

# Required parameters
parser.add_argument("--root_data_dir",
                    default = "/home/niallt/mimic-prompt-learning/Public_Prompt_Mimic_III/mimic-all-tasks",
                    type=str,
                    help = "The data path to the directory containing the notes and referral data files")
parser.add_argument("--dataset",
                    default = "icd9_triage",
                    type=str,
                    help = "The dataset to use e.g icd9_50 or icd9_triage")
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

parser.add_argument("--pretrained_plm_name",
                    default="emilyalsentzer/Bio_ClinicalBERT",
                    type=str,
                    help="The name of the pretrained plm")

parser.add_argument("--gpus_per_trial",
                    default=1,
                    type=int,
                    help="The number of individual gpus to use. Set to 0 for cpu.")

parser.add_argument("--gpus", type=int, default=6, help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

parser.add_argument("--n_trials",
                    default=30,
                    type=int,
                    help="The number of samples to take from hyperparameter search space.")

parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="The number epochs to run hyper param search for")

parser.add_argument("--few_shot_n", type=int, default = 128)

parser.add_argument("--grad_accum_steps",
                    default=0,
                    type=int,
                    help="The number of gradient accumulation steps")

parser.add_argument("--max_tokens",
                    default=512,
                    type=int,
                    help="Max sequence length for tokenizer and plm")

parser.add_argument("--label_col",
                    default = "label",
                    type=str,
                    help = "The column name for label/target")

parser.add_argument("--save_dir",
                    default = "/mnt/sdg/niallt/saved_models/mimic-tasks/pytorch-lightning-models/hp-search/optuna_results/",
                    type=str,
                    help = "The data path to the directory for ray tune results"
                    )
parser.add_argument(
    "--nr_frozen_epochs",
    default=0,
    type=int,
    help="Number of epochs we want to keep the encoder model frozen.",
)
parser.add_argument(
    "--transformer_type",
    default='bert', #'longformer', roberta-long
    type=str,
    help="Encoder model /tokenizer to be used (has consequences for tokenization and encoding; default = longformer).",
)   
parser.add_argument(
        "--encoder_model",
        default= 'emilyalsentzer/Bio_ClinicalBERT',# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased',
        type=str,
        help="Encoder model to be used.",
    )

parser.add_argument(
    "--training_size",
    default="full",
    type=str,
    help="full training used, fewshot, or zero"
)


# parse the argument files
args = parser.parse_args()
# set constants

num_epochs = args.num_epochs
gpus_per_trial = args.gpus_per_trial # set this to higher if using GPU
dataset = args.dataset
pretrained_model_name = args.encoder_model
root_data_dir = args.root_data_dir
max_tokens = args.max_tokens
label_col = args.label_col
nr_frozen_epochs = args.nr_frozen_epochs

# set save_dir based on the dataset provided and whether frozen
if nr_frozen_epochs > 0:
    save_dir = f"{args.save_dir}/{dataset}/frozen_plm/{pretrained_model_name}"
else:
    save_dir = f"{args.save_dir}/{dataset}/{pretrained_model_name}"

# redefine the mimic bert model to avoid certain logging etc

class hyperMimicBertModel(pl.LightningModule):
    def __init__(self,
                 bert_model,
                 num_labels,
                 class_labels = [],
                 bert_hidden_dim=768,
                 classifier_hidden_dim=768,
                 n_training_steps=None,
                 n_warmup_steps=50,                 
                 nr_frozen_epochs = 0,
                 config = None):

        super().__init__()
        # logger.warning(f"Building model based on following architecture. {bert_model}")

        # set all the relevant parameters
        self.num_labels = num_labels
        self.class_labels = class_labels

        
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps 
        self.nr_frozen_epochs = nr_frozen_epochs

        
        # get parameters from config
        self.plm_lr = config['plm_lr']
        self.classifier_lr = config['classifier_lr']
        self.dropout = config['dropout']
        self.optimizer = config['optimizer']
        self.classifier_hidden_dim = config['classifier_hidden_size']

        # self.save_hyperparameters()

        # load in bert model
        self.bert = AutoModel.from_pretrained(f"{bert_model}", return_dict=True)
        # nn.Identity does nothing if the dropout is set to None
        self.classification_head = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout) if self.dropout is not None else nn.Identity(),
                                        nn.Linear(classifier_hidden_dim, num_labels))
        

        self.criterion = nn.CrossEntropyLoss()

        # freeze if you wanted
        if self.nr_frozen_epochs > 0:
            logger.warning("Freezing the PLM i.e. the encoder - will just be tuning the classification head!")
            self.freeze_encoder()
        else:
            self._frozen = False    

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps



    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # obtaining the last layer hidden states of the Transformer
        last_hidden_state = output.last_hidden_state  # shape: (batch_size, seq_length, bert_hidden_dim)

        #         or can use the output pooler : output = self.classifier(output.pooler_output)
        # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation
        # by indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        logits = self.classification_head(CLS_token_state)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):        

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train/loss", loss, prog_bar=True, logger=True)
        # print(f"training loss: {loss}")
        return {"loss": loss, "predictions": outputs.detach(), "labels": labels.detach()}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("valid/loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs.detach(), "labels": labels.detach()}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test/loss", loss, prog_bar=True, logger=True)
        return loss 

    def validation_epoch_end(self, outputs):
        # logger.warning("on validation epoch end")

        # get class labels
        class_labels = self.class_labels


        labels = []
        predictions = []
        scores = []
        for output in outputs:
            
            for out_labels in output["labels"].to('cpu').detach().numpy():                                
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                
                # the handling of roc_auc score differs for binary and multi class
                if len(class_labels) > 2:
                    scores.append(torch.nn.functional.softmax(out_predictions).cpu().tolist())
                # append probas
                else:
                    scores.append(torch.nn.functional.softmax(out_predictions)[1].cpu().tolist())

                # get predictied labels                               
                predictions.append(np.argmax(out_predictions.to('cpu').detach().numpy(), axis = -1))

            #use softmax to normalize, as the sum of probs should be 1

        # print(f"Labels: {labels}  \nand num labels is: {len(labels)}")
        # print("\n")
        # print(f"Predicted labels: {predictions}")
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average = 'weighted')   

        # log this for monitoring
        self.log('monitor_balanced_accuracy', acc)
        
        # log to tensorboard

        self.log('valid/balanced_accuracy',acc)

        self.log('valid/f1_weighted',f1_weighted)



    def test_epoch_end(self, outputs):
        # get class labels
        class_labels = self.class_labels


        labels = []
        predictions = []
        scores = []
        for output in outputs:
            
            for out_labels in output["labels"].to('cpu').detach().numpy():                                
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                
                # the handling of roc_auc score differs for binary and multi class
                if len(class_labels) > 2:
                    scores.append(torch.nn.functional.softmax(out_predictions).cpu().tolist())
                # append probas
                else:
                    scores.append(torch.nn.functional.softmax(out_predictions)[1].cpu().tolist())

                # get predictied labels                               
                predictions.append(np.argmax(out_predictions.to('cpu').detach().numpy(), axis = -1))

            #use softmax to normalize, as the sum of probs should be 1
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average = 'weighted')

        # log to tensorboard

        self.log('test/balanced_accuracy',acc)
        self.log('test/f1_weighted',f1_weighted)



    def configure_optimizers(self):
            """ Sets different Learning rates for different parameter groups. """
            parameters = [
                {"params": self.classification_head.parameters()},
                {
                    "params": self.bert.parameters(),
                    "lr": self.plm_lr,
                },
            ]
            
            if self.optimizer == "adamw":
                optimizer = AdamW(parameters, lr=self.classifier_lr)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.n_warmup_steps,
                    num_training_steps=self.n_training_steps
                )
            elif self.optimizer == "adafactor":
                optimizer = Adafactor(parameters,  
                                    lr=self.classifier_lr,
                                    relative_step=False,
                                    scale_parameter=False,
                                    warmup_init=False)  
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps)
            else:
                raise NotImplementedError
            
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                )
            )

    def on_epoch_end(self):
        """ Pytorch lightning hook """        
        # logger.warning(f"On epoch {self.current_epoch}. Number of frozen epochs is: {self.nr_frozen_epochs}")
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            # logger.warning("unfreezing PLM(encoder)")
            self.unfreeze_encoder()
########################################################################################################################
#### main training functions ############################

def train_mimic(trial, dataset = dataset,
                pretrained_model_name = pretrained_model_name,
                root_data_dir = root_data_dir, 
                max_tokens = max_tokens, label_col = label_col,
                 num_epochs=num_epochs, warmup_steps = 50, total_training_steps = 20000, config = None
                ):
    
    logger.warning(f"got config: {config}")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_model_name}")

    # TODO update the dataloading to use the custom dataprocessors from data_utils in this folder

    if dataset == "icd9_50":
        # logger.warning(f"Using the following dataset: {dataset} ")
        Processor = Mimic_ICD9_Processor
        # update data_dir
        root_data_dir = f"{root_data_dir}/mimic3-icd9-data/intermediary-data/top_50_icd9/"

        # are we doing any downsampling or balancing etc
        class_weights = False
        balance_data = False

        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = root_data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = root_data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = root_data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)

        # few few shot sample

        support_sampler = FewShotSampler(num_examples_per_label = args.few_shot_n, also_sample_dev=False, label_col = label_col)
        # now apply to each dataframe but convert to dictionary in records form first
        train_df = support_sampler(train_df.to_dict(orient="records"), seed = 1)
        val_df = support_sampler(val_df.to_dict(orient="records"), seed = 1)
        test_df = support_sampler(test_df.to_dict(orient="records"), seed = 1)
        # load model
        model = hyperMimicBertModel(bert_model=pretrained_model_name,
                                 num_labels=len(class_labels),
                                 n_warmup_steps=warmup_steps,
                                 n_training_steps=total_training_steps,
                                    config = config
                                 )
    elif args.dataset == "icd9_triage":
        
        Processor = Mimic_ICD9_Triage_Processor
        # update data_dir
        root_data_dir = f"{root_data_dir}/mimic3-icd9-data/intermediary-data/triage/"

        # are we doing any downsampling or balancing etc
        class_weights = False
        balance_data = False
        # get different splits - the processor will return a dataframe and class_labels for each, but we only need training class_labels
        train_df, class_labels = Processor().get_examples(data_dir = root_data_dir, mode = "train", class_weights = class_weights, balance_data = balance_data)
        val_df,_ = Processor().get_examples(data_dir = root_data_dir, mode = "valid", class_weights = class_weights, balance_data = balance_data)
        test_df,_ = Processor().get_examples(data_dir = root_data_dir, mode = "test", class_weights = class_weights, balance_data = balance_data)  


        # few few shot sample

        support_sampler = FewShotSampler(num_examples_per_label = args.few_shot_n, also_sample_dev=False, label_col = label_col)
        # now apply to each dataframe but convert to dictionary in records form first
        train_df = support_sampler(train_df.to_dict(orient="records"), seed = 1)
        val_df = support_sampler(val_df.to_dict(orient="records"), seed = 1)
        test_df = support_sampler(test_df.to_dict(orient="records"), seed = 1)
        # load model
        model = hyperMimicBertModel(bert_model=pretrained_model_name,
                                 num_labels=len(class_labels),
                                 n_warmup_steps=warmup_steps,
                                 n_training_steps=total_training_steps,
                                 nr_frozen_epochs=args.nr_frozen_epochs,
                                    config = config
                                 )

    else:
        #TODO implement los and mimic readmission
        raise NotImplementedError
    

    # instantiate datamodule
    data_module = MimicDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=config["batch_size"],
        max_token_len=max_tokens,
        label_col = label_col,
        num_workers=1,
    )

    metrics = {"loss": "valid/loss", "f1_weighted":"valid/f1_weighted", "balanced_acc": "valid/balanced_accuracy"}

    
    # set pl metric callback
    metric_callback = PyTorchLightningPruningCallback(trial, monitor="monitor_balanced_accuracy")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=[args.gpus],
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=config['gradient_accum_steps'],
        callbacks=[metric_callback],
        checkpoint_callback = False)
    
    trainer.fit(model, data_module)
    ####################################
    score = trainer.callback_metrics["valid/balanced_accuracy"].item()
    logger.warning("trainer callback metrics is: ",{trainer.callback_metrics["valid/balanced_accuracy"]})

    return score
    
# run the actual analysis now


def objective(trial):

    
    # create optuna config
    config = {
        "plm_lr":trial.suggest_loguniform('plm_lr',1e-5, 1e-1),
        "classifier_lr":trial.suggest_loguniform('classifier_lr',1e-5, 3e-1),
        "batch_size": 4,
        "gradient_accum_steps":trial.suggest_int('gradient_accum_steps',2,10),
        "dropout": trial.suggest_float('dropout',0.1,0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw","adafactor"]),
        "classifier_hidden_size": 768
    }

    logger.warning(f"config inside objective is: {config}")

    # get the accuracy as the metric to monitor over trials
    accuracy = train_mimic(trial, config = config, num_epochs=args.num_epochs)

    return accuracy
    

# create optuna study
study_name = 'mimic_optuna_study'

# set up a study object with a median pruner - which will stop any trials that are too far from the median trial results
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                pruner= optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps = 5, interval_steps = 20),
                                 study_name = study_name)

logger.warning(f"Will be optimizing {study_name} with {args.n_trials} trials!")

study.optimize(objective, n_trials=args.n_trials) # add n_jobs = n if you want multiple runs in parallel - seems to act funny though

# try the multi gpu option

# study.optimize(Objective(GpuQueue()), n_trials=5, n_jobs=8)

# check if the logsdir exists

log_dir = save_dir

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# save it 
logger.warning(f"Saving the study to the following location: {log_dir}!!")
joblib.dump(study,f"{log_dir}/{study_name}.pkl")