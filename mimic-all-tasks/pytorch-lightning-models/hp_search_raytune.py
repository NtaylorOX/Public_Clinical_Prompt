import os
from sched import scheduler
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='6,7 '

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
import os
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
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

parser.add_argument("--num_samples",
                    default=100,
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
                    default = "/mnt/sdg/niallt/saved_models/mimic-tasks/pytorch-lightning-models/ray_tune_results/",
                    type=str,
                    help = "The data path to the directory for ray tune results"
                    )
parser.add_argument(
    "--nr_frozen_epochs",
    default=0,
    type=int,
    help="Number of epochs we want to keep the encoder model frozen.",
)

# parse the argument files
args = parser.parse_args()
# set constants
num_samples = args.num_samples
num_epochs = args.num_epochs
num_gpus = args.gpus_per_trial
gpus_per_trial = args.gpus_per_trial # set this to higher if using GPU
dataset = args.dataset
pretrained_model_name = args.pretrained_plm_name
root_data_dir = args.root_data_dir
max_tokens = args.max_tokens
label_col = args.label_col
nr_frozen_epochs = args.nr_frozen_epochs

# set save_dir based on the dataset provided and whether frozen
if nr_frozen_epochs > 0:
    save_dir = f"{args.save_dir}/{dataset}/frozen_plm/"
else:
    save_dir = f"{args.save_dir}/{dataset}/"

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
        self.lr = config['lr']
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

        print(f"Labels: {labels}  \nand num labels is: {len(labels)}")
        print("\n")
        print(f"Predicted labels: {predictions}")
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average = 'weighted')   

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
                    "lr": self.lr,
                },
            ]
            
            if self.optimizer == "adamw":
                optimizer = AdamW(parameters, lr=self.lr)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.n_warmup_steps,
                    num_training_steps=self.n_training_steps
                )
            elif self.optimizer == "adafactor":
                optimizer = Adafactor(parameters,  
                                    lr=self.lr,
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

def train_mimic(config, dataset = dataset,
                pretrained_model_name = pretrained_model_name,
                root_data_dir = root_data_dir, 
                max_tokens = max_tokens, label_col = label_col,
                 num_epochs=num_epochs, warmup_steps = 50, total_training_steps = 20000, 
                num_gpus=num_gpus, data_dir = f"{args.save_dir}"):
    

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

    
    # this is a scheduler that raytune uses to stop any runs that are likely going nowhere
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)
        # push data through pipeline
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
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
            # set logger
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=config['grad_accum_steps'],
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    
    trainer.fit(model, data_module)
    ####################################

    
# run the actual analysis now

# Defining a search space!
config = {
"batch_size": tune.choice([4]),
"grad_accum_steps": tune.choice([2,5,10]),   
"lr": tune.loguniform(1e-5,1e-1),
"dropout": tune.choice([0.1,0.2,0.5]),
"optimizer": tune.choice(['adamw']),
"classifier_hidden_size": tune.choice([768,768*2,768*3]),
}

# create the trainable ray tune class
trainable = tune.with_parameters(
    train_mimic,   
    num_epochs=num_epochs,
    num_gpus=args.gpus_per_trial,
    data_dir = args.save_dir)
# run the analysis
analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": args.gpus_per_trial
    },
    metric="loss",
    mode="min",
    config=config,    
    num_samples=num_samples,
    local_dir = f"{save_dir}",
    name=f"tune_mimic_{dataset}"
    
    )
import ray
ray.shutdown()


print(f"Best config based on ray tune analysis!:\n {analysis.best_config}")
    

