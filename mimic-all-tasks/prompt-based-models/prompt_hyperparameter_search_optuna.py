import os

import joblib
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='6,7'

from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import optuna


from typing import Dict
from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import pandas as pd
import seaborn as sn

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, ManualTemplate, SoftVerbalizer

from openprompt.prompts import SoftTemplate, MixedTemplate
from openprompt import PromptForClassification

import random

from openprompt.utils.reproduciblity import set_seed


from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm


from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

# from openprompt.utils.logging import logger
from loguru import logger

from utils import Mimic_ICD9_Processor, Mimic_ICD9_Triage_Processor, Mimic_Mortality_Processor, customPromptDataLoader
import time
import os
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from utils import SummaryWriter # this version is better for logging hparams with metrics..

import torchmetrics.functional.classification as metrics
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score 

from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import json
import itertools
from collections import Counter

import os 
# # Kill all processes on GPU 6 and 7
# os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 6 && $2 < 7 {print $5}')""")

'''
Script to run different setups of prompt learning.

Right now this is primarily set up for the mimic_top50_icd9 task, although it is quite flexible to other datasets. Any datasets need a corresponding processor class in utils.


example usage. python prompt_experiment_runner.py --model bert --model_name_or_path bert-base-uncased --num_epochs 10 --tune_plm

other example usage:
- python prompt_experiment_runner.py --model t5 --model_name_or_path razent/SciFive-base-Pubmed_PMC --num_epochs 10 --template_id 0 --template_type soft --max_steps 15000 --tune_plm


'''


# create a args parser with required arguments.
parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--few_shot_n", type=int, default = 128)
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--run_evaluation",action="store_true")
parser.add_argument("--model", type=str, default='bert', help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name_or_path", default='emilyalsentzer/Bio_ClinicalBERT')
parser.add_argument("--project_root", default="/mnt/sdg/niallt/saved_models/mimic-tasks/prompt-based-models/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int, default = 2)
parser.add_argument("--verbalizer_id", type=int, default = 0)
parser.add_argument("--template_type", type=str, default ="mixed")
parser.add_argument("--verbalizer_type", type=str, default ="soft")
parser.add_argument("--data_dir", type=str, default="../mimic3-icd9-data/intermediary-data/") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dataset",type=str, default = "icd9_triage") # or "icd9_triage"
parser.add_argument("--scripts_path", type=str, default="./scripts/")
parser.add_argument("--max_steps", default=15000, type=int)
parser.add_argument("--plm_lr", type=float, default=1e-05)
parser.add_argument("--plm_warmup_steps", type=float, default=50)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--init_from_vocab", action="store_true")
parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="adafactor")   
parser.add_argument("--gradient_accum_steps", type = int, default = 10)
parser.add_argument("--dev_run",action="store_true")
parser.add_argument("--gpu_num", type=int, default = 0)
parser.add_argument("--balance_data", action="store_true") # whether to downsample data to majority class
parser.add_argument("--ce_class_weights", action="store_true") # whether to apply class weights to cross entropy loss fn
parser.add_argument("--sampler_weights", action="store_true") # apply weights to weighted data sampler
parser.add_argument("--training_size", type=str, default="fewshot") # or fewshot or zero
parser.add_argument("--no_ckpt", action="store_true")
parser.add_argument("--gpus_per_trial",
                    default=1,
                    type=int,
                    help="The number of individual gpus to use. Set to 0 for cpu.")

parser.add_argument("--n_trials",
                    default=30,
                    type=int,
                    help="The number of samples to take from hyperparameter search space.")
parser.add_argument("--save_dir",
                    default = "/mnt/sdg/niallt/saved_models/mimic-tasks/prompt-based-models/hp-search/optuna_results/",
                    type=str,
                    help = "The data path to the directory for ray tune results"
                    )


# some classes to assist with multi-gpu running
from contextlib import contextmanager
import multiprocessing
N_GPUS = 2

class GpuQueue:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


class Objective:

    def __init__(self, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            best_val_loss = train_mimic(**trial.params, gpu=gpu_i)
            return best_val_loss




# instatiate args and set to variable
args = parser.parse_args()

logger.info(f" arguments provided were: {args}")


# set seed 
set_seed(args.seed)
# set up some variables to add to checkpoint and logs filenames
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
version = f"version_{time_now}"

    # actually want to save the checkpoints and logs in same place now. Becomes a lot easier to manage later
if args.tune_plm == True:
    logger.warning("Unfreezing the plm - will be updated during training")
    freeze_plm = False
    # set checkpoint, logs and params save_dirs    
    logs_dir = f"{args.save_dir}/{args.dataset}/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/"

else:
    logger.warning("Freezing the plm")
    freeze_plm = True
    # set checkpoint, logs and params save_dirs    
    logs_dir = f"{args.save_dir}/{args.dataset}/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/"

# set up tensorboard logger
# writer = SummaryWriter(logs_dir)

def train_mimic(trial,num_epochs, mode = "train", 
                ckpt_dir = None, dataset = "icd9_triage",
                data_dir = args.data_dir,
                config = None):

    '''
    Function to run the training procedure with a optuna trial class object:
    
    '''

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    # edit based on whether or not plm was frozen during training


    # initialise empty dataset
    dataset = {}

    # crude setting of sampler to None - changed for mortality with umbalanced dataset

    sampler = None
    # Below are multiple dataset examples, although right now just mimic ic9-top50. 
    if args.dataset == "icd9_triage":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        Processor = Mimic_ICD9_Triage_Processor
        # update data_dir
        data_dir = f"{args.data_dir}/triage"

        ce_class_weights = args.ce_class_weights
        sampler_weights = args.sampler_weights    
        balance_data = args.balance_data

        # get different splits
        dataset['train'] = Processor().get_examples(data_dir = data_dir, mode = "train")
        dataset['validation'] = Processor().get_examples(data_dir = data_dir, mode = "valid")
        dataset['test'] = Processor().get_examples(data_dir = data_dir, mode = "test")
        # the below class labels should align with the label encoder fitted to training data
        # you will need to generate this class label text file first using the mimic processor with generate_class_labels flag to set true
        # e.g. Processor().get_examples(data_dir = args.data_dir, mode = "train", generate_class_labels = True)[:10000]
        class_labels =Processor().load_class_labels()
        print(f"number of classes: {len(class_labels)}")
        scriptsbase = f"{args.scripts_path}/mimic_triage/"
        scriptformat = "txt"
        max_seq_l = 480 # this should be specified according to the running GPU's capacity 
        
        batchsize_t = config['batch_size'] 
        batchsize_e = config['batch_size'] 
        gradient_accumulation_steps = config['gradient_accum_steps']
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize

    else:
        
        raise NotImplementedError


    # Now define the template and verbalizer. 
    # Note that soft template can be combined with hard template, by loading the hard template from file. 
    # For example, the template in soft_template.txt is {}
    # The choice_id 1 is the hard template 

    # decide which template and verbalizer to use
    if args.template_type == "manual":
        print(f"manual template selected, with id :{args.template_id}")
        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"{scriptsbase}/manual_template.txt", choice=args.template_id)

    elif args.template_type == "soft":
        print(f"soft template selected, with id :{args.template_id}")
        mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"{scriptsbase}/soft_template.txt", choice=args.template_id)


    elif args.template_type == "mixed":
        print(f"mixed template selected, with id :{args.template_id}")
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"{scriptsbase}/mixed_template.txt", choice=args.template_id)
    # now set verbalizer
    if args.verbalizer_type == "manual":
        print(f"manual verbalizer selected, with id :{args.verbalizer_id}")
        myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)

    elif args.verbalizer_type == "soft":
        print(f"soft verbalizer selected!")
        myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(class_labels))
    # are we using cuda and if so which number of device
    use_cuda = True
    
    cuda_device = "cpu"
    if use_cuda:
        if torch.cuda.is_available():
            # cuda_device = "cuda:0"
    
    
            cuda_device = torch.device(f'cuda:{args.gpu_num}')
    # now set the default gpu to this one
    torch.cuda.set_device(cuda_device)


    print(f"tune_plm value: {args.tune_plm}")
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=freeze_plm, plm_eval_mode=args.plm_eval_mode)
    if use_cuda:
        prompt_model=  prompt_model.to(cuda_device)

    if model_parallelize:
        prompt_model.parallelize()


    # if doing few shot learning - produce the datasets here:
    if args.training_size == "fewshot":
        logger.warning(f"Will be performing few shot learning.")
    # create the few_shot sampler for when we want to run training and testing with few shot learning
        support_sampler = FewShotSampler(num_examples_per_label = args.few_shot_n, also_sample_dev=False)

        # create a fewshot dataset from training, val and test. Seems to be what several papers do...
        dataset['train'] = support_sampler(dataset['train'], seed=args.seed)
        dataset['validation'] = support_sampler(dataset['validation'], seed=args.seed)
        dataset['test'] = support_sampler(dataset['test'], seed=args.seed)

    # are we doing training?
    do_training = (not args.no_training)
    if do_training:
        # if we have a sampler .e.g weightedrandomsampler. Do not shuffle
        if "WeightedRandom" in type(sampler).__name__:
            logger.warning("Sampler is WeightedRandom - will not be shuffling training data!")
            shuffle = False
        else:
            shuffle = True
        logger.warning(f"Do training is True - creating train and validation dataloders!")
        train_dataloader = customPromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
            batch_size=batchsize_t,shuffle=shuffle, sampler = sampler, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")

        validation_dataloader = customPromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
            batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")


    # zero-shot test
    test_dataloader = customPromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
        batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")


    #TODO update this to handle class weights for imabalanced datasets
    if ce_class_weights:
        logger.warning("we have some task specific class weights - passing to CE loss")
        # get from the class_weight function
        # task_class_weights = torch.tensor(task_class_weights, dtype=torch.float).to(cuda_device)
        
        # set manually cause above didnt work
        task_class_weights = torch.tensor([1,16.1], dtype=torch.float).to(cuda_device)
        loss_func = torch.nn.CrossEntropyLoss(weight = task_class_weights, reduction = 'mean')
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    # get total steps as a function of the max epochs, batch_size and len of dataloader
    tot_step = args.max_steps

    if args.tune_plm:
        
        logger.warning("We will be tuning the PLM!") # normally we freeze the model when using soft_template. However, we keep the option to tune plm
        no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters_plm = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_plm = AdamW(optimizer_grouped_parameters_plm, lr=config['plm_lr'])
        scheduler_plm = get_linear_schedule_with_warmup(
            optimizer_plm, 
            num_warmup_steps=args.plm_warmup_steps, num_training_steps=tot_step)
    else:
        logger.warning("We will not be tunning the plm - i.e. the PLM layers are frozen during training")
        optimizer_plm = None
        scheduler_plm = None

    # if using soft template
    if args.template_type == "soft" or args.template_type == "mixed":
        logger.warning(f"{args.template_type} template used - will be fine tuning the prompt embeddings!")
        optimizer_grouped_parameters_template = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
        if args.optimizer.lower() == "adafactor":
            optimizer_template = Adafactor(optimizer_grouped_parameters_template,  
                                    lr=config['prompt_lr'],
                                    relative_step=False,
                                    scale_parameter=False,
                                    warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            scheduler_template = get_constant_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        elif args.optimizer.lower() == "adamw":
            optimizer_template = AdamW(optimizer_grouped_parameters_template, lr=config['prompt_lr']) # usually lr = 0.5
            scheduler_template = get_linear_schedule_with_warmup(
                            optimizer_template, 
                            num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

    elif args.template_type == "manual":
        optimizer_template = None
        scheduler_template = None


    if args.verbalizer_type == "soft":
        logger.warning("Soft verbalizer used - will be fine tuning the verbalizer/answer embeddings!")
        optimizer_grouped_parameters_verb = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":config['plm_lr']},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":config['plm_lr']}        
        ]
        optimizer_verb= AdamW(optimizer_grouped_parameters_verb)
        scheduler_verb = get_linear_schedule_with_warmup(
                            optimizer_verb, 
                            num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

    elif args.verbalizer_type == "manual":
        optimizer_verb = None
        scheduler_verb = None


    # set model to train 
    prompt_model.train()

    # set up some counters
    actual_step = 0
    glb_step = 0

    # some validation metrics to monitor
    best_val_acc = 0
    best_val_f1 = 0
    best_val_prec = 0    
    best_val_recall = 0

 

    # this will be set to true when max steps are reached
    leave_training = False

    for epoch in tqdm(range(num_epochs)):
        print(f"On epoch: {epoch}")
        tot_loss = 0 
        epoch_loss = 0
        for step, inputs in enumerate(train_dataloader):       

            if use_cuda:
                inputs = inputs.to(cuda_device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)

            # normalize loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps 

            # propogate backward to calculate gradients
            loss.backward()
            tot_loss += loss.item()

            actual_step+=1
            # log loss to tensorboard  every 50 steps    

            # clip gradients based on gradient accumulation steps
            if actual_step % gradient_accumulation_steps == 0:
                # log loss
                aveloss = tot_loss/(step+1)
                # write to tensorboard
                # writer.add_scalar("train/batch_loss", aveloss, glb_step)        

                # clip grads            
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                glb_step += 1

                # backprop the loss and update optimizers and then schedulers too
                # plm
                if optimizer_plm is not None:
                    optimizer_plm.step()
                    optimizer_plm.zero_grad()
                if scheduler_plm is not None:
                    scheduler_plm.step()
                # template
                if optimizer_template is not None:
                    optimizer_template.step()
                    optimizer_template.zero_grad()
                if scheduler_template is not None:
                    scheduler_template.step()
                # verbalizer
                if optimizer_verb is not None:
                    optimizer_verb.step()
                    optimizer_verb.zero_grad()
                if scheduler_verb is not None:
                    scheduler_verb.step()

                # check if we are over max steps
                if glb_step > args.max_steps:
                    logger.warning("max steps reached - stopping training!")
                    leave_training = True
                    break

        # get epoch loss and write to tensorboard

        epoch_loss = tot_loss/len(train_dataloader)
        print("Epoch {}, loss: {}".format(epoch, epoch_loss), flush=True)   
        # writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        
        # run a run through validation set to get some metrics        
        val_loss, val_acc, val_prec_weighted, val_prec_macro, val_recall_weighted,val_recall_macro, val_f1_weighted,val_f1_macro, val_auc_weighted,val_auc_macro = evaluate(prompt_model, validation_dataloader,
                                                                                                                                                                                        use_cuda=use_cuda, cuda_device = cuda_device,
                                                                                                                                                                                        loss_func = loss_func, class_labels = class_labels)

        if val_acc >= best_val_acc:
            # only save ckpts if no_ckpt is False - we do not always want to save - especially when developing code
            if ckpt_dir != None:
                logger.warning("Accuracy improved! Saving checkpoint!")
                # torch.save(prompt_model.state_dict(),f"{ckpt_dir}/best-checkpoint.ckpt")
            best_val_acc = val_acc

        # report the current epoch accuracy to the trial class object
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if glb_step > args.max_steps:
            leave_training = True
            break
    
        if leave_training:
            logger.warning("Leaving training as max steps have been met!")
            break 

        
    return best_val_acc
           
   
# ## evaluate

# %%

def evaluate(prompt_model, dataloader, mode = "validation", 
                class_labels = None, use_cuda = True, cuda_device = None, loss_func= None):

    prompt_model.eval()

    tot_loss = 0
    allpreds = []
    alllabels = []
    #record logits from the the model
    alllogits = []
    # store probabilties i.e. softmax applied to logits
    allscores = []

    allids = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.to(cuda_device)
            logits = prompt_model(inputs)
            labels = inputs['label']

            loss = loss_func(logits, labels)
            tot_loss += loss.item()

            # add labels to list
            alllabels.extend(labels.cpu().tolist())

            # add ids to list - they are already a list so no need to send to cpu
            allids.extend(inputs['guid'])

            # add logits to list
            alllogits.extend(logits.cpu().tolist())
            #use softmax to normalize, as the sum of probs should be 1
            # if binary classification we just want the positive class probabilities
            if len(class_labels) > 2:  
                allscores.extend(torch.nn.functional.softmax(logits).cpu().tolist())
            else:

                allscores.extend(torch.nn.functional.softmax(logits)[:,1].cpu().tolist())

            # add predicted labels    
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    
    val_loss = tot_loss/len(dataloader)    
    # get sklearn based metrics
    acc = balanced_accuracy_score(alllabels, allpreds)
    f1_weighted = f1_score(alllabels, allpreds, average = 'weighted')
    f1_macro = f1_score(alllabels, allpreds, average = 'macro')
    prec_weighted = precision_score(alllabels, allpreds, average = 'weighted')
    prec_macro = precision_score(alllabels, allpreds, average = 'macro')
    recall_weighted = recall_score(alllabels, allpreds, average = 'weighted')
    recall_macro = recall_score(alllabels, allpreds, average = 'macro')


    # roc_auc  - only really good for binary classification but can try for multiclass too
    # use scores instead of predicted labels to give probs
    
    if len(class_labels) > 2:   
        roc_auc_weighted = roc_auc_score(alllabels, allscores, average = "weighted", multi_class = "ovr")
        roc_auc_macro = roc_auc_score(alllabels, allscores, average = "macro", multi_class = "ovr")
                  
    else:
        roc_auc_weighted = roc_auc_score(alllabels, allscores, average = "weighted")
        roc_auc_macro = roc_auc_score(alllabels, allscores, average = "macro")         


    
   
    return val_loss, acc, prec_weighted, prec_macro, recall_weighted, recall_macro, f1_weighted, f1_macro, roc_auc_weighted, roc_auc_macro


# this is the objective function that optuna will use to train models with parameters produced by the config hp search
def objective(trial):

    # create optuna config
    config = {
        "plm_lr":trial.suggest_loguniform('plm_lr',1e-5, 1e-1),
        "prompt_lr":trial.suggest_loguniform('prompt_lr',1e-5, 3e-1),
        "batch_size": 4,
        "gradient_accum_steps":trial.suggest_int('gradient_accum_steps',2,10),
        "dropout": trial.suggest_float('dropout',0.1,0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw","adafactor"]),
    }
    # get the accuracy as the metric to monitor over trials
    accuracy = train_mimic(trial,config = config, num_epochs=args.num_epochs)

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

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
# save it 
logger.warning(f"Saving the study to the following location: {logs_dir}!!")
joblib.dump(study,f"{logs_dir}/{study_name}.pkl")
