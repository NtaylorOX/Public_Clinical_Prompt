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
# from openprompt.utils.logging import logger
from loguru import logger

from utils import Mimic_ICD9_Processor, Mimic_ICD9_Triage_Processor, Mimic_Mortality_Processor,Mimic_LOS_Processor, customPromptDataLoader
import time
import os
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from utils import SummaryWriter # this version is better for logging hparams with metrics..
from utils import get_n_trainable_params_prompt_model

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
parser.add_argument("--freeze_verbalizer_plm",action = "store_true")
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--few_shot_n", type=int, default = 100)
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--run_evaluation",action="store_true")
parser.add_argument("--model", type=str, default='t5', help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name_or_path", default='t5-base')
parser.add_argument("--project_root", default="/mnt/sdg/niallt/saved_models/mimic-tasks/prompt-based-models/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int, default = 2)
parser.add_argument("--verbalizer_id", type=int, default = 0)
parser.add_argument("--template_type", type=str, default ="manual")
parser.add_argument("--verbalizer_type", type=str, default ="soft")
parser.add_argument("--data_dir", type=str, default="../mimic3-icd9-data/intermediary-data/") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dataset",type=str, default = "icd9_50") # or "icd9_triage"
parser.add_argument("--result_file", type=str, default="./mimic_icd9_top50/st_results/results.txt")
parser.add_argument("--scripts_path", type=str, default="./scripts/")
parser.add_argument("--class_labels_file", type=str, default="./scripts/mimic_icd9_top50/labels.txt")
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
parser.add_argument("--training_size", type=str, default="full") # or fewshot or zero
parser.add_argument("--no_ckpt", action="store_true")

parser.add_argument(
        '--sensitivity',
        default=False,
        type=bool,
        help='Run sensitivity trials - investigating the influence of classifier hidden dimension on performance in frozen plm setting.'
    )

parser.add_argument(
    '--optimized_run',
    default=False,
    type=bool,
    help='Run the optimized frozen model after hp search '
)
# instatiate args and set to variable
args = parser.parse_args()

logger.info(f" arguments provided were: {args}")

import random

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm


# set up some variables to add to checkpoint and logs filenames
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
version = f"version_{time_now}"


plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# edit based on whether or not plm was frozen during training
# actually want to save the checkpoints and logs in same place now. Becomes a lot easier to manage later
#TODO - add sensitivity based dirs here
if args.tune_plm == True:
    logger.warning("Unfreezing the plm - will be updated during training")
    freeze_plm = False
    # set checkpoint, logs and params save_dirs
    if args.sensitivity:
        logger.warning(f"performing sensitivity analysis experiment!")
        logs_dir = f"{args.project_root}/logs/sensitivity/{args.dataset}/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"
    elif args.optimized_run:
        logger.warning(f"performing optimized run!")
        logs_dir = f"{args.project_root}/logs/optimized/{args.dataset}/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"

    else:
        logs_dir = f"{args.project_root}/logs/{args.dataset}/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"
else:
    logger.warning("Freezing the plm")
    freeze_plm = True
    # we have to account for the slight issue with softverbalizer being wrongly implemented by openprompt
    # here we have an extra agument which will correctly freeze the PLM parts of softverbalizer if set to true
    if args.freeze_verbalizer_plm and args.verbalizer_type == "soft":
        logger.warning("also will be explicitly freezing plm parts of the soft verbalizer")
        if args.sensitivity:
            logger.warning(f"performing sensitivity analysis experiment!")
            logs_dir = f"{args.project_root}/logs/sensitivity/{args.dataset}/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_frozenverb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            logs_dir = f"{args.project_root}/logs/{args.dataset}/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_frozenverb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/" 

    else:# set checkpoint, logs and params save_dirs    
        if args.sensitivity:
            logger.warning(f"performing sensitivity analysis experiment!")
            logs_dir = f"{args.project_root}/logs/sensitivity/{args.dataset}/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        elif args.optimized_run:
            logger.warning(f"performing optimized run!")
            logs_dir = f"{args.project_root}/logs/optimized/{args.dataset}/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            logs_dir = f"{args.project_root}/logs/{args.dataset}/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"        
# check if the checkpoint and params dir exists   

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# set up tensorboard logger
writer = SummaryWriter(logs_dir)

# initialise empty dataset
dataset = {}

# crude setting of sampler to None - changed for mortality with umbalanced dataset

sampler = None
# Below are multiple dataset examples, although right now just mimic ic9-top50. 
if args.dataset == "icd9_50":
    logger.warning(f"Using the following dataset: {args.dataset} ")
    Processor = Mimic_ICD9_Processor
    # update data_dir
    data_dir = f"{args.data_dir}/top_50_icd9"

    # are we doing any downsampling or balancing etc
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
    scriptsbase = f"{args.scripts_path}/mimic_icd9_top50/"
    scriptformat = "txt"
    max_seq_l = 480 # this should be specified according to the running GPU's capacity 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.batch_size 
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False


elif args.dataset == "icd9_triage":
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
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.batch_size 
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False

# this is a heavily imbalanced dataset so we also calculate weight classes and return these for training set
elif args.dataset == "mortality":
    logger.warning(f"Using the following dataset: {args.dataset} ")
    Processor = Mimic_Mortality_Processor
    # update data_dir
    data_dir = "../clinical-outcomes-data/mimic3-clinical-outcomes/mp/"

    # we dont want to do anything with ce loss anymore
    #TODO - add tese to args parse
    ce_class_weights = args.ce_class_weights
    sampler_weights = args.sampler_weights  

    # balance_data = args.balance_data
    # its easier to just presume we want to for the mortality dataset - results are crazy biased if we don't - and downsampling produces fairest results  
    args.balance_data = True
    balance_data = True

    # cant set both these to true
    logger.warning(f"ce_class weights : {ce_class_weights} and sampler weights: {sampler_weights}")
    assert ce_class_weights != True and sampler_weights != True, "can't have both ce class weights and sampler class weights set to True"

    # set balance_data to true and only return the samples
    if balance_data:
        logger.warning(f"Will be balancing the dataset! i.e Downsampling!")
        dataset['train'] = Processor().get_examples(data_dir = data_dir, mode = "train", balance_data = balance_data, class_weights=False, sampler_weights= False)
    # if getting these weights need to return both the samples and weights
    elif ce_class_weights or sampler_weights:
        logger.warning(f"Will getting class weights!")
    # get different splits - we want to return class weights for training set!
        dataset['train'], task_class_weights = Processor().get_examples(data_dir = data_dir, mode = "train", balance_data = False, class_weights=ce_class_weights, sampler_weights= sampler_weights)

    else:
        dataset['train'] = Processor().get_examples(data_dir = data_dir, mode = "train", balance_data = False, class_weights=False, sampler_weights= False)

    # if we wanna run through a small sample
    if args.dev_run:
        logger.warning("PEFORMING DEV RUN - LOGS WILL BE SAVED BUT NO CHECKPOINT! WILL ALSO USE A SUBSET OF DATA!")
        dataset['train'] = dataset['train'][:1000]
        dataset['validation'] = Processor().get_examples(data_dir = data_dir, mode = "valid", balance_data = False, class_weights=False, sampler_weights= False)[:300]
        dataset['test'] = Processor().get_examples(data_dir = data_dir, mode = "test", balance_data = False, class_weights=False, sampler_weights= False)[:300]
    else:
        dataset['validation'] = Processor().get_examples(data_dir = data_dir, mode = "valid", balance_data = False, class_weights=False, sampler_weights= False)
        dataset['test'] = Processor().get_examples(data_dir = data_dir, mode = "test", balance_data = False, class_weights=False, sampler_weights= False)

    # now set sampler based on these weights  - this will mean during batch loading the dataloader samples based on these weights
    if sampler_weights:
        sampler = WeightedRandomSampler(task_class_weights, len(task_class_weights), replacement = True)
    # the below class labels should align with the label encoder fitted to training data
    # you will need to generate this class label text file first using the mimic processor with generate_class_labels flag to set true
    # e.g. Processor().get_examples(data_dir = args.data_dir, mode = "train", generate_class_labels = True)[:10000]
    class_labels = Processor().load_class_labels()
    print(f"class labels: {class_labels}")
    print(f"number of classes: {len(class_labels)}")
    scriptsbase = f"{args.scripts_path}/mimic_mortality/"
    scriptformat = "txt"
    max_seq_l = 480 # this should be specified according to the running GPU's capacity 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.batch_size 
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False
# length of stay 
elif args.dataset == "los":
    logger.warning(f"Using the following dataset: {args.dataset} ")
    Processor = Mimic_LOS_Processor
    # update data_dir
    data_dir = "/mnt/sdd/yizhang/projects/data/mimic_los/"

    # we dont want to do anything with ce loss anymore
    #TODO - add tese to args parse
    ce_class_weights = args.ce_class_weights
    sampler_weights = args.sampler_weights    
    balance_data = args.balance_data

    # cant set both these to true
    logger.warning(f"ce_class weights : {ce_class_weights} and sampler weights: {sampler_weights}")
    assert ce_class_weights != True and sampler_weights != True, "can't have both ce class weights and sampler class weights set to True"

    # set balance_data to true and only return the samples
    if balance_data:
        logger.warning(f"Will be balancing the dataset! i.e Downsampling!")
        dataset['train'] = Processor().get_examples(data_dir = data_dir, mode = "train", balance_data = balance_data, class_weights=False, sampler_weights= False)
    # if getting these weights need to return both the samples and weights
    elif ce_class_weights or sampler_weights:
        logger.warning(f"Will getting class weights!")
    # get different splits - we want to return class weights for training set!
        dataset['train'], task_class_weights = Processor().get_examples(data_dir = data_dir, mode = "train", balance_data = False, class_weights=ce_class_weights, sampler_weights= sampler_weights)

    else:
        dataset['train'] = Processor().get_examples(data_dir = data_dir, mode = "train", balance_data = False, class_weights=False, sampler_weights= False)

    # if we wanna run through a small sample
    if args.dev_run:
        logger.warning("PEFORMING DEV RUN - LOGS WILL BE SAVED BUT NO CHECKPOINT! WILL ALSO USE A SUBSET OF DATA!")
        dataset['train'] = dataset['train'][:1000]
        dataset['validation'] = Processor().get_examples(data_dir = data_dir, mode = "valid", balance_data = False, class_weights=False, sampler_weights= False)[:300]
        dataset['test'] = Processor().get_examples(data_dir = data_dir, mode = "test", balance_data = False, class_weights=False, sampler_weights= False)[:300]
    else:
        dataset['validation'] = Processor().get_examples(data_dir = data_dir, mode = "valid", balance_data = False, class_weights=False, sampler_weights= False)
        dataset['test'] = Processor().get_examples(data_dir = data_dir, mode = "test", balance_data = False, class_weights=False, sampler_weights= False)

    # now set sampler based on these weights  - this will mean during batch loading the dataloader samples based on these weights
    if sampler_weights:
        sampler = WeightedRandomSampler(task_class_weights, len(task_class_weights), replacement = True)
    # the below class labels should align with the label encoder fitted to training data
    # you will need to generate this class label text file first using the mimic processor with generate_class_labels flag to set true
    # e.g. Processor().get_examples(data_dir = args.data_dir, mode = "train", generate_class_labels = True)[:10000]
    class_labels =Processor().load_class_labels()
    print(f"class labels: {class_labels}")
    print(f"number of classes: {len(class_labels)}")
    scriptsbase = f"{args.scripts_path}/mimic_los"
    scriptformat = "txt"
    max_seq_l = 480 # this should be specified according to the running GPU's capacity 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.batch_size 
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False

else:
    
    #TODO implement los and mimic readmission
    raise NotImplementedError

# write hparams to file
# lets write these arguments to file for later loading alongside the trained models
with open(f'{ckpt_dir}/hparams.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
# add the hparams to tensorboard logs
# print(f"hparams dict: {args.__dict__}")
save_metrics = {"random/metric": 0}
writer.add_hparams(args.__dict__, save_metrics)

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

    # we noticed a bug where soft verbalizer was technically not freezing alongside the PLM - meaning it had considerably greater number of trainable parameters
    # so if we want to properly freeze the verbalizer plm components as described here: https://github.com/thunlp/OpenPrompt/blob/4ba7cb380e7b42c19d566e9836dce7efdb2cc235/openprompt/prompts/soft_verbalizer.py#L82 
    # we now need to actively set grouped_parameters_1 to requires_grad = False
    if args.freeze_verbalizer_plm and freeze_plm:
        logger.warning(f"We have a soft verbalizer and want to freeze it alongside the PLM!")
        # now set the grouped_parameters_1 require grad to False
        for param in myverbalizer.group_parameters_1:
            param.requires_grad = False

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 


# are we using cuda and if so which number of device
use_cuda = True
if use_cuda:
    cuda_device = torch.device(f'cuda:{args.gpu_num}')
else:
    cuda_device = torch.device('cpu')

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

    # can try without also fewshot sampling val and test sets?
    # dataset['validation'] = support_sampler(dataset['validation'], seed=args.seed)
    # dataset['test'] = support_sampler(dataset['test'], seed=args.seed)

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

print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)

from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

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
    optimizer_plm = AdamW(optimizer_grouped_parameters_plm, lr=args.plm_lr)
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
                                lr=args.prompt_lr,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        scheduler_template = get_constant_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
        optimizer_template = AdamW(optimizer_grouped_parameters_template, lr=args.prompt_lr) # usually lr = 0.5
        scheduler_template = get_linear_schedule_with_warmup(
                        optimizer_template, 
                        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

elif args.template_type == "manual":
    optimizer_template = None
    scheduler_template = None


if args.verbalizer_type == "soft":
    logger.warning("Soft verbalizer used - will be fine tuning the verbalizer/answer embeddings!")
    optimizer_grouped_parameters_verb = [
    {'params': prompt_model.verbalizer.group_parameters_1, "lr":args.plm_lr},
    {'params': prompt_model.verbalizer.group_parameters_2, "lr":args.plm_lr},
    
    ]
    optimizer_verb= AdamW(optimizer_grouped_parameters_verb)
    scheduler_verb = get_linear_schedule_with_warmup(
                        optimizer_verb, 
                        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

elif args.verbalizer_type == "manual":
    optimizer_verb = None
    scheduler_verb = None


def train(prompt_model, train_dataloader, num_epochs, mode = "train", ckpt_dir = ckpt_dir):

    logger.warning(f"cuda current device inside training is: {torch.cuda.current_device()}")
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
                writer.add_scalar("train/batch_loss", aveloss, glb_step)        

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
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        
        # run a run through validation set to get some metrics        
        val_loss, val_acc, val_prec_weighted, val_prec_macro, val_recall_weighted,val_recall_macro, val_f1_weighted,val_f1_macro, val_auc_weighted,val_auc_macro, cm_figure = evaluate(prompt_model, validation_dataloader)

        writer.add_scalar("valid/loss", val_loss, epoch)
        writer.add_scalar("valid/balanced_accuracy", val_acc, epoch)
        writer.add_scalar("valid/precision_weighted", val_prec_weighted, epoch)
        writer.add_scalar("valid/precision_macro", val_prec_macro, epoch)
        writer.add_scalar("valid/recall_weighted", val_recall_weighted, epoch)
        writer.add_scalar("valid/recall_macro", val_recall_macro, epoch)
        writer.add_scalar("valid/f1_weighted", val_f1_weighted, epoch)
        writer.add_scalar("valid/f1_macro", val_f1_macro, epoch)

        #TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("valid/auc_weighted", val_auc_weighted, epoch)
        writer.add_scalar("valid/auc_macro", val_auc_macro, epoch)        

        # add cm to tensorboard
        writer.add_figure("valid/Confusion_Matrix", cm_figure, epoch)

        # save checkpoint if validation accuracy improved
        if val_acc >= best_val_acc:
            # only save ckpts if no_ckpt is False - we do not always want to save - especially when developing code
            if not args.no_ckpt:
                logger.warning(f"Accuracy improved! Saving checkpoint at :{ckpt_dir}!")
                torch.save(prompt_model.state_dict(),f"{ckpt_dir}/best-checkpoint.ckpt")
            best_val_acc = val_acc


        if glb_step > args.max_steps:
            leave_training = True
            break
    
        if leave_training:
            logger.warning("Leaving training as max steps have been met!")
            break 


# ## evaluate

# %%

def evaluate(prompt_model, dataloader, mode = "validation", class_labels = class_labels):

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

    # get confusion matrix
    cm = confusion_matrix(alllabels, allpreds)

    # plot using custom function defined below
    # cm_figure = plotConfusionMatrix(cm, class_labels)
    # below makes a slightly nicer plot 
    cm_figure = plot_confusion_matrix(cm, class_labels)

    # if we are doing final evaluation on test data - save labels, pred_labels, logits and some plots
    if mode == 'test':
        
        # create empty dict to store labels, pred_labels, logits
        results_dict = {}
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file as well as tensorboard!")
        # classification report
        print(classification_report(alllabels, allpreds, target_names=class_labels))
        
        # save to dict
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        # now to file
        test_report_df = pd.DataFrame(test_report).transpose()
        test_report_df.to_csv(f"{ckpt_dir}/test_class_report.csv", index = False)
        
        # save logits etc
        
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        # save dataframe and to csv
        pd.DataFrame(results_dict).to_csv(f"{ckpt_dir}/test_results.csv", index =False)

        # save confusion matrix
        cm_figure.savefig(f"{ckpt_dir}/test_cm.png")

    
    return val_loss, acc, prec_weighted, prec_macro, recall_weighted, recall_macro, f1_weighted, f1_macro, roc_auc_weighted, roc_auc_macro, cm_figure


# TODO - add a test function to load the best checkpoint and obtain metrics on all test data. Can do this post training but may be nicer to do after training to avoid having to repeat.

def test_evaluation(prompt_model, ckpt_dir, dataloader):
    # once model is trained we want to load best checkpoint back in and evaluate on test set - then log to tensorboard and save logits and few other things to file?

    # first load the state_dict using the ckpt_dir of the best model i.e. should be best-checkpoint.ckpt found in the ckpt_dir

    loaded_model = torch.load(f"{ckpt_dir}/best-checkpoint.ckpt")

    # now load this into the already create PromptForClassification object i.e. the supplied prompt_model
    prompt_model.load_state_dict(state_dict = loaded_model)

    # then run evaluation on test_dataloader

    test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted,test_recall_macro, test_f1_weighted,test_f1_macro, test_auc_weighted,test_auc_macro, cm_figure = evaluate(prompt_model,
                                                                                                                    mode = 'test', dataloader = dataloader)

    # write to tensorboard

    writer.add_scalar("test/loss", test_loss, 0)
    writer.add_scalar("test/balanced_accuracy", test_acc, 0)
    writer.add_scalar("test/precision_weighted", test_prec_weighted, 0)
    writer.add_scalar("test/precision_macro", test_prec_macro, 0)
    writer.add_scalar("test/recall_weighted", test_recall_weighted, 0)
    writer.add_scalar("test/recall_macro", test_recall_macro, 0)
    writer.add_scalar("test/f1_weighted", test_f1_weighted, 0)
    writer.add_scalar("test/f1_macro", test_f1_macro, 0)

    #TODO add binary classification metrics e.g. roc/auc
    writer.add_scalar("test/auc_weighted", test_auc_weighted, 0)
    writer.add_scalar("test/auc_macro", test_auc_macro, 0)        

    # add cm to tensorboard
    writer.add_figure("test/Confusion_Matrix", cm_figure, 0)

# nicer plot
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes

    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

   
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('normal')

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: {args.dataset}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.90
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # figure.savefig(f'experiments/{model}/test_mtx.png')

    return figure


# if refactor this has to be run before any training has occured
if args.zero_shot:
    logger.info("Obtaining zero shot performance on test set!")
    
    zero_loss, zero_acc, zero_prec_weighted, zero_prec_macro, zero_recall_weighted,zero_recall_macro, zero_f1_weighted,zero_f1_macro, zero_auc_weighted,zero_auc_macro, zero_cm_figure = evaluate(prompt_model, test_dataloader)


    writer.add_scalar("zero_shot/loss", zero_loss, 0)
    writer.add_scalar("zero_shot/balanced_accuracy", zero_acc, 0)
    writer.add_scalar("zero_shot/precision_weighted", zero_prec_weighted, 0)
    writer.add_scalar("zero_shot/precision_macro", zero_prec_macro, 0)
    writer.add_scalar("zero_shot/recall_weighted", zero_recall_weighted, 0)
    writer.add_scalar("zero_shot/recall_macro", zero_recall_macro, 0)
    writer.add_scalar("zero_shot/f1_weighted", zero_f1_weighted, 0)
    writer.add_scalar("zero_shot/f1_macro", zero_f1_macro, 0)

    #TODO add binary classification metrics e.g. roc/auc
    writer.add_scalar("zero_shot/auc_weighted", zero_auc_weighted, 0)
    writer.add_scalar("zero_shot/auc_macro", zero_auc_macro, 0)
    

    # add cm to tensorboard
    writer.add_figure("zero_shot/Confusion_Matrix", zero_cm_figure, 0)

# run training

logger.warning(f"do training : {do_training}")
if do_training:
    logger.warning("Beginning full training!")
    train(prompt_model, train_dataloader, args.num_epochs, ckpt_dir)

elif not do_training:
    logger.warning("No training will be performed!")

# run on test set if desired

if args.run_evaluation:
    logger.warning("Running evaluation on test set using best checkpoint!")
    test_evaluation(prompt_model, ckpt_dir, test_dataloader)
# write the contents to file

writer.flush()

