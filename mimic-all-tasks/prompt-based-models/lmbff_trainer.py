# %% [markdown]
# ## Text Classification with LM-BFF.
# In this tutorial, we do sentiment analysis with automatic template and verbalizer generation. We use SST-2 as an example.

# %% [markdown]
# ### 1. load dataset

# %%
# import argparse
# parser = argparse.ArgumentParser("")
# parser.add_argument("--lr", type=float, default=5e-5)
# args = parser.parse_args()


from utils import Mimic_ICD9_Processor, Mimic_ICD9_Triage_Processor
from openprompt.data_utils.text_classification_dataset import SST2Processor
from loguru import logger

import time
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torchmetrics.functional.classification as metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import json
import itertools
import argparse


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
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--model", type=str, default='t5', help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name_or_path", default='t5-base')
parser.add_argument("--project_root", default="./", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int, default = 2)
parser.add_argument("--verbalizer_id", type=int, default = 0)
parser.add_argument("--template_type", type=str, default ="manual")
parser.add_argument("--verbalizer_type", type=str, default ="soft")
parser.add_argument("--data_dir", type=str, default="../data/intermediary-data/") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dataset",type=str, default = "icd9_50") # or "icd9_triage"
parser.add_argument("--result_file", type=str, default="./mimic_icd9_top50/st_results/results.txt")
parser.add_argument("--scripts_path", type=str, default="./scripts/")
parser.add_argument("--class_labels_file", type=str, default="./scripts/mimic_icd9_top50/labels.txt")
parser.add_argument("--max_steps", default=15000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--init_from_vocab", action="store_true")
parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="adamw")
# generator args
parser.add_argument("--temp_gen_model", type=str, default="t5")
parser.add_argument("--temp_gen_model_name_or_path", type=str, default="razent/SciFive-base-Pubmed_PMC")
parser.add_argument("--verb_gen_model_name_or_path", type=str, default="allenai/biomed_roberta_base")

# instatiate args and set to variable
args = parser.parse_args()


# write arguments to a txt file to go with the model checkpoint and logs
content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"tune_plm {args.tune_plm}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += "\n"
dataset = {}


# Below are multiple dataset examples, although right now just mimic ic9-top50. 
if args.dataset == "icd9_50":
    logger.warning(f"Using the following dataset: {args.dataset} ")
    Processor = Mimic_ICD9_Processor
    # update data_dir
    data_dir = f"{args.data_dir}/top_50_icd9"

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
        gradient_accumulation_steps = 4
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = 4
        model_parallelize = False

elif args.dataset == "icd9_triage":
    logger.warning(f"Using the following dataset: {args.dataset} ")
    Processor = Mimic_ICD9_Triage_Processor
    # update data_dir
    data_dir = f"{args.data_dir}/triage"

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
        gradient_accumulation_steps = 4
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = 4
        model_parallelize = False

elif args.dataset == "sst2":
    dataset = {}
    dataset['train'] = SST2Processor().get_train_examples("./datasets/TextClassification/SST-2/16-shot/16-13")
    dataset['validation'] = SST2Processor().get_dev_examples("./datasets/TextClassification/SST-2/16-shot/16-13")
    dataset['test'] = SST2Processor().get_test_examples("./datasets/TextClassification/SST-2/16-shot/16-13")
    #TODO implement icd9 triage and mimic readmission

else:  
    raise NotImplementedError

# %% [markdown]
# ### 2. build initial verbalizer and template
# - note that if you wish to do automaitc label word generation, the verbalizer is not the final verbalizer, and is only used for template generation.
# - note that if you wish to do automatic template generation, the template text may desirably include `{"meta":"labelword"}` so that label word can be used and remember to use `LMBFFTemplateGenerationTemplate` class so that "labelword" can be handled properly. Else you can just use `ManualTemplate`
# - below is a template that expects plain text generation at each "mask" token position

# %%

from openprompt.plms import load_plm
# load mlm model for main tasks
# set up some variables to add to checkpoint and logs filenames
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
version = f"version_{time_now}"


plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# load generation model for template generation
template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm(args.temp_gen_model, args.temp_gen_model_name_or_path)

from openprompt.prompts import ManualVerbalizer, ManualTemplate

verbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)

from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate
template = LMBFFTemplateGenerationTemplate(tokenizer=template_generate_tokenizer, verbalizer=verbalizer, text='{"placeholder":"text_a"} {"mask"} {"meta":"labelword"} {"mask"}.')
# template = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} It is {"mask"}.')

# view wrapped example
wrapped_example = template.wrap_one_example(dataset['train'][0]) 
print(wrapped_example)

# %%
# parameter setting
cuda = True
auto_t = True # whether to perform automatic template generation
auto_v = True # whether to perform automatic label word generation


# %%
# train util function
from openprompt.plms import load_plm
# from openprompt.prompts.prompt_generator import T5TemplateGenerator
from custom_prompt_generator import T5TemplateGenerator
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.trainer import ClassificationRunner
import copy
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup


def fit(model, train_dataloader, val_dataloader, loss_func, optimizer):
    best_score = 0.0
    for epoch in range(10):
        train_epoch(model, train_dataloader, loss_func, optimizer)
        score = evaluate(model, val_dataloader)
        if score > best_score:
            best_score = score
    return best_score
        

def train_epoch(model, train_dataloader, loss_func, optimizer):
    model.train()
    for step, inputs in enumerate(train_dataloader):
        if cuda:
            inputs = inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, val_dataloader):
    model.eval()
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            if cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc


# %% [markdown]
# ### 3. automatic template and verbalizer generation

# %%
from tqdm import tqdm
# template generation
if auto_t:
    print('performing auto_t...')

    if cuda:
        template_generate_model = template_generate_model.cuda()
    template_generator = T5TemplateGenerator(template_generate_model, template_generate_tokenizer, template_tokenizer_wrapper, verbalizer, max_length = 7, beam_width=20) # beam_width is set to 5 here for efficiency, to improve performance, try a larger number.

    # this loads all of the data at once - causes gpu issues. Need to subset for now

    auto_t_dataset = dataset['train'][:1000]
    print(f"auto_t_datset length: {len(auto_t_dataset)}")
    dataloader = PromptDataLoader(auto_t_dataset, template, template_generate_tokenizer, template_tokenizer_wrapper, batch_size=len(dataset['train']), decoder_max_length=128) # register all data at once
    for data in dataloader:
        if cuda:
            print("cude true")
            data = data.cuda()        
        template_generator._register_buffer(data)
    
    print(f"template generator: {template_generator}")
    template_generate_model.eval()
    print('generating...')
    template_texts = template_generator._get_templates()

    print(f"template texts: {template_texts}")

    original_template = template.text
    template_texts = [template_generator.convert_template(template_text, original_template) for template_text in template_texts]
    # template_generator._show_template()
    template_generator.release_memory()
    # generate a number of candidate template text
    print(template_texts)
    # iterate over each candidate and select the best one
    best_metrics = 0.0
    best_template_text = None
    for template_text in tqdm(template_texts):
        template = ManualTemplate(tokenizer, template_text)

        train_dataloader = PromptDataLoader(dataset['train'], template, tokenizer, WrapperClass)
        valid_dataloader = PromptDataLoader(dataset['validation'], template, tokenizer, WrapperClass)

        model = PromptForClassification(copy.deepcopy(plm), template, verbalizer)

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        if cuda:
            model = model.cuda()
        score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer)

        if score > best_metrics:
            print('best score:', score)
            print('template:', template_text)
            best_metrics = score
            best_template_text = template_text
    # use the best template
    template = ManualTemplate(tokenizer, text=best_template_text)
    print(best_template_text)

# %%
# verbalizer generation
from openprompt.prompts.prompt_generator import RobertaVerbalizerGenerator
if auto_v:
    print('performing auto_v...')
    # load generation model for template generation
    if cuda:
        plm = plm.cuda()
    verbalizer_generator = RobertaVerbalizerGenerator(model=plm, tokenizer=tokenizer, candidate_num=20, label_word_num_per_class=20)
    # to improve performace , try larger numbers

    dataloader = PromptDataLoader(dataset['train'], template, tokenizer, WrapperClass, batch_size=32)
    for data in dataloader:
        if cuda:
            data = data.cuda()
        verbalizer_generator.register_buffer(data)
        print(verbalizer_generator)
    label_words_list = verbalizer_generator.generate()
    verbalizer_generator.release_memory()

    # iterate over each candidate and select the best one
    current_verbalizer = copy.deepcopy(verbalizer)
    best_metrics = 0.0
    best_label_words = None
    for label_words in tqdm(label_words_list):
        current_verbalizer.label_words = label_words
        train_dataloader = PromptDataLoader(dataset['train'], template, tokenizer, WrapperClass)
        valid_dataloader = PromptDataLoader(dataset['validation'], template, tokenizer, WrapperClass)

        model = PromptForClassification(copy.deepcopy(plm), template, current_verbalizer)

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        if cuda:
            model = model.cuda()
        score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer)

        if score > best_metrics:
            best_metrics = score
            best_label_words = label_words
    # use the best verbalizer
    print(best_label_words)
    verbalizer = ManualVerbalizer(tokenizer, num_classes=len(class_labels), label_words=best_label_words)

# %% [markdown]
# ### 4. main training loop

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, 
    batch_size=batchsize_t,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, 
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, 
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

model = PromptForClassification(copy.deepcopy(plm), template, verbalizer)
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
if cuda:
    model = model.cuda()
score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer)
test_score = evaluate(model, test_dataloader)
print(test_score)



