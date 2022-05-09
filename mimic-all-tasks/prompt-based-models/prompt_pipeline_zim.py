# This is the Pipeline functions of Prompt methods in zero-shot, few-shot and full-data cases
import os
import time
from openprompt.plms import load_plm
import pandas as pd
import numpy as np
import random
import torch
import utils
from tqdm import tqdm
import utils
from utils import MimicProcessor
from openprompt import PromptForClassification
from openprompt.prompts import ManualVerbalizer,SoftVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts.ptr_prompts import PTRTemplate, PTRVerbalizer
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate, PtuningTemplate
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from openprompt.data_utils.data_sampler import FewShotSampler
this_run_unicode = str(random.randint(0, 1e10))

# Here, we list all the parameters which defines the experiment
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
'Parameters Group 1: Experiment_details'

# Please define experiment type here: zero_shot, few_shot or full_data
experi_type = 'few_shot'
set_seed(144)
num_train_epochs = 3
use_cuda = True
model_parallelize = False
tune_plm = False # whether parameter efficient
plm_eval_mode = True # whether to turn off the dropout in the freezed model. Set to true to turn off

if experi_type == 'zero_shot':
    print('We are in zero-shot case')
    tune_plm = False 
    use_cuda = False 
    plm_eval_mode = True # whether to turn off the dropout in the freezed model. Set to true to turn off

elif experi_type == 'few_shot':
    num_example = 256 # This is number of examples per label, total num of examples will be 2*num_example
    sampler_seed = 100 # determine the random sample of data (11)
    sample_criterion = 'patient_ID' # choose from 'clinical_notes' and 'patient_ID'
    print('We are in few-shot case with num_example per label:', num_example)

elif experi_type == 'full_data':
    print('We are in full-data case')

else:
    print('Please define a correct experi_type: choose one from [full_data, zero_shot, few_shot]')
    raise NotImplementedError

location = 'prompt_pipeline' # where we save results

################################################################################################################################################################
'Parameters Group 2: Model'

model_name = 'bert'
model_path = "/home/s2174572/mlp/mlp1/model/pretraining/pretraining/clinical-pubmed-bert-base-512/for_prompt"
plm,tokenizer, model_config, WrapperClass = load_plm(model_name,model_path)
# Choice of template: 'manual', 'soft', 'mixed', 'ptr', 'ptuning'
model_template = 'manual' # Here we define which template we use
# Choice of verbalizer: 'manual', 'soft', 'ptr', 'knowledgeable'
model_verbalizer = 'knowledgeable' # Here we define which verbalizer we use 
num_tokens = 20 # note that if template is chosen to be soft, you MUST define this, a starting choice could be 20
if  model_template == 'soft' and num_tokens == None:
    print('This is a WRONG practice, please define num_tokens!')
    raise NotImplementedError

if model_template == 'manual':
    mytemplate = ManualTemplate(
        # text = '{"placeholder": "text_a"} the doctor predict the patient {"mask"} readmitted. bad health condition means yes readmission, recovery means no readmission',
        text = '{"placeholder": "text_a"} bad health condition means yes readmission, recovery means no readmission, predict readmission: {"mask"}',
        # text = '{"placeholder": "text_a"} predict readmission: {"mask"} bad health condition means yes readmission, recovery means no readmission',
        tokenizer = tokenizer,
    )

elif model_template == 'soft':
    initialization_path = '/home/s2174572/mlp/mlp1/soft_template.txt'
    mytemplate = SoftTemplate(
        model=plm, 
        tokenizer=tokenizer, 
        num_tokens=num_tokens).from_file(initialization_path)

elif model_template == 'mixed':
    mytemplate = MixedTemplate(
        model=plm, 
        tokenizer=tokenizer, 
        # text = 'clinical note is {"soft"} {"placeholder": "text_a"}, which suggest {"soft"} the patient {"mask"} readmitted',
        text = '{"soft"} bad health condition means yes readmission, {"soft"} recovery means no readmission, predict readmission base on {"soft"} {"placeholder": "text_a"} {"mask"} ',
        #text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} readmit {"mask"}.'# note that we can initialize soft tokens with manual hard tokens
    )

elif model_template == 'ptuning':
    mytemplate = PtuningTemplate(
        model = plm,
        tokenizer = tokenizer,
        text = '{"placeholder": "text_a"} {"soft"} predict readmission: {"mask"} bad health condition means yes readmission, recovery means no readmission',
        prompt_encoder_type = 'lstm' # 'mlp' or 'lstm'
    ) # TODO: is it legal to add {"soft"} into text?


elif model_template == 'ptr': # TODO: Check correctness of ptr method
    # please also use ptr_verbalizer together with ptr_template 
    mytemplate = PTRTemplate(
    model = plm,
    text = '{"placeholder": "text_a"} {"soft"} {"soft"} {"mask"} body health, recovery {"mask"}, readmission: {"mask"} readmitted',
    tokenizer = tokenizer,
    ) # TODO: is it legal to add {"soft"} into text?


if model_verbalizer == 'manual':
    classes = [
    "negative",
    "positive"]
    myverbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "negative": ['no'],
            "positive": ['yes'],},
        tokenizer = tokenizer,
    )

if model_verbalizer == 'soft':
    myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=2)

if model_verbalizer == 'ptr':
    classes = [
    "negative",
    "positive"]

    label_words = [
        ['good','yes','no'], # No readmission
        ['bad','no','yes']   # readmission
    ]
    myverbalizer = PTRVerbalizer(
        tokenizer = tokenizer, 
        classes = classes, 
        num_classes = 2, 
        label_words = label_words)


if model_verbalizer == 'knowledgeable':
    # NOTE for few-shot case, knowledgeverblizer will be redefined below
    remove_current_label_words_address = True # whether remove current_label_words.txt file
    verbalizer_lr = 5e-2 # default is 5e-2 
    classes = [
    "negative",
    "positive"]
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes = classes, num_classes = 2, verbalizer_lr = verbalizer_lr).from_file('/home/s2174572/mlp/mlp1/verbalizer_calibration.txt')


prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not tune_plm), plm_eval_mode=plm_eval_mode)
# Previously, I did some test here.
'''
# print(prompt_model) # Test
print(prompt_model)


print('word embedding in bert:')
for a in prompt_model.plm.bert.embeddings.word_embeddings.parameters():
    print(a)
    print(a.shape)
    print(a.requires_grad)
    


print('mytemplate parameters:')
for param in mytemplate.parameters():
    print('Get In')
    print(param)
    print(param.shape)
    print(param.requires_grad)

# print('for ptuning template, print out its head:')
# mytemplate.generate_parameters()
# print(mytemplate.new_mlp_head)


# I have met a bug using trainable ptr, and this fix the bug
if model_template == 'ptr': # Test
    for param in mytemplate.parameters():
        param.requires_grad = True
        print(param)
        print(param.shape)
        print(param.requires_grad)
'''
################################################################################################################################################################
'Parameters Group 3: Data and DataLoader'

mimic_data_dir = "/home/s2174572/mlp/mlp1/data/discharge" # datermine which dataset to use
# Note prompt_finetune dataset will use different set name to acquire dataset['train'] !
full_test_set = True # full validation data
full_val_set = True # full test data
if num_tokens == None:
    max_seq_l = 512
else:
    max_seq_l = 512 - num_tokens

batchsize_t = 4 # train_batchsize in PromptDataLoader
batchsize_e = 1 # val and test batchsize in PromptDataLoader

################################################################################################################################################################
'Parameters Group 4: Optimization'

'plm'
lr_plm = 3e-5 # lr for optimizer of plm
warmup_step_plm = 20

'template'
prompt_lr_template = 0.5 # 
warmup_step_template = 20 # 500

'verbalizer'
prompt_lr_verbalizer1 = 3e-5
prompt_lr_verbalizer2 = 3e-4
warmup_step_verbalizer = 20

eval_every_steps = 1 # default value 500, store best model every 500 steps
gradient_accumulation_steps = 4 # 4


################################################################################################################################################################
'Parameters Group 5: auxilary and preparation'

pbar_update_freq = 2 # 10

if use_cuda:
    prompt_model=  prompt_model.cuda()

if model_parallelize:
    prompt_model.parallelize()

# tot_step = ... total number of steps to run
# since tot_step only matters for few/full data cases, so we defined it below under if condition
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# End of parameter defining process
if experi_type == 'zero_shot':
    
    test_dataset = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "test")
    train_dataset = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "train")
    # zero-shot test
    test_dataloader = PromptDataLoader(dataset=test_dataset, template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, 
        batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")


    # NOTE: For zero-shot + KnowledgeableVerbalizer case, do calibration before inference
    if model_verbalizer == 'knowledgeable':

        num_knowledgeable_examples = 200 # use in zero-shot case
        support_sampler = FewShotSampler(num_examples_total = num_knowledgeable_examples, also_sample_dev=False)
        support_dataset = support_sampler(train_dataset, seed = 1)
        # remove the labels of support set for clarification
        for example in support_dataset:
            example.label = -1

        support_dataloader = PromptDataLoader(dataset=support_dataset, template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, 
            batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
        
        from openprompt.utils.calibrate import calibrate
        cc_logits = calibrate(prompt_model, support_dataloader)
        prompt_model.verbalizer.register_calibrate_logits(cc_logits)

    truncate_rate_test = test_dataloader.tokenizer_wrapper.truncate_rate
    print("truncate rate: {}".format(truncate_rate_test), flush=True)
    allpreds = []
    alllabels = []
    IDs = []

    m = torch.nn.Softmax(dim=1) # define a Softmax function
    # zero-shot predicting
    with tqdm(total=test_dataloader.__len__()) as pbar:
        pbar.set_description('Precessing')
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            ID = inputs['guid']
            alllabels.extend(labels.cpu().tolist())
            allpreds.append(m(logits)[0][1].cpu().tolist()) # give probability of being positve class
            IDs.extend(ID.cpu().tolist())
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame({'ID':IDs,'Label':alllabels})
    fpr, tpr, df_out = utils.vote_score(df,allpreds,location)
    utils.summarize_predictions(df_out, location)
    # TODO: acc, f1 score, confusion matrix
# -- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- ---- -- -- -- -- -- -- --
else: # Non zero-shot cases

    if experi_type == 'few_shot':

        if sample_criterion == 'clinical_notes': # TODO: full_val_set and full_test_set cases
            # sample data based on clinical_notes
            print('We will randomly sample', num_example, 'examples per label')
            print('Sampling process based on:', sample_criterion)
            dataset = {}
            # read data and get different splits
            if mimic_data_dir == "/home/s2174572/mlp/mlp1/data/discharge_split":
                dataset['train'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "train_0.1")
            else:
                dataset['train'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "train")
            dataset['val'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "val")
            dataset['test'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "test")
            sampler  = FewShotSampler(num_examples_per_label = num_example, num_examples_per_label_dev=num_example, also_sample_dev=True)
            dataset['train'], dataset['val'] = sampler(dataset['train'], dataset['val'],seed = sampler_seed)
            _, dataset['test'] = sampler(dataset['train'], dataset['test'], seed = sampler_seed)

            tot_step = int(2*num_example / batchsize_t / gradient_accumulation_steps * num_train_epochs)

        elif sample_criterion == 'patient_ID':
            # sample data based on clinical_notes
            print('We will randomly sample', num_example, 'examples per label')
            print('Sampling process based on:', sample_criterion)
            mimic_data_modified_dir = utils.Group_and_Sample(mimic_data_dir, num_examples_per_label = num_example, seed = sampler_seed)
            dataset = {}
            # read data and get different splits
            if mimic_data_dir == "/home/s2174572/mlp/mlp1/data/discharge_split": # prompt_finetune dataset
                dataset['train'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "train_0.1")
            else:
                dataset['train'] = MimicProcessor().get_examples(data_dir = mimic_data_modified_dir, set = "train")
            
            if full_val_set == False:
                dataset['val'] = MimicProcessor().get_examples(data_dir = mimic_data_modified_dir, set = "val")
            # NOTE: if you have a look at ./few_shot_data file, it will still be sampled val/test; but what was read in 
            # is changed here: we read from original mimic_data_dir instead of mimic_data_modified_dir
            elif full_val_set == True:
                dataset['val'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "val")

            if full_test_set == False:    
                dataset['test'] = MimicProcessor().get_examples(data_dir = mimic_data_modified_dir, set = "test")
            elif full_test_set == True:
                dataset['test'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "test")

            tot_step = int(2*num_example / batchsize_t / gradient_accumulation_steps * num_train_epochs)
            
            # NOTE: for few-shot + Knowledgeable Verbalizer case, the label_words are defined from each sampled train.csv
            if model_verbalizer == 'knowledgeable':
                train_dir = f'{mimic_data_modified_dir}/train.csv'
                current_label_words_address = '/home/s2174572/mlp/mlp1/current_lable_words.txt'
                list0=['not', 'no', 'never'] # prior word label for negative class
                list1=['yes', 'will', 'possibly'] # prior word label for positive class
                _ = utils.label_words_from_statistics(train_dir,200, current_label_words_address, list0, list1)
                prompt_model.verbalizer.from_file(current_label_words_address) # update label words
                if use_cuda:
                    prompt_model=  prompt_model.cuda()

                if remove_current_label_words_address == True:
                    os.remove(current_label_words_address)


    elif experi_type == 'full_data':
        dataset = {}
        # read data and get different splits
        if mimic_data_dir == "/home/s2174572/mlp/mlp1/data/discharge_split": # prompt_finetune dataset
            dataset['train'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "train_0.1")
        else:
            dataset['train'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "train")
        dataset['val'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "val")
        dataset['test'] = MimicProcessor().get_examples(data_dir = mimic_data_dir, set = "test")
        tot_step = int( len(dataset['train']) / batchsize_t / gradient_accumulation_steps * num_train_epochs)
    
    print('Total Step is', tot_step)

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batchsize_t,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, 
        batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    # zero-shot test
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    truncate_rate_test = test_dataloader.tokenizer_wrapper.truncate_rate
    print("truncate rate: {}".format(truncate_rate_test), flush=True)
    # print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Auxilary functions
    def evaluate(prompt_model, dataloader):
        # Note: this evaluate function does NOT based on patient, based on individual notes instead
        prompt_model.eval()
        allpreds = []
        alllabels = []
        tot_loss = 0
    
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels) # This function is defined outside
            tot_loss += loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        ave_loss = tot_loss / len(dataloader)
        return acc, ave_loss

    m = nn.Softmax(dim=1) # define a Softmax function

    def evaluate2(prompt_model, dataloader):
        # Note: this evaluate function DOES based on patient
        # used for validation_dataloader
        prompt_model.eval()
        allpreds = []
        alllabels = []
        IDs = []
        tot_loss = 0
    
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            ID = inputs['guid']
            loss = loss_func(logits, labels) # This function is defined outside
            tot_loss += loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.append(m(logits)[0][1].cpu().tolist()) # give probability of being positve class
            # allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            IDs.extend(ID.cpu().tolist())

        df = pd.DataFrame({'ID':IDs,'Label':alllabels})
        fpr, tpr, df_out = utils.vote_score_no_plot(df,allpreds,location)
        # Redefine predictions after grouping by patient ID
        allpreds_discrete = (df_out['logits'] > 0.5) # give a 0/1 based prediction
        allpreds = df_out['logits'] # Continuous values
        alllabels = df_out['label']
        # acc
        acc = sum([int(i==j) for i,j in zip(allpreds_discrete, alllabels)])/len(allpreds_discrete)
        # roc score
        roc_auc_score = auc(fpr, tpr)
        # val_loss
        ave_loss = tot_loss / len(dataloader)
        # prc score
        precision, recall, thres = precision_recall_curve(alllabels, allpreds)
        prc_auc_score = auc(recall, precision)
        # rp80 
        pr_thres = pd.DataFrame(data =  list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
        # print('pr_thres is ',pr_thres) # New!
        temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
        rp80 = 0
        if temp.size == 0:
            print('Test Sample too small or RP80=0')
        else:
            # print('temp is ',temp) # New!
            rp80 = temp.iloc[0].recall
            print('Recall at Precision of 80 is {}', rp80)

        return acc, ave_loss, roc_auc_score, prc_auc_score, rp80


    def evaluate_plot(prompt_model, dataloader,location):
        # used for test_dataloader
        prompt_model.eval()
        allpreds = []
        alllabels = []
        IDs = []
    
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            ID = inputs['guid']
            alllabels.extend(labels.cpu().tolist())
            allpreds.append(m(logits)[0][1].cpu().tolist()) # give probability of being positve class
            IDs.extend(ID.cpu().tolist())
        
        df = pd.DataFrame({'ID':IDs,'Label':alllabels})
        fpr, tpr, df_out = utils.vote_score(df,allpreds,location)
        df_out['pred_class'] = df_out['logits'] > 0.5 # give a 0/1 based prediction
        df_out.to_csv(f'/home/s2174572/mlp/mlp1/result_all_prompt/{location}/result.csv')
        rp80 = utils.vote_pr_curve(df,allpreds,location)
        return rp80, df_out

    from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
    loss_func = torch.nn.CrossEntropyLoss()

    # There are three parts of model parameters, namely from plm, template and verbalizer
    # different models defined above lead to different parts trainable or not

    if tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
        no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr_plm) # 3e-5
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1, 
            num_warmup_steps=warmup_step_plm, num_training_steps=tot_step)
    else:
        optimizer1 = None
        scheduler1 = None
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if model_template in ['soft', 'mixed', 'ptr', 'ptuning']:
        optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=prompt_lr_template) # usually lr = 0.5
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=warmup_step_template, num_training_steps=tot_step) # usually num_warmup_steps is 500

    elif model_template == 'manual':
        optimizer2 = None
        scheduler2 = None

    else:
        print('Please add Optimizer and scheduler for this new template')
        raise NotImplementedError

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if model_verbalizer == 'soft':
        optimizer_grouped_parameters3 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":prompt_lr_verbalizer1}, # 3e-5
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":prompt_lr_verbalizer2},] # 3e-4

        optimizer3 = AdamW(optimizer_grouped_parameters3)
        scheduler3 = get_linear_schedule_with_warmup(optimizer3, num_warmup_steps=warmup_step_verbalizer, num_training_steps=tot_step) # usually num_warmup_steps is 500
    
    if model_verbalizer == 'knowledgeable':
        # NOTE: knowledgeable verbalizer will use its own inner optimizer
        optimizer3 = None
        scheduler3 = None


    elif model_verbalizer == 'manual' or 'ptr':
        optimizer3 = None
        scheduler3 = None

    else: 
        print('Please add Optimizer and scheduler for this new verbalizer')
        raise NotImplementedError

    tot_loss = 0 
    log_loss = 0
    best_val_acc = 0
    best_val_roc = 0
    best_val_prc = 0
    glb_step = 0
    actual_step = 0
    test_acc_, test_loss_, roc_auc_test_, prc_auc_test_, rp80_test_ = 0,0,0,0,0
    leave_training = False

    df_for_summary = None
    train_loss_history = []
    val_loss_history = []
    acc_traces = []
    roc_history = []
    prc_history = []
    rp80_history = []
    tot_train_time = 0
    max_steps = tot_step
    prompt_model.train()

    print('Start Training!')

    pbar = tqdm(total=tot_step, desc="Train")
    for epoch in range(1000000):
        print(f"Begin epoch {epoch}")
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            tot_train_time -= time.time()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            '''
            print('logits',logits) # Test
            print('labels',labels) # Test
            print('loss_item',loss.item()) # Test
            '''
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1

            if actual_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                glb_step += 1
                if glb_step % pbar_update_freq == 0:
                    aveloss = (tot_loss - log_loss)/pbar_update_freq
                    train_loss_history.append(aveloss)
                    pbar.update(pbar_update_freq)
                    pbar.set_postfix({'loss': aveloss})
                    log_loss = tot_loss

            
            if optimizer1 is not None:
                optimizer1.step()
                optimizer1.zero_grad()
            if scheduler1 is not None:
                scheduler1.step()
            if optimizer2 is not None:
                optimizer2.step()
                optimizer2.zero_grad()
            if scheduler2 is not None:
                scheduler2.step()
            if optimizer3 is not None:
                optimizer3.step()
                optimizer3.zero_grad()
            if scheduler3 is not None:
                scheduler3.step()
            if model_verbalizer == 'knowledgeable': # NOTE: inner optimizer of knowledge verbalizer
                myverbalizer.optimize() 


            tot_train_time += time.time()

            if actual_step % gradient_accumulation_steps == 0 and glb_step >0 and glb_step % eval_every_steps == 0:
                # choose evaluate or evaluate2: val_acc based on clinical notes directly or by patient id
                val_acc, val_loss, roc_auc_score, prc_auc_score, rp80 = evaluate2(prompt_model, validation_dataloader)
                if val_acc >= best_val_acc and roc_auc_score > 0.9*best_val_roc and prc_auc_score > 0.9*best_val_prc:
                    torch.save(prompt_model.state_dict(),f"/home/s2174572/mlp/mlp1/result_all_prompt/{location}/ckpts/{this_run_unicode}.ckpt")
                    best_val_acc = val_acc
                    best_val_roc = roc_auc_score # not really largest val_roc, means the best model's val_roc
                    best_val_prc = prc_auc_score # not really largest val_prc, means the best model's val_prc
                    _, df_for_summary = evaluate_plot(prompt_model, test_dataloader, location)
                    test_acc, test_loss, roc_auc_test, prc_auc_test, rp80_test = evaluate2(prompt_model, test_dataloader)
                    test_acc_, test_loss_, roc_auc_test_, prc_auc_test_, rp80_test_ = test_acc, test_loss, roc_auc_test, prc_auc_test, rp80_test
                
                acc_traces.append(val_acc)
                val_loss_history.append(val_loss)
                roc_history.append(roc_auc_score)
                prc_history.append(prc_auc_score)
                rp80_history.append(rp80_test)
                print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
                prompt_model.train()

            if glb_step > max_steps:
                leave_training = True
                break
        
        if leave_training:
            break  
        
    # train_loss_history plot
    plt.figure(10)
    plt.plot(np.arange(len(train_loss_history)),train_loss_history)
    plt.xlabel('Steps') # for actual steps: 
    plt.ylabel('Loss')
    plt.title('Train Loss History')
    plt.legend(loc='best')
    plt.show()
    string = os.path.join(location,'train_loss_history'+'.png')
    plt.savefig(os.path.join('/home/s2174572/mlp/mlp1/result_all_prompt', string))

    # val_loss_history plot
    plt.figure(20)
    plt.plot(np.arange(len(val_loss_history)),val_loss_history)
    plt.xlabel('Steps') # for actual steps: 
    plt.ylabel('Loss')
    plt.title('Val Loss History')
    plt.legend(loc='best')
    plt.show()
    string = os.path.join(location,'val_loss_history'+'.png')
    plt.savefig(os.path.join('/home/s2174572/mlp/mlp1/result_all_prompt', string))

    # acc_traces plot (val_acc)
    plt.figure(30)
    plt.plot(np.arange(len(acc_traces)),acc_traces)
    plt.xlabel('Steps') # for actual steps: 
    plt.ylabel('acc')
    plt.title('Val acc traces')
    plt.legend(loc='best')
    plt.show()
    string = os.path.join(location,'acc_traces'+'.png')
    plt.savefig(os.path.join('/home/s2174572/mlp/mlp1/result_all_prompt', string))

    # validation metrices store in csv
    summary_dir = {'val_acc': acc_traces,
                    'val_loss': val_loss_history,
                    'val_roc':roc_history,
                    'val_prc': prc_history,
                    'val_rp80': rp80_history
                    }
    df_summary = pd.DataFrame(summary_dir)
    df_summary.to_csv(f'/home/s2174572/mlp/mlp1/result_all_prompt/{location}/val_metrices_history.csv')


    # a simple measure for the convergence speed.
    thres99 = 0.99*best_val_acc
    thres98 = 0.98*best_val_acc
    thres100 = best_val_acc
    step100=step98=step99=max_steps
    for val_time, acc in enumerate(acc_traces):
        if acc>=thres98:
            step98 = min(val_time*eval_every_steps, step98)
            if acc>=thres99:
                step99 = min(val_time*eval_every_steps, step99)
                if acc>=thres100:
                    step100 = min(val_time*eval_every_steps, step100)

    # 'Record experiment details and results, write in .txt file'

    content_write = "="*70+"\n"
    content_write += f"ExperimentType:{experi_type}\tNumEpoches:{num_train_epochs}\n"
    content_write += f"Template:{model_template}\tVerbalizer:{model_verbalizer}\tTuneLM:{tune_plm}\n"
    content_write += f"TruncateRateONTest:{truncate_rate_test}\n"
    content_write += f"TotalSteps:{tot_step}\n"
    if experi_type == 'full_data':
        content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}\n"
    elif experi_type == 'few_shot': # We exclude converge speed for few_shot cases
        content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\n"
        content_write += f"NumOfExamples:{num_example}\tSamplerSeed:{sampler_seed}\n"
    content_write += f'BestModelTestAcc:{test_acc_}\tBestModelTestLoss:{test_loss_}\tBestModelROC:{roc_auc_test_}\tBestModelPRC:{prc_auc_test_}\tBestModelTestRP80:{rp80_test_}\n'
    

    # result_path = '/home/s2174572/mlp/mlp1/result_all_prompt/{location}/result.csv' #  path where result has been saved
    true_label, pred_label, pred_mean, pred_var, pred_posi_mean, pred_posi_var, pred_neg_mean, pred_neg_var = utils.summary_from_df(df_for_summary)
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    f1 = f1_score(true_label,pred_label,average = 'binary')
    cm = confusion_matrix(true_label, pred_label)

    content_write += f"PredMean:{pred_mean}\tPredVar:{pred_var}\n"
    content_write += f"PredMeanForPosi_Examples:{pred_posi_mean}\tPredVarForPosi_Examples:{pred_posi_var}\n"
    content_write += f"PredMeanForNeg_Examples:{pred_neg_mean}\tPredVarForNeg_Examples:{pred_neg_var}\n"
    content_write += f"f1_score:{f1}\n"
    content_write += f"ConfusionMatrix:\n{cm}\n"
    content_write += "\n"
    content_write += f"classification_report:\n{classification_report(true_label, pred_label)}\n"
    # Confusion matrix plot
    class_names = ['not r/a', 'r/a']
    save_dir = f'/home/s2174572/mlp/mlp1/result_all_prompt/{location}'
    utils.plot_confusion_matrix(cm, class_names, save_dir = save_dir)
    print(content_write)

    with open(f'/home/s2174572/mlp/mlp1/result_all_prompt/{location}/result_file.txt', "a") as fout:
        fout.write(content_write)
    import os
    remove_final_model = True
    if remove_final_model == True:
        os.remove(f"/home/s2174572/mlp/mlp1/result_all_prompt/{location}/ckpts/{this_run_unicode}.ckpt")




















