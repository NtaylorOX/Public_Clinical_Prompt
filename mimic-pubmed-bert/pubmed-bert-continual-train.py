import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import re
import string
import torch
from tqdm import tqdm
from pathlib import Path
import argparse

from spacy.lang.en import English

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import TextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from transformers import pipeline


parser = argparse.ArgumentParser("")
parser.add_argument("--data_path", default="")
parser.add_argument("--heldout", default=False)
parser.add_argument("--model_name", default='emilyalsentzer/Bio_ClinicalBERT')
parser.add_argument("--block_size", type = int, default=128)
parser.add_argument("--output_dir", default = "")
parser.add_argument("--output_dir_model", default = "")

args = parser.parse_args()

# data preprocessing
data_path = args.data_path
df_notes = pd.read_csv(data_path + '/NOTEEVENTS.csv')

if args.heldout == True:
    # held-out dataset if you want to fine-tune on MIMIC data, it is better to exclude them prior to the training
    # exclude all data in test set for re-admission task where we are interested in 
    # ID of patients can be generated via https://github.com/kexinhuang12345/clinicalBERT
    df_test_ids = pd.read_csv(data_path + 'discharge/test.csv').ID.unique()
    df_notes= df_notes[~df_notes.HADM_ID.isin(df_test_ids)]

# choose interested categories, for more information, please refer to 
category_list = ['Discharge summary', 'Echo', 'Nursing', 'Physician ',
       'Rehab Services', 'Respiratory ', 'Nutrition',
       'General', 'Pharmacy', 'Consult', 'Radiology',
       'Nursing/other']
df_notes = df_notes[df_notes.CATEGORY.isin(category_list)]

# data cleaning
def preprocess1(x):
    y=re.sub('\\[(.*?)\\]','',x) 
    y=re.sub('[0-9]+\. ','',y) 
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    #more substituion can be made to align with general knowledge such as "p.o." to "by mouth"
    
    # remove, spaces
    y = y.translate(str.maketrans("", ""))
    y = " ".join(y.split())
    return y

def preprocessing(df_notes): 
    df_notes['TEXT']=df_notes['TEXT'].fillna(' ')
    df_notes['TEXT']=df_notes['TEXT'].str.replace('\n',' ')
    df_notes['TEXT']=df_notes['TEXT'].str.replace('\r',' ')
    df_notes['TEXT']=df_notes['TEXT'].apply(str.strip)
    #We use uncased text which is also used in PubMedBERT
    df_notes['TEXT']=df_notes['TEXT'].str.lower()

    df_notes['TEXT']=df_notes['TEXT'].apply(lambda x: preprocess1(x))
    
    return df_notes

df_notes_processed= preprocessing(df_notes)
# to reuse the processed data in other tasks and save time
df_notes_processed.to_csv('df_notes_processed')

# sentencize notes using spacy
nlp = English()
nlp.add_pipe('sentencizer')

def toSentence(x):
    doc = nlp(x)
    text=[]
    try:
        for sent in doc.sents:
            st=str(sent).strip() 
            if len(st)<30:
                #Merging too-short sentences to appropriate length, this is inherited from ClinicalBERT with changes in merged length 
                if len(text)!=0:
                    text[-1]=' '.join((text[-1],st))
                else:
                    text=[st]
            else:
                text.append((st))
    except:
        print(doc)
    return text

pretrain_sent=df_notes_processed['TEXT'].apply(lambda x: toSentence(x))


file=open(data_path + '/clinical_sentences_pretrain_wo_ECG_30_length_down_sampled.txt','w')
text_file_path = data_path + '/clinical_sentences_pretrain_wo_ECG_30_length_down_sampled.txt'

pretrain_sent = pretrain_sent.values

# random sample 500,000 documents
pretrain_sent = np.random.choice(pretrain_sent,500000)

# write the txt file for building dataset, empty lines between docs (for potential NSP task)
for i in tqdm(range(len(pretrain_sent))):
    if len(pretrain_sent[i]) > 0:
        # remove the one token note
        note = pretrain_sent[i]
        for sent in note:
            file.write(sent+'\n')
        file.write('\n')


# continual training
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForMaskedLM.from_pretrained(args.model_name)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=text_file_path,
    block_size=args.block_size,
    # You can also use 512 block_size to train the model, remember also adjust batch size.
)

# Use Whole Word Masking instead of ordinary masking instead for better performance
data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# we use 5000 steps to warm-up, other optimization parameters are default
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=2_500,
    save_total_limit=3,
    prediction_loss_only=True,
    warmup_steps = 5000
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# training start
trainer.train()
trainer.save_model(args.output_dir_model)