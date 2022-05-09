from collections import defaultdict, namedtuple
from typing import *

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from openprompt.utils.logging import logger

from typing import Union

import os

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders import LabelEncoder
import pytorch_lightning as pl
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
from sklearn.utils.class_weight import compute_class_weight
from openprompt.data_utils.data_processor import DataProcessor

from bert_classifier import MimicBertModel, MimicDataset, MimicDataModule
import argparse
from datetime import datetime
import warnings


class FewShotSampler(object):
    '''
    Few-shot learning is an important scenario this is sampler that samples few examples over each class.
    Args:
        num_examples_total(:obj:`int`, optional): Sampling strategy ``I``: Use total number of examples for few-shot sampling.
        num_examples_per_label(:obj:`int`, optional): Sampling strategy ``II``: Use the number of examples for each label for few-shot sampling.
        also_sample_dev(:obj:`bool`, optional): Whether to apply the sampler to the dev data.
        num_examples_total_dev(:obj:`int`, optional): Sampling strategy ``I``: Use total number of examples for few-shot sampling.
        num_examples_per_label_dev(:obj:`int`, optional): Sampling strategy ``II``: Use the number of examples for each label for few-shot sampling.
    
    '''

    def __init__(self,
                 num_examples_total: Optional[int]=None,
                 num_examples_per_label: Optional[int]=None,
                 also_sample_dev: Optional[bool]=False,
                 num_examples_total_dev: Optional[int]=None,
                 num_examples_per_label_dev: Optional[int]=None,
                 label_col = "label"
                 ):
        if num_examples_total is None and num_examples_per_label is None:
            raise ValueError("num_examples_total and num_examples_per_label can't be both None.")
        elif num_examples_total is not None and num_examples_per_label is not None:
            raise ValueError("num_examples_total and num_examples_per_label can't be both set.")
        
        if also_sample_dev:
            if num_examples_total_dev is not None and num_examples_per_label_dev is not None:
                raise ValueError("num_examples_total and num_examples_per_label can't be both set.")
            elif num_examples_total_dev is None and num_examples_per_label_dev is None:
                logger.warning(r"specify neither num_examples_total_dev nor num_examples_per_label_dev,\
                                set to default (equal to train set setting).")
                self.num_examples_total_dev = num_examples_total
                self.num_examples_per_label_dev = num_examples_per_label
            else:
                self.num_examples_total_dev  = num_examples_total_dev
                self.num_examples_per_label_dev = num_examples_per_label_dev

        self.num_examples_total = num_examples_total
        self.num_examples_per_label = num_examples_per_label
        self.also_sample_dev = also_sample_dev
        self.label_col = label_col

    def __call__(self, 
                 dataset: Union[Dataset, List],
                 valid_dataset: Optional[Union[Dataset, List]] = None,
                 seed: Optional[int] = None
                ) -> Union[Dataset, List]:
        '''
        The ``__call__`` function of the few-shot sampler.
        Args:
            dataset (:obj:`Dictionary or dataframe`): The train dataset for the sampler.
            valid_dataset (:obj:`Union[Dataset, List]`, optional): The valid datset for the sampler. Default to None.
            seed (:obj:`int`, optional): The random seed for the sampling.
        
        Returns:
            :obj:`(Union[Dataset, List], Union[Dataset, List])`: The sampled dataset (dataset, valid_dataset), whose type is identical to the input.
        '''
        if valid_dataset is None:
            if self.also_sample_dev:
                return self._sample(dataset, seed, sample_twice=True)
            else:
                dataset = self._sample(dataset, seed, sample_twice=False)
                return pd.DataFrame(dataset)
        else:
            dataset = self._sample(dataset, seed)
            if self.also_sample_dev:
                valid_dataset = self._sample(valid_dataset, seed)
            return pd.DataFrame(dataset)
    
    def _sample(self, 
                data: Union[Dataset, List], 
                seed: Optional[int],
                sample_twice = False,
               ) -> Union[Dataset, List]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        indices = [i for i in range(len(data))]

        if self.num_examples_per_label is not None:  
            assert self.label_col in data[0].keys(), "sample by label requires the data has the label_col attribute."
            labels = [x[self.label_col] for x in data]
            selected_ids = self.sample_per_label(indices, labels, self.num_examples_per_label) # TODO fix: use num_examples_per_label_dev for dev
        else:
            selected_ids = self.sample_total(indices, self.num_examples_total)
        
        if sample_twice:
            selected_set = set(selected_ids)
            remain_ids = [i for i in range(len(data)) if i not in selected_set]
            if self.num_examples_per_label_dev is not None:
                assert self.label_col in data[0].keys(), "sample by label requires the data has a 'label' attribute."
                remain_labels = [x[self.label_col] for idx, x in enumerate(data) if idx not in selected_set]
                selected_ids_dev = self.sample_per_label(remain_ids, remain_labels, self.num_examples_per_label_dev)
            else:
                selected_ids_dev = self.sample_total(remain_ids, self.num_examples_total_dev)
        
            if isinstance(data, Dataset):
                return Subset(data, selected_ids), Subset(data, selected_ids_dev)
            elif isinstance(data, List):
                return [data[i] for i in selected_ids], [data[i] for i in selected_ids_dev]
        
        else:
            if isinstance(data, Dataset):
                return Subset(data, selected_ids)
            elif isinstance(data, List):
                return [data[i] for i in selected_ids]
        
    
    def sample_total(self, indices: List, num_examples_total):
        '''
        Use the total number of examples for few-shot sampling (Strategy ``I``).
        
        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            num_examples_total(:obj:`int`): The total number of examples.
        
        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.
            
        '''
        self.rng.shuffle(indices)
        selected_ids = indices[:num_examples_total]
        logger.info("Selected examples (mixed) {}".format(selected_ids))
        return selected_ids

    def sample_per_label(self, indices: List, labels, num_examples_per_label):
        '''
        Use the number of examples per class for few-shot sampling (Strategy ``II``). 
        If the number of examples is not enough, a warning will pop up.
        
        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            labels(:obj:`List`): The list of the labels.
            num_examples_per_label(:obj:`int`): The total number of examples for each class.
        
        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.
        '''

        ids_per_label = defaultdict(list)
        selected_ids = []
        for idx, label in zip(indices, labels):
            ids_per_label[label].append(idx)
        for label, ids in ids_per_label.items():
            tmp = np.array(ids)
            self.rng.shuffle(tmp)
            if len(tmp) < num_examples_per_label:
                logger.info("Not enough examples of label {} can be sampled".format(label))
            selected_ids.extend(tmp[:num_examples_per_label].tolist())
        selected_ids = np.array(selected_ids)
        self.rng.shuffle(selected_ids)
        selected_ids = selected_ids.tolist()    
        logger.info("Selected examples {}".format(selected_ids))
        return selected_ids


class Mimic_ICD9_Processor():


    '''
    Function to convert mimic icd9 dataset to a open prompt ready dataset. 
    
    We also instantiate a LabelEncoder() class which is fitted to the given dataset. Fortunately it appears
    to create the same mapping for each set, given each set contains all classes. 

    This is not ideal, and need to think of a better way to store the label encoder based on training data.
    

  
    
    '''
    # TODO Test needed
    def __init__(self):
        super().__init__()


    def balance_dataset(self,df, random_state = 42):
    
        '''
        Function to balance the training dataset - won't bother with the valid and test sets


        '''   

        # slightly clunky but works
        g = df.groupby('label')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state = random_state).reset_index(drop=True)))
        g.reset_index(drop=True, inplace = True)

        return g.sample(frac=1, random_state=random_state)

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = True, class_labels_save_dir = "scripts/mimic_icd9_top50/", class_weights = False, balance_data = False):

        path = f"{data_dir}/{mode}.csv"
        print(f"loading {mode} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)
        
        # balance data based on minority class if desired
        if balance_data:
            df = self.balance_dataset(df)

        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df.label).tolist(), reserved_labels = [])
        else: 
            print("we were given a label encoder")
            self.label_encoder = label_encoder

        # new df to fill in examples list
        new_dfs = []
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label).numpy() 
            # add to new df
            new_df = pd.DataFrame({'text':[body],'label':[int(label)]})
            
            new_dfs.append(new_df)
          
        
#         concat all examples
        all_dfs = pd.concat(new_dfs)
        logger.info(f"Returning {len(all_dfs)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
            logger.info(f"generating class labels!")
            class_labels = self.generate_class_labels()

        return all_dfs, class_labels

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_icd9_top50/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels

class Mimic_ICD9_Triage_Processor():


    '''
    Function to convert mimic icd9 dataset to a open prompt ready dataset. 
    
    We also instantiate a LabelEncoder() class which is fitted to the given dataset. Fortunately it appears
    to create the same mapping for each set, given each set contains all classes. 

    This is not ideal, and need to think of a better way to store the label encoder based on training data.
    

  
    
    '''
    # TODO Test needed
    def __init__(self):
        super().__init__()


    def balance_dataset(self,df, random_state = 42):
    
        '''
        Function to balance the training dataset - won't bother with the valid and test sets


        '''   

        # slightly clunky but works
        g = df.groupby('label')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state = random_state).reset_index(drop=True)))
        g.reset_index(drop=True, inplace = True)

        return g.sample(frac=1, random_state=random_state)

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = True, class_labels_save_dir = "scripts/mimic_icd9_top50/", class_weights = False, balance_data = False):

        path = f"{data_dir}/{mode}.csv"
        print(f"loading {mode} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)
        
        # balance data based on minority class if desired
        if balance_data:
            df = self.balance_dataset(df)

        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df["triage-category"]).tolist(), reserved_labels = [])
        else: 
            print("we were given a label encoder")
            self.label_encoder = label_encoder

        # new df to fill in examples list
        new_dfs = []
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['triage-category']
            label = self.label_encoder.encode(label).numpy() 
            # add to new df
            new_df = pd.DataFrame({'text':[body],'label':[int(label)]})
            
            new_dfs.append(new_df)
          
        
#         concat all examples
        all_dfs = pd.concat(new_dfs)
        logger.info(f"Returning {len(all_dfs)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
            logger.info(f"generating class labels!")
            class_labels = self.generate_class_labels()

        return all_dfs, class_labels

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_icd9_top50/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels

class Mimic_Mortality_Processor():


    '''
    Function to convert mimic mortality prediction dataset from the clinical outcomes paper: https://aclanthology.org/2021.eacl-main.75/
    
    to a open prompt ready dataset. 
    
    We also instantiate a LabelEncoder() class which is fitted to the given dataset. Fortunately it appears
    to create the same mapping for each set, given each set contains all classes.    
    
    '''
    # TODO Test needed
    def __init__(self):
        super().__init__()   

    def get_ce_class_weights(self,df):
        
        '''
        Function to calculate class weights to pass to cross entropy loss in pytorch framework.
        
        Here we use the sklearn compute_class_weight function.
        
        Returns: un-normalized class weights inverse to sample size. i.e. lower number given to majority class
        '''

        # calculate class weights 
        ce_class_weights = compute_class_weight("balanced", classes = np.unique(df["hospital_expire_flag"]),
                                             y = df['hospital_expire_flag'] )

        return ce_class_weights
    
    def get_weighted_sampler_class_weights(self, df, normalized = True):
        
        '''
        Function to create array of per sample class weights to pass to the weighted random sampler.
        
        Purpose is to create batches which sample from the entire dataset based on class weights ->
        this attempts to create balanced batches during training.
        
        DO NOT SHUFFLE DATASET WHEN TRAINING - use weightedrandomsampler
        '''
        
        if normalized:
            nSamples = df["hospital_expire_flag"].value_counts()
            class_weights = [1 - (x / sum(nSamples)) for x in nSamples]
            
        # can use the class weights derived from the get_ce_class weights function
        else:
            class_weights = self.get_ce_class_weights(df)
        
        # creata dict for easy mapping
        class_weights_dict = {0:class_weights[0], 1:class_weights[1]}        
        
        # then need to assign these class specific weights to each sample based on their class
        class_weights_array = df["hospital_expire_flag"].map(class_weights_dict)
        
        return class_weights_array
        
        
        
        
    def balance_dataset(self,df, random_state = 42):
    
        '''
        Function to balance the training dataset - won't bother with the valid and test sets


        '''   


        # slightly clunky but works
        g = df.groupby('hospital_expire_flag')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state = random_state).reset_index(drop=True)))
        g.reset_index(drop=True, inplace = True)

        return g.sample(frac=1, random_state=random_state)

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = True, class_labels_save_dir = "./scripts/mimic_mortality/",
                     balance_data = False, class_weights = False, sampler_weights = False):

        path = f"{data_dir}/{mode}.csv"
        print(f"loading {mode} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)
        
        # if balance data - balance based on minority class
        
        if balance_data:
            df = self.balance_dataset(df)

        # map the binary classification label to a new string class label
        df["label"] = df["hospital_expire_flag"].map({0:"alive",1:"deceased"})
        
        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df["label"]).tolist(),reserved_labels = [])
        else: 
            print("we were given a label encoder")
            self.label_encoder = label_encoder
        
        # calculate class_weights
        if class_weights:
            print("getting class weights")
            task_class_weights = self.get_ce_class_weights(df)
        
        # calculate all sample weights for weighted sampler
        if sampler_weights:
            print("getting weights for sampler!")
            sampler_class_weights = self.get_weighted_sampler_class_weights(df)
        
        # new df to fill in examples list
        new_dfs = []
        for idx, row in tqdm(df.iterrows()):
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label).numpy() 
            # add to new df
            new_df = pd.DataFrame({'text':[body],'label':[int(label)]})
            
            new_dfs.append(new_df)
        
#         concat all examples
        all_dfs = pd.concat(new_dfs)
        logger.info(f"Returning {len(all_dfs)} samples!") 
#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
            class_labels = self.generate_class_labels()            
        if class_weights and sampler_weights:
            print("cannot return both class and sample weights. Just returning samples")
            return all
        if class_weights:
            return all_dfs, class_labels, task_class_weights
        elif sampler_weights:
            return all_dfs, class_labels, sampler_class_weights
        else:
            return all_dfs, class_labels

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_mortality/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels
