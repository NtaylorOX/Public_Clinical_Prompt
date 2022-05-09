from openprompt.data_utils import InputExample
import torch
import pandas as pd
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

import pandas as pd
import numpy as np
from tqdm import tqdm

from torchnlp.encoders import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from transformers import  AdamW, get_linear_schedule_with_warmup



import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


# tweaked version of tensorboard summary writer to avoid duplication of hparams
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


#TODO - may need to refactor how labe encoder is instantiated. At moment it does it separatley for each set

class Mimic_ICD9_Processor(DataProcessor):


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
        g = df.groupby('hospital_expire_flag')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state = random_state).reset_index(drop=True)))
        g.reset_index(drop=True, inplace = True)

        return g.sample(frac=1, random_state=random_state)

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = False, class_labels_save_dir = "scripts/mimic_icd9_top50/", balance_data = False):

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

        
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            # write each label to a file separated by new line, but do not add new line to last entry as this will create an empty "" label
            for element in class_labels[:-1]:

                textfile.write(element + "\n")
            # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close() 

        return examples

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




class Mimic_ICD9_Triage_Processor(DataProcessor):


    '''
    Function to convert mimic icd9 triage dataset to a open prompt ready dataset. 
    
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
        g = df.groupby('hospital_expire_flag')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state = random_state).reset_index(drop=True)))
        g.reset_index(drop=True, inplace = True)  

        # give dataframe a shuffle to remove the order imposes by the above 
        return g.sample(frac=1, random_state=random_state)     

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = False, class_labels_save_dir = "./scripts/mimic_triage/", balance_data = False):

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

        
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['triage-category']
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
        
            if not os.path.exists(class_labels_save_dir):
                os.makedirs(class_labels_save_dir)
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline           

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            for element in class_labels[:-1]:

                textfile.write(element + "\n")
                # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close() 

        return examples

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_triage/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels


class Mimic_Mortality_Processor(DataProcessor):


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
                     generate_class_labels = False, class_labels_save_dir = "./scripts/mimic_mortality/",
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

        print("label encoder idx to token: ", self.label_encoder.token_to_index)
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label)
            
            text_a = body.replace('\\', ' ')
                
            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
        
            if not os.path.exists(class_labels_save_dir):
                os.makedirs(class_labels_save_dir)
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline           

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            for element in class_labels[:-1]:

                textfile.write(element + "\n")
            # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close()
            
        if class_weights and sampler_weights:
            print("cannot return both class and sample weights. Just returning samples")
            return examples
        if class_weights:
            return examples, task_class_weights
        elif sampler_weights:
            return examples, sampler_class_weights
        else:
            return examples

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

class Mimic_LOS_Processor(DataProcessor):


    '''
    Function to convert mimic length-of-stay (4 classes) prediction dataset from the clinical outcomes paper: https://aclanthology.org/2021.eacl-main.75/
    
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
        ce_class_weights = compute_class_weight("balanced", classes = np.unique(df["los_label"]),
                                             y = df['los_label'] )

        return ce_class_weights
    
    def get_weighted_sampler_class_weights(self, df, normalized = True):
        
        '''
        Function to create array of per sample class weights to pass to the weighted random sampler.
        
        Purpose is to create batches which sample from the entire dataset based on class weights ->
        this attempts to create balanced batches during training.
        
        DO NOT SHUFFLE DATASET WHEN TRAINING - use weightedrandomsampler
        '''
        
        if normalized:
            nSamples = df["los_label"].value_counts()
            class_weights = [1 - (x / sum(nSamples)) for x in nSamples]
            
        # can use the class weights derived from the get_ce_class weights function
        else:
            class_weights = self.get_ce_class_weights(df)
        
        # creata dict for easy mapping
        class_weights_dict = {0:class_weights[0], 1:class_weights[1]}        
        
        # then need to assign these class specific weights to each sample based on their class
        class_weights_array = df["los_label"].map(class_weights_dict)
        
        return class_weights_array
        
        
        
        
    def balance_dataset(self,df, random_state = 42):
    
        '''
        Function to balance the training dataset - won't bother with the valid and test sets


        '''   


        # slightly clunky but works
        g = df.groupby('los_label')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state = random_state).reset_index(drop=True)))
        g.reset_index(drop=True, inplace = True)

        return g.sample(frac=1, random_state=random_state)

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = True, class_labels_save_dir = "./scripts/mimic_los/",
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
        df["label"] = df["los_label"].map({0:"immediate",1:"short",2:"medium",3:"long"})
        
        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df["label"]).tolist(),reserved_labels = [])
            print(np.unique(df["label"]).tolist())
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

        print("label encoder idx to token: ", self.label_encoder.token_to_index)
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label)
            
            text_a = body.replace('\\', ' ')
                
            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
        
            if not os.path.exists(class_labels_save_dir):
                os.makedirs(class_labels_save_dir)
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline           

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            for element in class_labels[:-1]:

                textfile.write(element + "\n")
            # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close()
            
        if class_weights and sampler_weights:
            print("cannot return both class and sample weights. Just returning samples")
            return examples
        if class_weights:
            return examples, task_class_weights
        elif sampler_weights:
            return examples, sampler_class_weights
        else:
            return examples

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_los/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels


class customPromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer. 
    
    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper. 
    """
    def __init__(self, 
                    dataset: Union[Dataset, List],
                    template: Template,
                    tokenizer: PreTrainedTokenizer,
                    tokenizer_wrapper_class: TokenizerWrapper,
                    verbalizer: Optional[Verbalizer] = None,
                    max_seq_length: Optional[str] = 512,
                    batch_size: Optional[int] = 1,
                    shuffle: Optional[bool] = False,
                    teacher_forcing: Optional[bool] = False,
                    decoder_max_length: Optional[int] = -1,
                    predict_eos_token: Optional[bool] = False,
                    truncate_method: Optional[str] = "tail",
                    drop_last: Optional[bool] = False,
                    sampler: Optional[str] = None,
                    **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset
        
        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
            "max_seq_length" : max_seq_length,
            "truncate_method" : truncate_method,
            "decoder_max_length" : decoder_max_length,
            "predict_eos_token" : predict_eos_token,
            "tokenizer" : tokenizer,
            **kwargs,
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
        

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        
        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                            named wrap_one_example"
        
        # processs
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        # else:
        #     sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset, 
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List): 
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer, 
            the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)
        
    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()



def get_n_trainable_params_prompt_model(model, verbose = True):    

    '''
    Function to count the number of trainable parameters of a prompt model. 
    '''
    
    # all trainable
    num_total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # split into the plm and classisifcation head
    num_plm_trainable = sum(p.numel() for p in model.prompt_model.plm.parameters() if p.requires_grad)
    
    # template trainable - if manual will not have any parameters at all
    try:
        num_template_trainable = sum(p.numel() for p in model.template.soft_embedding.parameters() if p.requires_grad)
    except:
        num_template_trainable = 0
    
    # verbalizer trainable 
    num_verbalizer_trainable = sum(p.numel() for p in model.verbalizer.parameters() if p.requires_grad)
    
    # assert sum of the two = total
    assert num_plm_trainable+num_template_trainable+num_verbalizer_trainable == num_total_trainable
    
    print(f"Number of trainable parameters of PLM: {num_plm_trainable}\n")
    print('#'*50)
    print(f"Number of trainable parameters of template: {num_template_trainable}\n")
    print('#'*50)
    print(f"Number of trainable parameters of verbalizer: {num_verbalizer_trainable}\n")
    print('#'*50)
    print(f"Total number of trainable parameters of whole model: {num_total_trainable}")

    return num_total_trainable