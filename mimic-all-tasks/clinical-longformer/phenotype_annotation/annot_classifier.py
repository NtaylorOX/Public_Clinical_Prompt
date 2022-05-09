# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler,Dataset
from transformers import AutoModel

import pytorch_lightning as pl
from tokenizer import Tokenizer
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from utils import mask_fill

from loguru import logger




Run in Google Colab

Download Notebook

View on GitHub
WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS
Author: Sasank Chilamkurthy

A lot of effort in solving any machine learning problem goes in to preparing the data. PyTorch provides many tools to make data loading easy and hopefully, to make your code more readable. In this tutorial, we will see how to load and preprocess/augment data from a non trivial dataset.

To run this tutorial, please make sure the following packages are installed:

scikit-image: For image io and transforms
pandas: For easier csv parsing
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class MimicAnnotDataset(Dataset):
    """MIMIC Phenotype annotations dataset."""

    def __init__(self, dataframe):
        """
        Args:
            DataFrame (string): The preprocessed df with annotations.
                on a sample.
        """

        self.df = dataframe
        
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        text = self.df.iloc[idx]['text']
        one_hot = torch.from_numpy(np.array(list(self.df.iloc[idx,1:]))).to(torch.long)
        sample = {'text': text, 'label': one_hot}

        return sample


class Classifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    # *************************
    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()
            self.hparams = classifier_instance.hparams
            if self.hparams.transformer_type == 'longformer':
                self.hparams.batch_size = 1
            self.classifier = classifier_instance

            self.transformer_type = self.hparams.transformer_type
            self.raw_data = self.get_mimic_data()

            msk = np.random.rand(len(self.raw_data)) < 0.8
            self.train = self.raw_data[msk]
            self.test = self.raw_data[~msk]

            # self.label_encoder.unknown_index = None

        def get_mimic_data(self, path: str) -> list:
            """ Reads a comma separated value file.

            :param path: path to a csv file.
            
            :return: List of records as dictionaries
            """
            df = pd.read_csv(path)
            df = df.drop('ROW_ID',axis=1)
            return df
            # df["text"] = df["text"].astype(str)
            # df["label"] = df["label"].astype(str)
            # return df.to_dict("records")

        def train_dataloader(self) -> DataLoader:
            """ Function that loads the train set. """
            self._train_dataset = MimicAnnotDataset(self.train)
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        # def val_dataloader(self) -> DataLoader:
        #     """ Function that loads the validation set. """
        #     self._dev_dataset = self.get_mimic_data(self.hparams.dev_csv)
        #     return DataLoader(
        #         dataset=self._dev_dataset,
        #         batch_size=self.hparams.batch_size,
        #         collate_fn=self.classifier.prepare_sample,
        #         num_workers=self.hparams.loader_workers,
        #     )

        def test_dataloader(self) -> DataLoader:
            """ Function that loads the validation set. """
            self._test_dataset = self.MimicAnnotDataset(self.test)

            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

#    ****************

    def __init__(self, hparams: Namespace) -> None:
        super(Classifier,self).__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # Build Data module
        self.data = self.DataModule(self)
        
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs


    def __build_model(self) -> None:
        """ Init transformer model + tokenizer + classification head."""

        #simonlevine/biomed_roberta_base-4096-speedfix'
        
        self.transformer = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
            # gradient_checkpointing=True, #critical for training speed.
        )

        if self.hparams.transformer_type == 'longformer':
            logger.warning('Turnin ON gradient checkpointing...')

            self.transformer = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
            gradient_checkpointing=True, #critical for training speed.
                )

        else: self.transformer = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
                )
            
           #others to try:
            # bert-base-uncased
            #'emilyalsentzer/Bio_ClinicalBERT'
            # allenai/biomed_roberta_base
            # simonlevine/biomed_roberta_base-4096-speedfix'
        
        # set the number of features our encoder model will return...
        self.encoder_features = 768

        # Tokenizer
        if self.hparams.transformer_type  == 'longformer':
            self.tokenizer = Tokenizer(
                pretrained_model=self.hparams.encoder_model,
                max_tokens = self.hparams.max_tokens_longformer)

        else: self.hparams.tokenizer = Tokenizer(
            pretrained_model=self.hparams.encoder_model,
            max_tokens = self.hparams.max_tokens)

           #others:
             #'emilyalsentzer/Bio_ClinicalBERT' 'simonlevine/biomed_roberta_base-4096-speedfix'

 
    def __build_loss(self):
        """ Initializes the loss function/s. """
        #FOR SINGLE LABELS --> MSE (linear regression) LOSS (like a regression problem)
        # For multiple POSSIBLE discrete single labels, CELoss
        # for many possible categoricla labels, binary cross-entropy (logistic regression for all labels.)
        self._loss = nn.CrossEntropyLoss()

        # self._loss = nn.MSELoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.transformer.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def forward(self, tokens, lengths):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run transformer model.
        word_embeddings = self.transformer(tokens, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        return {"logits": self.classification_head(sentemb)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        self.log('loss',loss_val)

        # can also return just a scalar instead of a dict (return loss_val)
        return loss_val


    
    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            
        self.log('test_loss',loss_val)

        # can also return just a scalar instead of a dict (return loss_val)
        return loss_val

    

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.transformer.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
    
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default='simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased',
            type=str,
            help="Encoder model to be used.",
        )

        parser.add_argument(
            "--transformer_type",
            default='longformer',
            type=str,
            help="Encoder model /tokenizer to be used (has consequences for tokenization and encoding; default = longformer).",
        )

        parser.add_argument(
            "--single_label_encoding",
            default='default',
            type=str,
            help="How should labels be encoded? Default for torch-nlp label-encoder...",
        )
        

        parser.add_argument(
            "--max_tokens_longformer",
            default=4096,
            type=int,
            help="Max tokens to be considered per instance..",
        )
        parser.add_argument(
            "--max_tokens",
            default=512,
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
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/intermediary-data/notes2diagnosis-icd-train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/intermediary-data/notes2diagnosis-icd-validate.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/intermediary-data/notes2diagnosis-icd-test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
