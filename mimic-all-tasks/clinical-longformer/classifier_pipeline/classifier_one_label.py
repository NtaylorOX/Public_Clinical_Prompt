# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import io
# import tensorflow as tf

import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel,RobertaForMaskedLM
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

import pytorch_lightning as pl
from tokenizer import Tokenizer
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from utils import mask_fill

import torchmetrics.functional.classification as metrics
# below pytorch... does not work with later versions, cannot figure out which version it worked with in first place
# import pytorch_lightning.metrics.functional.classification as old_metrics

from sklearn import metrics as skmetrics
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from loguru import logger

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



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
    
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.95
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # figure.savefig(f'experiments/{model}/test_mtx.png')

    return figure



class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)


class Classifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    # *************************
    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()    
            # this needs to update to latest pytorch lightning 
            self.hparams.update(classifier_instance.hparams)

            print("hparams inside datamodule: ",classifier_instance.hparams )

            if self.hparams.transformer_type == 'longformer':
                self.hparams.batch_size = 1
            self.classifier = classifier_instance

            self.transformer_type = self.hparams.transformer_type

            self.dataset = self.hparams.dataset


             #TODO - include logic to select subset of icd9 codes
             # we now have already subsetted training data to read in

            # change logic based on the dataset name

            if self.dataset == "icd9_50":
                logger.info(f"Dataset probided was : icd9_50")

                # set the data_dir based on dataset selected
                self.data_dir = f"{self.hparams.data_dir}/top_50_icd9/"

                self.n_labels = 50
                self.top_codes = pd.read_csv(f"{self.data_dir}{self.hparams.train_csv}")['label'].value_counts()[:self.n_labels].sort_index().index.tolist()
                logger.warning(f'Classifying against the top {self.n_labels} most frequent ICD codes: {self.top_codes}')
                

                # Label Encoder
                if self.hparams.single_label_encoding == 'default':
                    self.label_encoder = LabelEncoder(
                        np.unique(self.top_codes).tolist(), 
                        reserved_labels=[]
                    )

                self.label_encoder.unknown_index = None
            
            elif self.dataset == "icd9_triage":
                logger.info(f"Dataset probided was : icd9_triage")
                # set the data_dir based on dataset selected
                self.data_dir = f"{self.hparams.data_dir}/triage/"

                # get the triage label/classes
                self.triage_labels = pd.read_csv(f"{self.data_dir}{self.hparams.train_csv}")['triage-category'].value_counts().sort_index().index.tolist()
                # define label encoder based on these                  
                self.label_encoder = LabelEncoder(
                    np.unique(self.triage_labels).tolist(), 
                    reserved_labels=[]
                )

                self.label_encoder.unknown_index = None#

            else:

                #TODO implement mimic readmission
                raise NotImplementedError

        def get_mimic_data(self,path: str) -> list: #REWRITE
            """ Reads a comma separated value file.

            :param path: path to a csv file.
            
            :return: List of records as dictionaries
            """
        
            # df = pd.read_csv(path)
            # df = df[["TEXT", "ICD9_CODE"]]
            # df = df.rename(columns={'TEXT':'text', 'ICD9_CODE':'label'})           

            # df = df[df['label'].isin(self.top_codes)]
            # df["text"] = df["text"].astype(str)
            # df["label"] = df["label"].astype(str)

            # df.to_csv(f'{path}_top_codes_filtered.csv')

            
            # for icd9_50 dataset
            if self.dataset == "icd9_50":

                df = pd.read_csv(path)
                logger.warning(f'{path} dataframe has {len(df)} examples.' )
                return df.to_dict("records")

            elif self.dataset == "icd9_triage":
                df = pd.read_csv(path)
                df = df[["text", "triage-category"]]
                df = df.rename(columns={'triage-category':'label'})        
                df["text"] = df["text"].astype(str)
                df["label"] = df["label"].astype(str)
                return df.to_dict("records")


            else:

                #TODO implement mimic readmission
                raise NotImplementedError

        def train_dataloader(self) -> DataLoader:
            """ Function that loads the train set. """
            logger.warning('Loading training data...')
            self._train_dataset = self.get_mimic_data(f"{self.data_dir}{self.hparams.train_csv}")
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            logger.warning('Loading validation data...')

            """ Function that loads the validation set. """
            self._dev_dataset = self.get_mimic_data(f"{self.data_dir}{self.hparams.dev_csv}")
            return DataLoader(
                dataset=self._dev_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def test_dataloader(self) -> DataLoader:
            logger.warning('Loading testing data...')

            """ Function that loads the validation set. """
            self._test_dataset = self.get_mimic_data(f"{self.data_dir}{self.hparams.test_csv}")

            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

#    ****************

    def __init__(self, hparams: Namespace) -> None:
        super(Classifier,self).__init__()
        self.save_hyperparameters(hparams)        
        self.batch_size = hparams.batch_size
        

        # Build Data module
        self.data = self.DataModule(self)

        # get original class labels
        self.class_labels = list(self.data.label_encoder.tokens.keys())
        
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            logger.warning("Freezing the PLM i.e. the encoder - will just be tuning the classification head!")
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

        self.test_conf_matrices=[]

    def __build_model(self) -> None:
        """ Init transformer model + tokenizer + classification head."""

        if self.hparams.transformer_type == 'roberta-long':
            print("loaded roberta long model!")
            self.transformer= RobertaLongForMaskedLM.from_pretrained(
                self.hparams.encoder_model,
                output_hidden_states=True,
                gradient_checkpointing=True
            )

        elif self.hparams.transformer_type == 'longformer':
            self.transformer = AutoModel.from_pretrained(
                self.hparams.encoder_model,
                output_hidden_states=True,
                gradient_checkpointing=True, #critical for training speed.
            )

        else: #BERT
            self.transformer = AutoModel.from_pretrained(
                self.hparams.encoder_model,
                output_hidden_states=True,
                )

        logger.warning(f'model is {self.hparams.encoder_model}')

        if self.hparams.transformer_type == 'longformer':
            logger.warning('Turnin ON gradient checkpointing...')

            self.transformer = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
            gradient_checkpointing=True, #critical for training speed.
                )

        else: self.transformer = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
                )
            

        
        # set the number of features our encoder model will return...
        self.encoder_features = 768

        # Tokenizer
        if self.hparams.transformer_type  == 'longformer' or self.hparams.transformer_type == 'roberta-long':
            self.tokenizer = Tokenizer(
                pretrained_model=self.hparams.encoder_model,
                max_tokens = self.hparams.max_tokens_longformer)
            self.tokenizer.max_len = 4096
 

        else: self.tokenizer = Tokenizer(
            pretrained_model=self.hparams.encoder_model,
            max_tokens = 512)

           #others:
             #'emilyalsentzer/Bio_ClinicalBERT' 'simonlevine/biomed_roberta_base-4096-speedfix'

        # Classification head
        if self.hparams.single_label_encoding == 'default':
            self.classification_head = nn.Sequential(

                nn.Linear(self.encoder_features, self.encoder_features * 2),
                nn.Tanh(),
                nn.Linear(self.encoder_features * 2, self.encoder_features),
                nn.Tanh(),
                nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),

            )

        elif self.hparams.single_label_encoding == 'graphical':
            logger.critical('Graphical embedding not yet implemented!')
            # self.classification_head = nn.Sequential(
                #TODO
            # )


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

        # Run BERT model.       
        
        word_embeddings = self.transformer(tokens, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        # return both the logits and the sentence embeddings - we want to play with these later

        return {"logits": self.classification_head(sentemb), "sent_embs": sentemb}

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
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
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
        if self.trainer.data_parallel:
            loss_val = loss_val.unsqueeze(0)


        self.log('train/batch_loss',loss_val)

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
        if self.trainer.data_parallel:
            loss_val = loss_val.unsqueeze(0)
            
        self.log('test/loss',loss_val)


        y_hat=model_out['logits']
        labels_hat = torch.argmax(y_hat, dim=1)
        y=targets['labels']


        # didn't work wiith version of pytorch_lightning - 
        # f1 = metrics.f1_score(labels_hat,y,    class_reduction='weighted')
        # prec =metrics.precision(labels_hat,y,  class_reduction='weighted')
        # recall = metrics.recall(labels_hat,y,  class_reduction='weighted')
        # acc = metrics.accuracy(labels_hat,y,   class_reduction='weighted')

        f1 = metrics.f1(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))
        prec =metrics.precision(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))
        recall = metrics.recall(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))
        acc = metrics.accuracy(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))

        self.log('test/prec',prec)
        self.log('test/f1',f1)
        self.log('test/recall',recall)
        self.log('test/balanced_accuracy', acc)

        # # get class labels
        # class_labels = self.class_labels

        # # get confusion matrix
        # cm = confusion_matrix(y.cpu().tolist(),labels_hat.cpu().tolist(), labels = list(self.data.label_encoder.token_to_index.values()))

        # # make plot 
        # cm_figure = plot_confusion_matrix(cm, class_labels)

        # # log to tensorboard
        # self.logger.experiment.add_figure("test/confusion_matrix", cm_figure, self.current_epoch)

        # self.test_conf_matrices.append(cm)


    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        if self.hparams.fast_dev_run:
            
            print(f"f y_hats:: {y_hat[0:50]}")
            print(f"f_hats shape is : {y_hat.shape}")
            print(f"labels predicted are: { torch.argmax(y_hat, dim=1)}")
            print(f"y targets = {y}")

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.data_parallel:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # print(f"one example of the logits: {y_hat[0:50]}")
        # print(f"20 of the predicted labels:{labels_hat[0:20]}")     
        # print(f"20 of the labels:{y}")        
        
        self.log('val_loss',loss_val,prog_bar=True)

        # f1 = metrics.f1(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))
        # prec =metrics.precision(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))
        # recall = metrics.recall(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))
        # acc = metrics.accuracy(labels_hat,y, average = 'weighted', num_classes = len(self.class_labels))

        # # print(f"f1 from torch metrics is: ", f1)
        # # print(f"f1 from pytorch metrics is : { old_metrics.f1_score(labels_hat,y,    class_reduction='weighted')}")
        # # print(f"F1 from sk learn is : {skmetrics.f1_score(y,labels_hat, average='weighted')}")
        # # logger.info("val_prec at moment is: ",prec)
        # # logger.info("val_f1 at moment is: ",f1)

        # self.log('valid/prec',prec, on_step = False, on_epoch = True)
        # self.log('valid/f1',f1, on_step = False, on_epoch = True)
        # self.log('valid/recall',recall, on_step = False, on_epoch = True)
        # self.log('valid/balanced_accuracy', acc, on_step = False, on_epoch = Trueprog_bar=True)

        return {"loss": loss_val, "predictions": model_out["logits"], "labels": y}

    def validation_epoch_end(self, outputs):
        
        labels = []
        predictions = []
        for output in outputs:
            
            for out_labels in output["labels"].to('cpu').detach().numpy():                                
                labels.append(out_labels)
            for out_predictions in output["predictions"].to('cpu').detach().numpy():                               
                predictions.append(np.argmax(out_predictions, axis = -1))

        # below is torch metrics but needs to still be tensors
        # f1 = metrics.f1(allpreds,alllabels, average = 'weighted', num_classes = len(class_labels))
        # prec =metrics.precision(allpreds,alllabels, average = 'weighted', num_classes =len(class_labels))
        # recall = metrics.recall(allpreds,alllabels, average = 'weighted', num_classes = len(class_labels))
        # acc = metrics.accuracy(allpreds,alllabels, average = 'weighted', num_classes = len(class_labels))
        
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average = 'weighted')
        prec = precision_score(labels, predictions, average = 'weighted')
        recall = recall_score(labels, predictions, average = 'weighted')    

        # get class labels
        class_labels = self.class_labels

        # get confusion matrix
        cm = confusion_matrix(labels,predictions, labels = list(self.data.label_encoder.token_to_index.values()))

        # make plot 
        cm_figure = plot_confusion_matrix(cm, class_labels)
        
        # log this for monitoring
        self.log('monitor_balanced_accuracy', acc)

        # log to tensorboard
        self.logger.experiment.add_figure("valid/confusion_matrix", cm_figure, self.current_epoch)
        self.logger.experiment.add_scalar('valid/balanced_accuracy',acc, self.current_epoch)
        self.logger.experiment.add_scalar('valid/prec',prec, self.current_epoch)
        self.logger.experiment.add_scalar('valid/f1',f1, self.current_epoch)
        self.logger.experiment.add_scalar('valid/recall',recall, self.current_epoch)




    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.transformer.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]

        #TODO add adafactor as an option here
        if self.hparams.optimizer == "adamw":
            optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.n_warmup_steps,
                num_training_steps=self.hparams.max_steps
            )
        elif self.hparams.optimizer == "adafactor":
            optimizer = Adafactor(parameters,  
                                lr=self.hparams.learning_rate,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.n_warmup_steps)
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
        logger.warning(f"On epoch {self.current_epoch}. Number of frozen epochs is: {self.nr_frozen_epochs}")
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            logger.warning("unfreezing PLM(encoder)")
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
            default= 'emilyalsentzer/Bio_ClinicalBERT',# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased',
            type=str,
            help="Encoder model to be used.",
        )

        parser.add_argument(
            "--transformer_type",
            default='bert', #'longformer', roberta-long
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
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )

        parser.add_argument(
            "--data_dir",
            default="../../data/intermediary-data/",
            type=str,
            help="name of dataset",
        )

        parser.add_argument(
            "--dataset",
            default="icd9_50", #or: icd9_triage
            type=str,
            help="name of dataset",
        )

        parser.add_argument(
            "--train_csv",
            default="train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="valid.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="test.csv",
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
