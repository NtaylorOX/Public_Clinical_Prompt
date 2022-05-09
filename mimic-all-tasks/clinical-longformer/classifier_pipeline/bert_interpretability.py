import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from torchnlp.encoders import Encoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text.text_encoder import TextEncoder
# from tokenizer import Tokenizer
from classifier_one_label import Classifier

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torchnlp.random import set_seed


def main(hparams, model_dir) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------

    checkpoint = torch.load(model_dir)
    print(f"Checkpoint hyper parameters: {checkpoint['hyper_parameters']}")
    print(f"Checkpoint state_dict: {checkpoint.keys()}")
    model = Classifier(hparams)

    #below isn't working for some reason - it ends up having hparams issues
    # model.load_from_checkpoint(model_dir)   
    model.load_state_dict(checkpoint['state_dict'])
    print(f"model is: {model}")


    
    

if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )

    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=12, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args - 
    parser.add_argument("--gpus", type=str, default=1, help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

    # use ddp 
    parser.add_argument("--accelerator",default = None, type=str, help ="whether or not to use data paralell and switch accelerator for trainer class. Use dp for multiple gpus 1 machine")
    

    # each LightningModule defines arguments relevant to it
    parser = Classifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams,
        model_dir = "/home/niallt/NLP_DPhil/NLP_Mimic_only/clinical-longformer/experiments/emilyalsentzer/Bio_ClinicalBERT/version_20-09-2021--11-08-38/checkpoints/epoch=3-step=2871.ckpt"
        )