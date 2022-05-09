"""
Runs a model on a single node across N-gpus.
"""
import argparse
from gc import callbacks
import os
from datetime import datetime
from pathlib import Path

from classifier_one_label import Classifier
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torchnlp.random import set_seed
from loguru import logger



'''
The trivial solution to Pr = Re = F1 is TP = 0. So we know precision, recall and F1 can have the same value in general
'''


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """

    logger.warning(f"hparams provided: {hparams}")
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------

    model = Classifier(hparams)
    
    time_now = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    # set up the ckpt and logging dirs



    # update ckpt and logs dir based on the dataset
    
    ckpt_dir = f"../ckpts/{hparams.dataset}/{hparams.encoder_model}/version_{time_now}"
    log_dir = f"../logs/{hparams.dataset}/"

    # update ckpt and logs dir based on whether plm (encoder) was frozen during training

    if hparams.nr_frozen_epochs > 0:
        logger.warning(f"Freezing the encoder/plm for {hparams.nr_frozen_epochs} epochs")
        ckpt_dir = f"../ckpts/{hparams.dataset}/frozen_plm/{hparams.encoder_model}/version_{time_now}"
        log_dir = f"../logs/{hparams.dataset}/frozen_plm/"

    #setup checkpoint and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ckpt_dir}",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor=hparams.monitor,
        mode=hparams.metric_mode,
        save_last = True
    )

    tb_logger = TensorBoardLogger(
        save_dir=f"{log_dir}",
        version="version_" + time_now,
        name=f'{hparams.encoder_model}',
    )

    # early stopping based on val loss
    early_stopping_callback = EarlyStopping(monitor=hparams.monitor, mode = hparams.metric_mode, patience=hparams.max_epochs)

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        gpus=[hparams.gpus],
        log_gpu_memory="all",
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        checkpoint_callback = checkpoint_callback,
        callbacks = [early_stopping_callback],
        max_epochs=hparams.max_epochs,
        default_root_dir=f'./classifier_pipeline/{hparams.encoder_model}',
        strategy = hparams.accelerator
    )

    print(f"trainer is {trainer}")
    # ------------------------
    # 6 START TRAINING
    # ------------------------

    # datamodule = MedNLIDataModule
    trainer.fit(model, model.data)
    trainer.test(model, model.data.test_dataloader())

    cms = np.array(model.test_conf_matrices)
    np.save(f'../experiments/{model.hparams.encoder_model}/test_confusion_matrices.npy',cms)


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
        "--monitor", default="monitor_balanced_accuracy", type=str, help="Quantity to monitor."
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
        "--max_steps",
        default=20000,
        type=int,
        help="Limits number of steps i.e. number of gradient backward passes/optimizer updates",
    )

    parser.add_argument(
        "--n_warmup_steps",
        default=300,
        type=int,
        help="Warmup steps before learning rate scheduler begins",
    )

    parser.add_argument(
        "--optimizer",
        default="adafactor",
        type=str,
        help="Optimization algorithm to use e.g. adamw, adafactor",
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
    parser.add_argument("--gpus", type=int, default=1, help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

    # use ddp 
    parser.add_argument("--accelerator",default = None, type=str, help ="whether or not to use data paralell and switch accelerator for trainer class. Use dp for multiple gpus 1 machine")
    

    # each LightningModule defines arguments relevant to it
    parser = Classifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)