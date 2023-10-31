import argparse
import json
import os
from datetime import datetime
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.datamodule import DataModule
from utils.load_model import load_model

# Set Constants
SEED = 10
pl.seed_everything(SEED)
EXPERIMENTS_DIR = "experiments"
EXPERIMENT_TIME = datetime.now().strftime("%Y-%m-%d_%H:%M")


def setup_arguments(print_args: bool = True, save_args: bool = True):
    """
    Set up and return command-line arguments.
    """
    parser = argparse.ArgumentParser("Train script")

    # Training Configurations
    parser.add_argument("--config", type=str, required=True, help="Path to configs")

    # Trainer Configurations
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--ckpt_path", type=str, default=None)

    # Logging Configurations
    parser.add_argument(
        "--project",
        type=str,
        default="Lightning generative models",
        help="W&B project name.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=EXPERIMENT_TIME,
        help="W&B experiment name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume W&B.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="W&B run ID to resume from.",
    )

    args = parser.parse_args()

    # Load json file configs
    with open(args.config, "r") as f:
        args.config = json.load(f)

    # Creates an experiment directory
    args.experiment_dir = os.path.join(
        EXPERIMENTS_DIR, args.config["model"]["name"], args.experiment_name
    )
    os.makedirs(args.experiment_dir, exist_ok=True)

    if print_args:
        pprint(vars(args))

    return args


if __name__ == "__main__":
    # Load args
    args = setup_arguments(print_args=True, save_args=True)

    # Load model, datamodule, logger, and callbacks
    model = load_model(args.config["model"])
    datamodule = DataModule(**args.config["dataset"])
    logger = WandbLogger(
        name=args.experiment_name,
        save_dir=args.experiment_dir,
        project=args.project,
        resume="must" if args.resume else None,
        id=args.id if args.resume else None,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=args.experiment_dir,
            save_last=True,
            monitor="val_loss",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=True,
            mode="min",
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.experiment_dir,
        logger=logger,
        callbacks=callbacks,
    )

    # Start training 🔥
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.ckpt_path,
    )
