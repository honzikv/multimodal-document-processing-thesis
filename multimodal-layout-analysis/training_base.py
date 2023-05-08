import argparse
import torch
import yaml
import pytorch_lightning as pl

from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

DEFAULT_ENVIRONMENT_CONFIG_PATH = Path('environment.yaml')


def load_yml(hyperparams_path: Path):
    with open(hyperparams_path, 'r', encoding='utf-8') as f:
        hyperparams = yaml.safe_load(f)

    return hyperparams


def merge_with_default_params(hyperparams: dict, default_hyperparams: dict):
    """
    Merges the hyperparams with the default hyperparams
    Args:
        hyperparams: hyperparams to merge
        default_hyperparams: default hyperparams

    Returns:
        merged hyperparams
    """
    for key, value in default_hyperparams.items():
        if key not in hyperparams:
            hyperparams[key] = value

    return hyperparams


def load_training_args(default_hyperparams: dict,
                       environment_config_path: Path = DEFAULT_ENVIRONMENT_CONFIG_PATH):
    """
    Loads the training arguments from the command line and from the environment config file.
    Args:
        default_hyperparams: default hyperparams to use if no hyperparams are provided
        environment_config_path: path to the environment config file

    Returns:
        environment_config: environment config
        hyperparams: hyperparams
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--environment-config-path', type=Path, required=False,
                           help='Path to the environment config yaml file. E.g. "./environment.yml"',
                           default=None)
    argparser.add_argument('--hyperparams-path', type=Path, required=False,
                           help='Path to the hyperparams yaml file. E.g. "/hyperparams/layoutlmv3/config_1.yaml"',
                           default=None)
    argparser.add_argument('--force-seed', type=int, required=False,
                           help='Force a specific seed for the random number generator. Leaving this argument out'
                                ' will instead use seed from the environment.yaml file',
                           default=None)
    argparser.add_argument('--no-wandb', action='store_true', required=False,
                           help='Disable logging to wandb',
                           default=False)

    args = argparser.parse_args()

    if args.environment_config_path is not None:
        print(f'Loading environment config from {args.environment_config_path}')
        environment_config = load_yml(args.environment_config_path)
    else:
        print(f'Using default environment config {environment_config_path}')
        environment_config = load_yml(environment_config_path)

    if args.hyperparams_path is not None:
        print(f'Loading hyperparams from {args.hyperparams_path}')
        hyperparams = load_yml(args.hyperparams_path)
        hyperparams = merge_with_default_params(hyperparams, default_hyperparams)
    else:
        print(f'Using default hyperparams {default_hyperparams}')
        hyperparams = default_hyperparams

    if args.force_seed is not None:
        print(f'Forcing seed to {args.force_seed}')
        environment_config['seed'] = args.force_seed

    environment_config['no_wandb'] = args.no_wandb
    print(f'Environment config: {environment_config}')
    print(f'Hyperparams: {hyperparams}')

    return environment_config, hyperparams


def create_trainer(seed: int,
                   epochs: int,
                   model_name_checkpoint: str,
                   float_bit_precision: int = 16,
                   accelerator: str = 'auto',
                   wandb_project: Optional[str] = None):
    """
    Creates trainer for the model

    Args:
        seed: seed for the random number generator
        epochs: number of epochs to train
        model_name_checkpoint: name of the model for checkpointing
        float_bit_precision: bit precision for the model. 16 for half precision, 32 for full precision
        accelerator: accelerator to use
        wandb_project: wandb project to use. If None, wandb is not used
    """
    wandb_logger = None
    if wandb_project is not None:
        wandb_logger = WandbLogger(project=wandb_project)

    pl.seed_everything(seed)
    # torch.set_float32_matmul_precision(torch_matmul_precision)
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename=model_name_checkpoint + '-{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}-{val_f1:.4f}',
        save_last=False,
        save_top_k=1,
        monitor='val_f1_epoch',
        mode='max',
    )

    trainer = pl.Trainer(
        precision=float_bit_precision,
        accelerator=accelerator,
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
    )

    return trainer
