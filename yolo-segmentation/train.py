from pathlib import Path

from ultralytics import YOLO

import yaml
import wandb
import argparse

PRETRAINED_MODELS = {
    'n': 'yolov8n-seg.pt',
    'm': 'yolov8m-seg.pt',
    's': 'yolov8s-seg.pt',
    'l': 'yolov8l-seg.pt',
    'x': 'yolov8x-seg.pt',
}

OPTIMIZERS = {'SGD', 'Adam', 'AdamW', 'RMSProp'}
DEFAULTS = {
    'save_period': 5,
    'n_warmup_epochs': 10,
    'optimizer': 'SGD',
    'initial_lr': 0.01,
    'final_lr': 0.01,
    'n_epochs': 100,
    'batch_size': -1,  # Will trigger auto-detection
    'image_size': 1280,
    'use_pretrained': True,
    'verbose': False,
    'model_size': 'l',
}


def merge_with_defaults(config: dict):
    for key in DEFAULTS.keys():
        if key not in config:
            config[key] = DEFAULTS[key]

    return config


def load_config(yaml_path: Path, dataset_yaml: str):
    if not yaml_path.exists():
        print(f'Config file "{yaml_path}" does not exist, using default config')
        config = DEFAULTS
        config['dataset_yaml'] = dataset_yaml
        return config

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        config['dataset_yaml'] = dataset_yaml

    return merge_with_defaults(config)


def run_training(dataset_yaml: str, config_yaml: str):
    config = load_config(Path(config_yaml), dataset_yaml)

    print(f'Running with config: {config}')
    if config['model_size'] not in PRETRAINED_MODELS.keys():
        raise ValueError(f'Invalid model size: {config["model_size"]}')

    if config['optimizer'] not in OPTIMIZERS:
        raise ValueError(f'Invalid optimizer: {config["optimizer"]}, must be one of {list(OPTIMIZERS)}')

    model = YOLO(PRETRAINED_MODELS[config['model_size']])

    # Train the model
    print(f'Training model {PRETRAINED_MODELS[config["model_size"]]}')
    model.train(
        data=config['dataset_yaml'],
        epochs=config['n_epochs'],
        batch=config['batch_size'],
        imgsz=config['image_size'],
        pretrained=config['use_pretrained'],
        optimizer=config['optimizer'],
        verbose=config['verbose'],
        warmup_epochs=config['n_warmup_epochs'],
        save_period=config['save_period'],
        lr0=config['initial_lr'],
        lrf=config['final_lr'],
        save=True,
        workers=1,
    )


if __name__ == '__main__':
    wandb.tensorboard.patch(root_logdir='runs/segment/train')
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset-yaml', type=str, help='Path to the dataset', required=True)
    argparser.add_argument('--config-yaml', type=str, help='Path to the config file', required=True)
    argparser.add_argument('--no-wandb', action='store_true', required=False, default=False,
                           help='Disable wandb logging')

    args = argparser.parse_args()

    if not args.no_wandb:
        wandb.init(project='yolo-heimatkunde', sync_tensorboard=True)

    run_training(args.dataset_yaml, args.config_yaml)
