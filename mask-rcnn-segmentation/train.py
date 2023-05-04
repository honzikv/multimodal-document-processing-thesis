import argparse
import pathlib
import yaml
import wandb

from pycocotools.coco import COCO
from detectron_init import register_dataset, build_cfg, TEST_DATASET_NAME
from heimatkunde_trainer import HeimatkundeTrainer


def train(args):
    if not args.config_yaml.exists():
        raise ValueError(f'Config file {args.config_yaml} does not exist')

    yaml_config = yaml.safe_load(args.config_yaml.read_text())

    if not args.no_wandb:
        wandb.init(project='detectron2-heimatkunde', sync_tensorboard=True, config=yaml_config)

    if args.n_classes is not None:
        n_classes = args.n_classes
    else:
        n_classes = len(COCO(args.train_json).loadCats(COCO(args.train_json).getCatIds()))

    # Register the dataset
    register_dataset(args.train_json, args.test_json)

    # Build the config
    cfg = build_cfg(yaml_config, n_classes)
    trainer = HeimatkundeTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train-json', type=pathlib.Path, required=True)
    argparser.add_argument('--test-json', type=pathlib.Path, required=True)
    argparser.add_argument('--config-yaml', type=pathlib.Path, required=True)
    argparser.add_argument('--no-wandb', action='store_true', required=False, default=False,
                           help='Disable wandb logging')
    argparser.add_argument('--n-classes', type=int, required=False, default=None,
                           help='Number of classes in the dataset. If not provided, '
                                'it will be inferred from the train json')

    args = argparser.parse_args()
    train(args)
