import wandb
import warnings
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.datasets import register_coco_instances
from ditod import add_vit_config, MyTrainer


# Disable the FutureWarning related to the `device` argument
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set logging level to WARNING to avoid printing unnecessary information
logging.getLogger('transformers').setLevel(logging.WARNING)

def setup(args):
    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if not args.no_wandb:
        # Init wandb
        wandb.init(
            project='layoutlmv3-heimatkunde',
            config=cfg,
            sync_tensorboard=True,
            name=args.run_name,
        )

    # Register heimatkunde dataset instances
    register_coco_instances(
        'heimatkunde_train',
        {},
        cfg.HEIMATKUNDE_DATA_DIR_TRAIN + '/train.json',
        cfg.HEIMATKUNDE_DATA_DIR_TRAIN + '/images'
    )

    register_coco_instances(
        'heimatkunde_test',
        {},
        cfg.HEIMATKUNDE_DATA_DIR_TEST + '/test.json',
        cfg.HEIMATKUNDE_DATA_DIR_TEST + '/images'
    )

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    argparser = default_argument_parser()
    argparser.add_argument('--debug', action='store_true', help='enable debug mode')
    argparser.add_argument('--run-name', type=str, help='Name of the run', required=False, default=None)
    argparser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = argparser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    wandb.finish()
