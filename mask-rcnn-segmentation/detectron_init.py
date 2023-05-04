from pathlib import Path

from detectron2.config import get_cfg
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.model_zoo import model_zoo
from pycocotools.coco import COCO

TRAIN_DATASET_NAME = 'heimatkunde_train'
TEST_DATASET_NAME = 'heimatkunde_test'
DEFAULT_OUTPUT_DIR = 'models/detectron2'


def register_dataset(train_coco_json_path: Path, test_coco_json_path: Path):
    """
    Registers datasets for the detectron2 framework.
    Dataset must follow the COCO format and must have 'annotations.json' and 'images' folder in the same directory.
    Args:
        train_coco_json_path: Path to the training dataset coco json
        test_coco_json_path: Path to the test dataset coco json
    """

    if not train_coco_json_path.exists():
        raise FileNotFoundError(f'Training dataset not found at {train_coco_json_path}')

    if not test_coco_json_path.exists():
        raise FileNotFoundError(f'Test dataset not found at {test_coco_json_path}')

    register_coco_instances(
        TRAIN_DATASET_NAME,
        {},
        str(train_coco_json_path),
        str(train_coco_json_path.parent / 'images')
    )

    register_coco_instances(
        TEST_DATASET_NAME,
        {},
        str(test_coco_json_path),
        str(test_coco_json_path.parent / 'images')
    )


def build_cfg(yaml_conf: dict, n_classes: int):
    """
    Builds detectron config for training / eval
    Args:
        yaml_conf: yaml config
        n_classes: number of classes in the dataset
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_conf['model_name']))
    cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (TEST_DATASET_NAME,)
    cfg.OUTPUT_DIR = yaml_conf.get('output_dir', 'detectron2')
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_conf['model_name'])
    cfg.DATALOADER.NUM_WORKERS = yaml_conf['num_workers']
    cfg.SOLVER.IMS_PER_BATCH = yaml_conf['ims_per_batch']
    cfg.SOLVER.BASE_LR = yaml_conf['base_lr']
    cfg.SOLVER.MAX_ITER = yaml_conf['max_iter']
    cfg.SOLVER.CHECKPOINT_PERIOD = yaml_conf['checkpoint_period']
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    # cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.TEST.EVAL_PERIOD = yaml_conf['eval_period']

    return cfg
