import argparse
from pathlib import Path

import yaml
import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from tqdm import tqdm


N_CLASSES = 7


def get_segmentation_model_cfg(segmentation_model_config_path: Path):
    """
    Returns the config for the segmentation model.

    Args:
        segmentation_model_config_path: path to the segmentation model config file
    """

    with open(segmentation_model_config_path, 'r', encoding='utf-8') as file:
        local_config = yaml.safe_load(file)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(local_config['model_name']))
    cfg.MODEL.WEIGHTS = local_config['model_weights']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = N_CLASSES
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.HEIMATKUNDE_DATA_DIR_TRAIN = local_config['data_dir_train']
    cfg.HEIMATKUNDE_DATA_DIR_TEST = local_config['data_dir_test']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = local_config.get('output_dir', 'output')

    return cfg


def load_images(image_path: str) -> list:
    """
    Loads image or images from the path depending on the type of the path
    Args:
        image_path: Path to the image or directory with images
    """
    image_path = Path(image_path)
    if image_path.is_dir():
        images = [(cv2.imread(str(image_path / image)), image) for image in image_path.iterdir()]
    else:
        images = [(cv2.imread(str(image_path)), image_path)]
    return images


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--image-path', type=str, required=True, help='Path to the image to be processed')
    argparser.add_argument('--config-path', metavar='FILE', required=True, help='Path to the model config file')
    argparser.add_argument('--output-path', type=str, default='.', required=False, help='Path to the output directory')

    args = argparser.parse_args()


    cfg = get_segmentation_model_cfg(args.config_path)
    register_coco_instances(
        'heimatkunde_test',
        {},
        cfg.HEIMATKUNDE_DATA_DIR_TEST + '/test.json',
        cfg.HEIMATKUNDE_DATA_DIR_TEST + '/images'
    )

    predictor = DefaultPredictor(cfg)
    model = predictor.model

    # Compute number of params
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')

    images = load_images(args.image_path)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = MetadataCatalog.get('heimatkunde_test')
    for image, image_path in tqdm(images):
        outputs = predictor(image)

        instances = outputs['instances'].to('cpu')
        vis_res = Visualizer(
            image[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION,
        )

        result = vis_res.draw_instance_predictions(predictions=instances)
        result_image = result.get_image()[:, :, ::-1]

        # Save result
        cv2.imwrite(str(output_path / image_path.name), result_image)

