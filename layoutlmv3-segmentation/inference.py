# Inference script for LayoutLMv3 model

import argparse
import cv2
from typing import List

from pathlib import Path

from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, default_setup
from tqdm import tqdm


def create_config(args):
    """
    Creates detectron2 config for the network
    Args:
        args: Arguments from the command line
    """

    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file(args.config_path)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
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


def run_inference(args, cfg):
    """
    Runs the inference from the given arguments and network config
    Args:
        args: Arguments from the command line
        cfg: Network config
    """
    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Load files
    images = load_images(args.image_path)

    # Make output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute number of params
    model = predictor.model
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')

    # Run inference
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
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
        cv2.imwrite(str(output_path / f'{image_path.name}.jpg'), result_image)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--image-path', type=str, required=True, help='Path to the image to be processed')
    argparser.add_argument('--config-path', metavar='FILE', required=True, help='Path to the model config file')
    argparser.add_argument('--output-path', type=str, default='.', required=False, help='Path to the output directory')
    argparser.add_argument('--opts', help='Modify config options using the command-line', default=[],
                           nargs=argparse.REMAINDER)

    args = argparser.parse_args()
    cfg = create_config(args)

    run_inference(args, cfg)


if __name__ == '__main__':
    main()
