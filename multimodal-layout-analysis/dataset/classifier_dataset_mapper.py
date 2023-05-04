import json
import logging
import tqdm
import numpy as np

from pathlib import Path
from PIL import Image
from typing import List, Tuple

from .instance_preprocessor import InstancePreprocessor, InstancePreprocessorConfig

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

_logger = logging.getLogger(__name__)


class ClassifierDatasetMapper:
    """
    Maps the COCO json file to the format digestible by the classifier model.
    I.e. we take each annotation and treat it as a classification sample with tokens, their bboxes, image (only the bbox)
    and the class.
    """

    def __init__(self, json_path: Path, output_json_path: Path, resize_method: str = 'resize',
                 ocr_lang=None,
                 discard_classes: List[int] = None):
        """
        Constructor
        Args:
            json_path: path to the json file
            output_json_path: path to the output json file
            resize_method: method to resize the image. Legal values are 'truncate' and 'resize'
        """
        self._json_path = json_path
        self._output_json_path = output_json_path
        self._images_path = self._json_path.parent / 'images'
        self._output_images_path = output_json_path.parent / 'images'
        self._result = []
        self._next_item_id = 0
        self._coco_image_id_to_image_name = {}
        self._instance_preprocessor = InstancePreprocessor(
            InstancePreprocessorConfig(resize_method=resize_method) if ocr_lang is None
            else InstancePreprocessorConfig(resize_method=resize_method, tess_lang=ocr_lang)
        )
        self._class_info = {}
        self._discard_class_ids = [] if discard_classes is None else discard_classes

        if not self._output_images_path.exists():
            self._output_images_path.mkdir(parents=True, exist_ok=True)

    def process(self, save_classes=True):
        """
        Process the json file and saves the result to the output json file.
        Files are automatically saved to the "output json file / images" directory
        """
        with open(self._json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        self._configure_classes(json_data, save_classes)

        if 'images' not in json_data:
            raise ValueError(f'No images in json file {self._json_path}')

        if 'annotations' not in json_data:
            raise ValueError(f'No annotations in json file {self._json_path}')

        if 'categories' not in json_data:
            raise ValueError(f'No categories in json file {self._json_path}')

        for image_info in json_data['images']:
            self._coco_image_id_to_image_name[image_info['id']] = image_info['file_name']

        for annotation in tqdm.tqdm(json_data['annotations']):
            self._process_annotation(annotation)

        # Save the json file
        with open(self._output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self._result, f, indent=4)

    def _configure_classes(self, json_data: dict, save_classes):
        """
        Configures the classes from the json file
        Args:
            json_data: json data
        """

        coco_classes = {}
        next_class = 0
        for category in json_data['categories']:
            if category['id'] in self._discard_class_ids:
                # Skip this class as it is in the discard list
                continue
            category['label'] = next_class
            coco_classes[category['id']] = {
                'name': category['name'],
                'label': next_class
            }
            next_class += 1

        id2label = {coco_class['label']: coco_class['name'] for coco_class in coco_classes.values()}
        label2id = {coco_class['name']: coco_class['label'] for coco_class in coco_classes.values()}

        self._class_info = {
            'coco_classes': coco_classes,
            'id2label': id2label,
            'label2id': label2id,
            'num_classes': len(coco_classes),
        }

        if save_classes:
            with open(self._output_json_path.parent / 'classes.json', 'w', encoding='utf-8') as f:
                json.dump(self._class_info, f, indent=4)

    def _process_annotation(self, annotation: dict):
        if annotation['image_id'] not in self._coco_image_id_to_image_name:
            _logger.info(f'No image with id {annotation["image_id"]} in the json file, skipping this annotation')
            return

        if annotation['category_id'] in self._discard_class_ids:
            # Skip this class as it is in the discard list
            return

        image_name = self._coco_image_id_to_image_name[annotation['image_id']]
        image_path = self._images_path / image_name

        # Preprocess the instance with preprocessor
        image_bbox, tokens, token_bboxes, bbox_features = self._instance_preprocessor.preprocess_coco_instance(
            instance=annotation,
            image_path=image_path
        )

        coco_cls = annotation['category_id']
        cls = self._class_info['coco_classes'][coco_cls]['label']

        self._result.append({
            'tokens': tokens,
            'token_bboxes': token_bboxes,
            'label': cls,
            'id': self._next_item_id,
            'bbox_features': bbox_features,
        })

        pil_img = Image.fromarray(image_bbox).convert('RGB')
        pil_img.save(str(self._output_json_path.parent / 'images' / f'{self._next_item_id}.jpg'))
        self._next_item_id += 1
