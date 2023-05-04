import json
import shutil
import cv2
import tqdm

from pathlib import Path
from typing import List, Dict, Tuple

from pycocotools.coco import COCO
from detectron2.structures.instances import Instances

from models.model_pipeline_inference import ModelPipelineInference

BoundingBox = Tuple[float, float, float, float]


def bbox_intersection(bbox_a: BoundingBox, bbox_b: BoundingBox):
    """
    Compute the intersection over union of two bounding boxes
    Args:
        bbox_a: First bounding box
        bbox_b: Second bounding box
    Returns:
        Intersection over union
    """
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)
    return iou


def bbox_to_segmentation(bbox: BoundingBox):
    """
    Converts COCO bounding box to segmentation
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]

    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def get_gt_label(predicted_bbox: BoundingBox, image_annotations: List[Dict], iou_threshold: float):
    """
    Get the ground truth label for the predicted bounding box.
    Easiest way to perform this is to compute the intersection over union (IoU) between the predicted bounding box
    and all the ground truth bounding boxes. If the IoU is above a certain threshold, we consider the ground truth
    bounding box as a match. Otherwise the network is not confident enough and we ignore the prediction.
    Args:
        predicted_bbox: Predicted bounding box
        image_annotations: Annotations for the image
        iou_threshold: Intersection over union threshold - if the IoU is below this threshold, we ignore the prediction
    Returns:
        Ground truth label or None if no label was found
    """
    candidate = None
    # Iterate over all annotations and find the one with the highest IoU
    for annotation in image_annotations:
        anno_bbox = annotation['bbox']
        anno_bbox = anno_bbox[0], anno_bbox[1], anno_bbox[0] + anno_bbox[2], anno_bbox[1] + anno_bbox[3]
        anno_iou = bbox_intersection(anno_bbox, predicted_bbox)

        if anno_iou < iou_threshold:
            continue

        if candidate is None or candidate[0] < anno_iou:
            if candidate:
                print(f'Found better candidate: {candidate[0]} -> {anno_iou}')
            candidate = anno_iou, annotation

    return candidate


class SegmentationModelMapper:
    """
    SegmentationModelMapper uses the segmentation model to map the given COCO dataset to a new coco dataset that
    contains annotations from the model predictions. I.e. we create new annotations that are based on the model
    predictions. This could be useful when trying to improve accuracy of the predictions
    """

    def __init__(self, coco_json_path: Path, model_pipeline: ModelPipelineInference, iou_threshold: float = .5,
                 certainty_threshold: float = .5):
        self._coco_json_path = coco_json_path
        self._base_image_path = coco_json_path.parent / 'images'
        assert self._base_image_path.exists(), f'Images directory {self._base_image_path} does not exist'
        self._coco = COCO(coco_json_path)
        self._model_pipeline = model_pipeline
        self._iou_threshold = iou_threshold
        self._certainty_threshold = certainty_threshold

        self._model_pipeline.use_classifier = False  # disable classifier to predict from segmentation
        image_infos = self._coco.loadImgs(self._coco.getImgIds())
        self._image_id_to_image = {image['id']: image for image in image_infos}
        self._image_id_to_annotations = {image['id']: self._coco.loadAnns(self._coco.getAnnIds(imgIds=image['id']))
                                         for image in image_infos}

        # Mapping results
        self._next_annotation_id = 0
        self._extracted_categories = self._coco.loadCats(self._coco.getCatIds())
        self._extracted_annotations = []  # mapped examples
        self._extracted_images = {}  # image_id -> image info dict
        self._result = None

    def map(self):
        """
        Map the dataset to the examples and returns them in a list
        """
        for image_id, image_info in tqdm.tqdm(self._image_id_to_image.items()):
            image_annotations = self._image_id_to_annotations[image_id]
            image = cv2.imread(str(self._base_image_path / image_info['file_name']))
            instances = self._model_pipeline(image)
            self._process_instances(image_info, instances, image_annotations)

        self._result = {
            'annotations': self._extracted_annotations,
            'categories': self._extracted_categories,
            'images': list(self._extracted_images.values()),
            'info': self._coco.dataset['info'],
        }

        self._result['info']['description'] = 'Segmentation model mapped dataset'
        self._result['info']['n_annotations'] = len(self._extracted_annotations)

    def save(self, coco_output_path: Path, save_images: bool = True):
        """
        Save the mapped dataset to the given path.
        self.map() must be called before calling this method
        Args:
            coco_output_path: Path to save the dataset to - json output, e.g. 'train.json', 'val.json', etc.
            save_images: Whether to save the images to the same directory as the json file
        """
        if self._result is None:
            raise ValueError('You must call map() before calling save()')

        coco_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coco_output_path, 'w', encoding='utf-8') as file:
            json.dump(self._result, file, indent=4)

        if save_images:
            output_image_path = coco_output_path.parent / 'images'
            output_image_path.mkdir(parents=True, exist_ok=True)
            for image in self._result['images']:
                image_filename = image['file_name']

                assert (self._base_image_path / image_filename).exists(), \
                    f'Image {image_filename} does not exist in path {self._base_image_path}'

                shutil.copy(self._base_image_path / image_filename, output_image_path / image_filename)

        print(f'Saved coco dataset to {coco_output_path}')

    def _get_gt_label(self, predicted_bbox, image_annotations: List[Dict]):
        return get_gt_label(predicted_bbox, image_annotations, self._iou_threshold)

    def _process_instances(self, image_info: Dict, instances: Instances, image_annotations: List[Dict]):
        """
        Process the instances and add them to the examples
        Args:
            image_info: Image info dictionary from the coco json
            instances: Instances object predicted by the model
            image_annotations: All annotations from the image
        """
        bboxes = instances.pred_boxes.tensor
        scores = instances.scores
        predictions = instances.pred_classes

        # Filter out invalid instances - i.e. if the segmentation network is not sure about the prediction it will
        # not be extracted
        valid_instances = scores > self._certainty_threshold
        for bbox, score, prediction in zip(bboxes[valid_instances], scores[valid_instances],
                                           predictions[valid_instances]):
            bbox = bbox.cpu().tolist()
            score = score.cpu().item()
            prediction = prediction.cpu().item()
            candidate = self._get_gt_label(bbox, image_annotations)
            if candidate is None:
                continue

            iou, annotation = candidate
            # Instances prediction bbox is in the format [x1, y1, x2, y2] while the COCO format is [x1, y1, w, h]
            coco_bbox = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            coco_seg = bbox_to_segmentation(coco_bbox)

            self._extracted_annotations.append({
                'id': self._next_annotation_id,
                'area': coco_bbox[2] * coco_bbox[3],
                'image_id': image_info['id'],
                'bbox': coco_bbox,
                'attributes': {
                    'occluded': False,
                    'segmentation_model_score': score,
                    'segmentation_pred_class': prediction,
                    'iou': iou,
                },
                'iscrowd': 0,
                'category_id': annotation['category_id'],
                'segmentation': coco_seg,
            })

            self._next_annotation_id += 1
            if image_info['id'] not in self._extracted_images:
                self._extracted_images[image_info['id']] = image_info
