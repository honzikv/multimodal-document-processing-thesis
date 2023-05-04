# Ground truth model for estimating the required accuracy of the classifier to improve the segmentation
import copy
import random

import detectron2
import numpy as np
import torch

from pathlib import Path
from typing import List, Dict, Union

from detectron2.evaluation.coco_evaluation import COCOevalMaxDets
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Instances
from pycocotools.coco import COCO
from torchmetrics import Accuracy, F1Score

from dataset.segmentation_model_mapper import get_gt_label


class GroundTruthInstanceClassifier:
    """
    This is usable instead of a real classifier to estimate the accuracy required to improve the segmentation
    """

    def __init__(self,
                 coco_json_path: Path,
                 n_classes: int,
                 iou_threshold: float = .5,
                 certainty_threshold: float = .0,
                 predict_incorrectly: List[int] = None,
                 average_accuracy=None):
        """
        Args:
            coco_json_path: Path to the COCO json file
            n_classes: Number of classes
            iou_threshold: Intersection over union threshold - if the IoU is below this threshold, we ignore the prediction
        """
        self._coco = COCO(coco_json_path)

        images = self._coco.loadImgs(self._coco.getImgIds())
        self._n_classes = n_classes
        self._image_file_name_to_image = {image['file_name']: image for image in images}
        self._iou_threshold = iou_threshold
        self._certainty_threshold = certainty_threshold
        self._accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro')
        self._f1 = F1Score(task='multiclass', num_classes=n_classes, average='macro')
        self._predict_incorrectly = predict_incorrectly or []
        self._average_accuracy = average_accuracy

        if average_accuracy is not None and len(self._predict_incorrectly) > 0:
            print('WARNING: average_accuracy is ignored when predict_incorrectly is set')
            self._average_accuracy = None

        if len(self._predict_incorrectly) > 0:
            print(f'Predicting classes: {self._predict_incorrectly} incorrectly')

    def _get_image_annotations(self, image_file_name: str) -> List[Dict]:
        image = self._image_file_name_to_image[image_file_name]
        return self._coco.loadAnns(self._coco.getAnnIds(imgIds=image['id']))

    def _invalid_prediction(self, prediction):
        return prediction + 1 % self._n_classes

    def _predict(self, pred_class):
        if self._average_accuracy is not None:
            flip = random.random()
            return pred_class if flip < self._average_accuracy else self._invalid_prediction(pred_class)

        if self._predict_incorrectly is not None and pred_class in self._predict_incorrectly:
            return self._invalid_prediction(pred_class)

        return pred_class

    def inference(self, image, instances: Instances, image_file_name: Union[str, Path]) -> Instances:
        """
        During inference the classifier finds the ground truth label for each instance and replaces the prediction
        while also computing the accuracy
        """
        bboxes = instances.pred_boxes.tensor
        scores = instances.scores
        model_predictions = instances.pred_classes.to('cpu')

        if isinstance(image_file_name, str):
            image_file_name = Path(image_file_name)

        image_annotations = self._get_image_annotations(image_file_name.name)
        gt_predictions = model_predictions.clone()

        preds = []
        labels = []
        for idx, (bbox, score, prediction) in enumerate(zip(bboxes, scores, model_predictions)):
            bbox = bbox.cpu().tolist()
            gt_label = get_gt_label(bbox, image_annotations, self._iou_threshold)
            if gt_label is None:
                continue

            pred_class = gt_label[1]['category_id'] - 1  # 0-indexed
            gt_predictions[idx] = self._predict(pred_class)  # this can modify the prediction (only for testing)

            if score >= self._certainty_threshold:
                preds.append(prediction)
                labels.append(pred_class)

        predictions = {
            'scores': instances.scores,
            'pred_boxes': instances.pred_boxes,
            'pred_classes': gt_predictions,
        }

        if hasattr(instances, 'pred_masks'):
            predictions['pred_masks'] = instances.pred_masks

        if len(preds) > 0:
            self._accuracy(torch.tensor(preds), torch.tensor(labels))
            self._f1(torch.tensor(preds), torch.tensor(labels))

        return Instances(
            image_size=instances.image_size,
            **predictions,
        )

    @property
    def accuracy(self):
        return self._accuracy.compute().item()

    @property
    def f1(self):
        return self._f1.compute().item()

    def reset(self):
        self._accuracy.reset()
