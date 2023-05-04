import cv2

from typing import List
from pathlib import Path

from detectron2.evaluation import DatasetEvaluator

from models.instance_classifier import InstanceClassifier


from detectron2.evaluation import COCOEvaluator


class MultimodalCOCOEvaluator(DatasetEvaluator):
    """
    This class is a wrapper over standard COCOEvaluator in detectron2.
    It allows us to use predictions from the multimodal classifier in the detectron2 framework
    """

    def __init__(self,
                 instance_classifier: InstanceClassifier,
                 dataset_name: str,
                 images_path: Path = None,
                 output_dir=None):
        """
        Initializes the evaluator

        Args:
            instance_classifier: instance classifier
            dataset_name: name of the dataset
            output_dir: output directory
        """
        self._coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
        self._instance_classifier = instance_classifier
        self._images_path = images_path

    def process(self, inputs, outputs):
        """
        Intercept code for process of the COCO evaluator
        Here we predict with the classification model if the segmentation
        score is above the threshold
        """

        processed_outputs = []
        for input, output in zip(inputs, outputs):
            image_file_name = input['file_name']
            image = cv2.imread(image_file_name)
            processed_outputs.append(self._instance_classifier.inference(
                image=image,
                instances=output['instances'],
                image_file_name=image_file_name,
            ))

        # Delegate the rest of the process to the COCO evaluator
        self._coco_evaluator.process(inputs, [{'instances': output} for output in processed_outputs])

    def reset(self):
        self._coco_evaluator.reset()

    def evaluate(self, image_ids=None):
        return self._coco_evaluator.evaluate(image_ids)
