import torch

from typing import Union
from pathlib import Path

from detectron2.structures import Instances
from tqdm import tqdm

from dataset.instance_preprocessor import InstancePreprocessor
from models.base_lightning_model import MultimodalLightningModuleForClassification
from models.preprocessing import ClassifierPreprocessor


class InstanceClassifier:
    """
    Wrapper around pytorch lightning model that ensures
    seamless processing of instances output from the segmentation
    """

    def __init__(self,
                 classification_model: MultimodalLightningModuleForClassification,
                 classifier_preprocessor: ClassifierPreprocessor,
                 certainty_threshold: float = .75,
                 instance_preprocessor: InstancePreprocessor = None):
        """
        Initializes the instance
        Args:
            classification_model: reference to a pytorch lightning model - must contain a inference_on_batch function
            classifier_preprocessor:
            certainty_threshold:
            instance_preprocessor:
        """
        if instance_preprocessor is None:
            instance_preprocessor = InstancePreprocessor()

        self._classification_model = classification_model
        self._classifier_preprocessor = classifier_preprocessor
        self._certainty_threshold = certainty_threshold
        self._instance_preprocessor = instance_preprocessor

    def inference(self, image, instances: Instances, image_file_name: Union[str, Path]) -> Instances:
        """
        Inference on the instances
        """

        pred_classes = instances.pred_classes.clone()
        for idx, seg_score in tqdm(enumerate(instances.scores)):
            if seg_score < self._certainty_threshold:
                continue

            image_bbox, tokens, token_bboxes, bbox_features = self._instance_preprocessor.preprocess_detectron2_instance(
                instance=instances[idx],
                image=image,
                threshold=self._certainty_threshold,
            )

            preprocessed_input = self._classifier_preprocessor.preprocess(
                tokens=tokens,
                bboxes=token_bboxes,
                image=image_bbox,

            )

            # TODO fix tuple of bbox tensors
            bbox_features = torch.tensor(list(bbox_features)).unsqueeze(0)
            preprocessed_input['bbox_features'] = bbox_features

            # TODO this could be batched
            pred_classes[idx] = self._classification_model.inference_on_batch(preprocessed_input)

        model_predictions = {
            'pred_classes': pred_classes,
            'scores': instances.scores,
            'pred_boxes': instances.pred_boxes,
        }

        if hasattr(instances, 'pred_masks'):
            model_predictions['pred_masks'] = instances.pred_masks

        return Instances(
            image_size=instances.image_size,
            **model_predictions,
        )
