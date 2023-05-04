#  LayoutLMv3 segmentation with classification model

from PIL.Image import Image
import cv2
import numpy as np

from pathlib import Path
from typing import List, NamedTuple, Union, Optional

from detectron2.config import get_cfg

from dataset.instance_preprocessor import InstancePreprocessor
from ditod import add_vit_config
from ditod.mytrainer import default_setup, DefaultPredictor
from models.base_lightning_model import MultimodalLightningModuleForClassification
from models.instance_classifier import InstanceClassifier
from models.preprocessing import ClassifierPreprocessor

SegmentationModelArgs = NamedTuple('SegmentationModelArgs', [('config_file', str), ('opts', List[str])])


def build_layoutlmv3_seg_config(args: SegmentationModelArgs):
    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    return cfg


def load_images(images_path: str) -> list:
    """
    Loads image or images from the path depending on the type of the path
    Args:
        images_path: Path to the image or directory with images
    """
    images_path = Path(images_path)
    if images_path.is_dir():
        images = [(cv2.imread(str(images_path / image)), image) for image in images_path.iterdir()]
    else:
        images = [(cv2.imread(str(images_path)), images_path)]
    return images


class ModelPipelineInference:
    """
    LayoutLMv3 segmentation with classification model.
    This class is only usable for inference
    """

    def __init__(self, args: SegmentationModelArgs,
                 classification_model: Optional[MultimodalLightningModuleForClassification],
                 classifier_preprocessor: Optional[ClassifierPreprocessor],
                 classifier_threshold: float = .75,
                 use_classifier=True,
                 instance_preprocessor: Optional[InstancePreprocessor] = None):
        """
        Creates a new instance of the model pipeline. Optional arguments can be None if the model is used only for
        segmentation prediction.
        Args:
            args: Arguments for the segmentation model
            classification_model: Classification model - None if no classification is performed
            classifier_preprocessor: Preprocessor for the classification model - None if no classification is performed
            classifier_threshold: Threshold for the classification model
            use_classifier: Whether to use the classification model
            instance_preprocessor: Preprocessor for the instances - None if no classification is performed
        """
        self._cfg = build_layoutlmv3_seg_config(args)
        self._segmentation_predictor = DefaultPredictor(self._cfg)
        self._instance_classifier = InstanceClassifier(
            classification_model=classification_model,
            classifier_preprocessor=classifier_preprocessor,
            certainty_threshold=classifier_threshold,
            instance_preprocessor=instance_preprocessor,
        )
        self.use_classifier = use_classifier

    def __call__(self, inputs: Union[Image, List[Image]]):
        if isinstance(inputs, Image) or isinstance(inputs, np.ndarray):
            inputs = [inputs]

        outputs = []
        for image in inputs:
            instances = self._segmentation_predictor(image)['instances']
            if self.use_classifier:
                instances = self._instance_classifier.inference(image, instances)
            outputs.append(instances)

        if len(outputs) == 1:
            return outputs[0]

        return outputs
