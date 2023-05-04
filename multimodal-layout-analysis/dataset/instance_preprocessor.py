import numpy as np
import cv2
import pytesseract as tess
import torch

from pathlib import Path
from typing import Union, Tuple
from dataclasses import dataclass

# This should fit most of the objects while being relatively small
DEFAULT_MAX_WIDTH = 1280
DEFAULT_MAX_HEIGHT = 800


def stretch_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def truncate_image(img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    Truncates the image to the given maximum width and height if the image is larger than the given size
    Args:
        img: Image to truncate
        max_width: Maximum width of the image
        max_height: Maximum height of the image
    """

    # Get the current size of the image
    height, width = img.shape[:2]

    # Create a new empty image of the desired size
    truncated = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Copy the source image to the new image, with padding or clipping as necessary
    if width <= max_width and height <= max_height:
        truncated[:height, :width, :] = img
    else:
        truncated = img[:max_height, :max_width, :]

    return truncated


def resize_image(img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    Resizes the image to the given maximum width and height if the image is larger than the given size
    Args:
        img: Image to resize
        max_width: Maximum width of the image
        max_height: Maximum height of the image
    """
    # Get the current size of the image
    height, width = img.shape[:2]

    # Check if the image is already smaller than the desired size
    if height <= max_height and width <= max_width:
        return img

    # Calculate the aspect ratio of the image
    aspect_ratio = width / height

    # Calculate the new width and height based on the aspect ratio
    new_width = int(max_height * aspect_ratio)
    new_height = int(max_width / aspect_ratio)

    # If the new height is within the desired limits, use it
    if new_height <= max_height:
        dim = (new_width, max_height)
    # Otherwise, use the new width
    else:
        dim = (max_width, new_height)

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized_img


@dataclass(frozen=True)
class InstancePreprocessorConfig:
    """
    Configuration for resizing the image
    """
    max_width: int = DEFAULT_MAX_WIDTH
    max_height: int = DEFAULT_MAX_HEIGHT
    resize_method: str = 'resize'  # truncate, resize, stretch
    tess_lang: str = 'fraktur_custom'
    psm: int = 6
    oem: int = 1

    @property
    def tess_args(self):
        return f'-l {self.tess_lang} --oem {self.oem} --psm {self.psm}'


class InstancePreprocessor:
    """
    Preprocesses the instance from the detectron2 / LayoutLMv3 model
    """

    def __init__(self, config: InstancePreprocessorConfig = None):
        """
        Creates a new instance of the preprocessor

        Args:
            config: Configuration for the preprocessor
        """
        if not config:
            config = InstancePreprocessorConfig()

        self.config = config

    @classmethod
    def _get_normalized_bbox_features(cls, image: np.ndarray, bbox: Tuple[float, float, float, float]):
        if isinstance(bbox, torch.Tensor):
            bbox = bbox[0]

        width, height = image.shape[1], image.shape[0]

        x1, y1, bbox_width, bbox_height = bbox
        x2, y2 = x1 + bbox_width, y1 + bbox_height

        # Normalize the bbox
        x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height

        # Compute normalized area
        bbox_area = (bbox_width * bbox_height) / (width * height)
        return bbox_area, x1, y1, x2, y2

    def _get_bbox_image_with_features(self, image: Union[Path, np.ndarray], bbox: tuple[float, float, float, float]):
        # Load the image if it is passed as a path
        if isinstance(image, Path):
            image = cv2.imread(str(image))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_image_features = self._get_normalized_bbox_features(image, bbox)

        # Extract bbox from the image
        if isinstance(bbox, torch.Tensor):
            # This gets called when using the detectron2 model
            x1, y1, x2, y2 = bbox[0]
            width, height = x2 - x1, y2 - y1
        else:
            # This gets called when mapping from coco dataset
            x1, y1, width, height = bbox

        bbox_image = image[int(y1):int(y1 + height), int(x1):int(x1 + width), :]

        if self.config.resize_method == 'truncate':
            bbox_image = truncate_image(bbox_image, self.config.max_width, self.config.max_height)
        elif self.config.resize_method == 'resize':
            bbox_image = resize_image(bbox_image, self.config.max_width, self.config.max_height)
        else:
            bbox_image = stretch_image(bbox_image, self.config.max_width, self.config.max_height)

        return bbox_image, bbox_image_features

    def _get_tokens_with_bboxes(self, image: np.ndarray):
        """
        OCRs the given image and returns all valid tokens. Tokens are preprocessed by replacing ſ with s.
        """
        ocr_dict = tess.image_to_data(
            image,
            output_type=tess.Output.DICT,
            config=self.config.tess_args,
        )

        n_boxes = len(ocr_dict['level'])
        tokens, bboxes = [], []
        for i in range(n_boxes):
            # Extract token
            token = ocr_dict['text'][i]

            # Skip empty tokens
            if token == '':
                continue

            # Replace ſ with s
            token = token.replace('ſ', 's')

            start_x, start_y, end_x, end_y = ocr_dict['left'][i], \
                ocr_dict['top'][i], ocr_dict['left'][i] + ocr_dict['width'][i], ocr_dict['top'][i] + \
                                    ocr_dict['height'][i]
            token_bbox = start_x, start_y, end_x, end_y

            tokens.append(token)
            bboxes.append(token_bbox)

        return tokens, bboxes

    def preprocess_detectron2_instance(self, instance, image: Union[Path, np.ndarray], threshold: float = .7):
        """
        Preprocesses an instance from the detectron2 / LayoutLMv3 model
        Args:
            instance: Instance from the detectron2 / LayoutLMv3 model
            image: Image to extract the instance from
            threshold: Threshold for the instance score
        Returns:
            Preprocessed instance or None if the score is below the threshold
        """
        predicted_bbox = instance.pred_boxes.tensor
        score = instance.scores

        if score < threshold:
            return None

        image_bbox, bbox_features = self._get_bbox_image_with_features(image, predicted_bbox)
        tokens, token_bboxes = self._get_tokens_with_bboxes(image_bbox)

        return image_bbox, tokens, token_bboxes, bbox_features

    def preprocess_coco_instance(self, instance: dict, image_path: Path):
        """
        Preprocesses an instance from COCO annotation
        Args:
            instance: Coco annotation instance
            image_path: Image to extract the instance from

        Returns:
            3-tuple of bbox image, tokens and token bboxes
        """

        bbox = instance['bbox']

        image_bbox, bbox_features = self._get_bbox_image_with_features(image_path, bbox)
        tokens, token_bboxes = self._get_tokens_with_bboxes(image_bbox)

        return image_bbox, tokens, token_bboxes, bbox_features
