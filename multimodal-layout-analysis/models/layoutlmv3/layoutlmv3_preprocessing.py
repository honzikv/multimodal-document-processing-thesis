from typing import List, Dict, Any, Tuple

from models.preprocessing import ClassifierPreprocessor


def prepare_examples(examples, processor):
    """
    Prepare examples for the network
    """
    image = examples['image']
    words = examples['tokens']
    boxes = examples['bboxes']

    encoding = processor(image, words, boxes=boxes, padding='max_length', truncation=True, return_tensors='pt')
    encoding['label'] = examples['label']

    return encoding


def normalize_bboxes(bboxes: List[Tuple[float, float, float, float]], size):
    """
    Normalize bounding boxes to 1000x1000
    Args:
        bboxes: list of bounding boxes
        size: image size - (width, height)
    """
    return [[
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ] for bbox in bboxes]


class LayoutLMv3ClassifierPreprocessor(ClassifierPreprocessor):

    def __init__(self, layoutlmv3_processor):
        self._layoutlmv3_processor = layoutlmv3_processor

    def preprocess(self, tokens: List[str], bboxes: List[tuple], image) -> List[Dict[str, Any]]:
        boxes = normalize_bboxes(bboxes, (image.shape[1], image.shape[0]))
        return self._layoutlmv3_processor(
            image,
            tokens,
            boxes=boxes,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
