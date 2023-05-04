import cv2

from pathlib import Path

import numpy as np
from detectron2.structures import Instances, Boxes
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image


class YoloSegmentationModelWrapper:
    """
    Wrapper for Ultralytics YOLO model to be usable in the Detectron2 framework for easier evaluation.
    """

    def __init__(self, weights_path: Path):
        """
        Initializes the wrapper.
        Args:
            weights_path: path to the weights of the YOLO model, this model must be trained for segmentation
        """
        self.model = YOLO(weights_path)

        torch_model = self.model.model
        # Compute number of parameters
        pytorch_total_params = sum(p.numel() for p in torch_model.parameters())
        print(f'Number of parameters: {pytorch_total_params}')

    def __call__(self, inputs, output_tensors=True):
        """
        Calls the model. Inputs need to be a dict with a key 'file_name' that contains the path to the image.
        Args:
            inputs: dict with a key 'file_name' that contains the path to the image
        """
        batch_instances = []
        for input in inputs:
            image_file_name = input['file_name']
            image = cv2.imread(image_file_name)

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            instances = self._predict_instances(image)
            batch_instances.append({'instances': instances})

        return batch_instances

    def _predict_instances(self, image: np.ndarray) -> Instances:
        """
        Predicts instances from the image
        """
        results = self.model(image)[0]
        boxes = results.boxes
        boxes = boxes.cpu()

        # Masks are output in a different shape than the image, which is not compatible with the COCOEvaluator,
        # and it will compute wrong AP values. Therefore, we need to scale the masks to the image size.
        # Thankfully, this is implemented with scale_image in the YOLO code, so we can just use that.
        masks = results.masks.data.cpu().numpy()
        masks = masks.transpose(1, 2, 0)  # convert num_masks x H x W to H x W x num_masks
        masks = scale_image(masks.shape[:2], masks, image.shape)
        masks = masks.transpose(2, 0, 1)  # convert H x W x num_masks to num_masks x H x W
        classes = boxes.cls  # classes
        scores = boxes.conf  # confidence
        boxes = boxes.xyxy  # x1, y1, x2, y2

        return Instances(
            image_size=image.shape[:2],
            pred_boxes=Boxes(boxes),
            pred_classes=classes.int(),
            scores=scores,
            pred_masks=masks,
        )


if __name__ == '__main__':
    # Test the wrapper
    model_path = '/mnt/e/ml-models/yolo/a/weights/best.pt'

    wrapper = YoloSegmentationModelWrapper(model_path)
    image_path = '/mnt/c/ml-data/heimatkunde-v4-listings/test-images/heimatkunde-a15539_0560.jpg'
    print(wrapper({'file_name': image_path}))
