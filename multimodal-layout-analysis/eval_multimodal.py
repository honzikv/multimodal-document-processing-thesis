import argparse
import os
import json
import datetime
import warnings
import logging

import detectron2.evaluation
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
import yaml

from pathlib import Path

from models.ground_truth_instance_classifier import GroundTruthInstanceClassifier
from transformers import AutoTokenizer, AutoProcessor
from eval.multimodal_evaluator import MultimodalCOCOEvaluator
from models.fusion.vitbert import ViTBertLightning
from models.fusion.vitbert_preprocessing import ViTBertFusionClassifierPreprocessor
from models.instance_classifier import InstanceClassifier
from models.layoutlmv3.layoutlmv3 import LayoutLMv3Lightning
from models.layoutlmv3.layoutlmv3_preprocessing import LayoutLMv3ClassifierPreprocessor
from models.model_pipeline_inference import build_layoutlmv3_seg_config, SegmentationModelArgs
from models.yolo.yolo_wrapper import YoloSegmentationModelWrapper

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable the FutureWarning related to the `device` argument
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set logging level to WARNING to avoid printing unnecessary information
logging.getLogger('transformers').setLevel(logging.WARNING)

SEGMENTATION_MODELS = ['layoutlmv3', 'mask_rcnn', 'yolo']
CLASSIFICATION_MODELS = ['layoutlmv3', 'vitbert', 'ground_truth', None]  # None is used for --no-classifier
N_CLASSES = 7  # Number of classes in the dataset, change this if a different dataset is used
MODEL_CERTAINTY_THRESHOLD = .5
IOU_GT_THRESHOLD = .7  # Used for ground truth classifier
DETECTION_OUTPUT_FOLDER = 'detection_output'


def get_segmentation_model_cfg(segmentation_model_name: str, segmentation_model_config_path: Path):
    """
    Returns the config for the segmentation model.
    If the segmentation model is layoutlmv3, the config is built from the config file.
    If the segmentation model is mask_rcnn or fast_rcnn, the config is built by getting the base from model zoo
    and loading the weights from the config file

    Args:
        segmentation_model_name: name of the segmentation model
        segmentation_model_config_path: path to the segmentation model config file
    """

    if segmentation_model_name == 'layoutlmv3':
        return build_layoutlmv3_seg_config(SegmentationModelArgs(str(segmentation_model_config_path), []))

    if segmentation_model_name == 'mask_rcnn' or segmentation_model_name == 'fast_rcnn':
        with open(segmentation_model_config_path, 'r', encoding='utf-8') as file:
            local_config = yaml.safe_load(file)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(local_config['model_name']))
        cfg.MODEL.WEIGHTS = local_config['model_weights']
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = N_CLASSES
        cfg.TEST.DETECTIONS_PER_IMAGE = 100

        return cfg


def get_instance_classifier(args):
    if args.class_model_name == 'ground_truth':
        print('Using ground truth instance classifier (no model loaded)')
        return GroundTruthInstanceClassifier(
            coco_json_path=args.dataset_path / 'test.json',
            n_classes=N_CLASSES,
            iou_threshold=args.iou_gt_threshold,
            certainty_threshold=args.model_certainty_threshold,
        )

    classifier_model_name = args.class_model_name
    weights_path = args.class_model_weights

    print(f'Loading instance classifier from {args.class_model_weights}...')
    if not weights_path.exists():
        print(f'Weights file not found: {weights_path}')
        exit(1)

    if classifier_model_name == 'layoutlmv3':
        model = LayoutLMv3Lightning.load_from_checkpoint(checkpoint_path=str(weights_path))
        preprocessor = LayoutLMv3ClassifierPreprocessor(
            layoutlmv3_processor=AutoProcessor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False),
        )
    elif classifier_model_name == 'vitbert':
        model = ViTBertLightning.load_from_checkpoint(checkpoint_path=str(weights_path))
        preprocessor = ViTBertFusionClassifierPreprocessor(
            bert_tokenizer=AutoTokenizer.from_pretrained(model.hparams['bert_model_name']),
            vit_feature_extractor=AutoProcessor.from_pretrained(model.hparams['vit_model_name']),
        )
    else:
        raise ValueError(f'Unknown model name: {classifier_model_name}')

    return InstanceClassifier(
        classification_model=model,
        classifier_preprocessor=preprocessor,
        certainty_threshold=args.model_certainty_threshold,
    )


def build_detectron2_model(args):
    segmentation_model_cfg = get_segmentation_model_cfg(args.seg_model_name, args.seg_model_path)
    segmentation_model = build_model(segmentation_model_cfg)
    checkpointer = DetectionCheckpointer(segmentation_model)

    print(f'Loading checkpoint from {segmentation_model_cfg.MODEL.WEIGHTS}...')
    checkpointer.load(segmentation_model_cfg.MODEL.WEIGHTS)

    return segmentation_model_cfg, segmentation_model


def build_yolo_model(args):
    segmentation_model_cfg = get_cfg()
    yolo_detectron2_wrapper = YoloSegmentationModelWrapper(weights_path=args.seg_model_path)

    return segmentation_model_cfg, yolo_detectron2_wrapper


def evaluate(args):
    # Register Heimatkunde dataset
    register_coco_instances(
        'heimatkunde_test',
        {},
        f'{args.dataset_path}/test.json',
        f'{args.dataset_path}/images'
    )

    if args.seg_model_name != 'yolo':
        segmentation_model_cfg, segmentation_model = build_detectron2_model(args)
    else:
        # Create an empty config and a YoloSegmentationModelWrapper
        segmentation_model_cfg, segmentation_model = build_yolo_model(args)

    if not args.no_classifier:
        # If classifier is enabled load the instance classifier and evaluate the model
        instance_classifier = get_instance_classifier(args)

        test_loader = DefaultTrainer.build_test_loader(segmentation_model_cfg, 'heimatkunde_test')
        print('All loaded successfully, starting evaluation...')
        eval_results = inference_on_dataset(
            segmentation_model,
            test_loader,
            # Run with the MultimodalCOCOEvaluator
            evaluator=[
                MultimodalCOCOEvaluator(
                    instance_classifier,
                    dataset_name='heimatkunde_test',
                )
            ]
        )

        if isinstance(instance_classifier, GroundTruthInstanceClassifier):
            print(f'Ground truth accuracy: {instance_classifier.accuracy}')
            print(f'Ground truth F1: {instance_classifier.f1}')

        with open(args.output_file_name, 'w', encoding='utf-8') as file:
            json.dump(eval_results, file, indent=4)

    print('Evaluating segmentation only...')
    test_loader = DefaultTrainer.build_test_loader(segmentation_model_cfg, 'heimatkunde_test')
    seg_only_results = inference_on_dataset(
        segmentation_model,
        test_loader,
        evaluator=[COCOEvaluator(dataset_name='heimatkunde_test')]
    )

    # Remove .json from the output file name
    eval_only_file_name = args.output_file_name[:-5]

    with open(f'{eval_only_file_name}_{args.seg_model_name}.seg_only.json', 'w', encoding='utf-8') as file:
        print(f'Writing segmentation only results to {eval_only_file_name}_{args.seg_model_name}.seg_only.json')
        json.dump(seg_only_results, file, indent=4)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seg-model-name', type=str, required=True,
                           choices=SEGMENTATION_MODELS,
                           help='Name of the segmentation model to use')
    argparser.add_argument('--seg-model-path', type=Path, required=False,
                           help='Path to the segmentation model yaml file or YOLO weights', default=None)

    argparser.add_argument('--class-model-name', type=str, required=False,
                           choices=CLASSIFICATION_MODELS,
                           default=None,
                           help='Name of the model to use')
    argparser.add_argument('--class-model-weights', type=Path, required=False,
                           help='Path to the classification model')

    argparser.add_argument('--dataset-path', type=Path, required=True,
                           help='Root path to the COCO dataset. E.g. "~/ml-data/mydataset".'
                                ' The dataset must contain test.json and /images directory')
    argparser.add_argument('--no-classifier', action='store_true',
                           help='If specified, no classifier will be used')
    argparser.add_argument('--output-file-name', type=str, required=False,
                           help='Path to the output file with scores, saved in JSON format',
                           default=f'eval_results_{datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")}.json')

    argparser.add_argument('--model-certainty-threshold', type=float, required=False,
                           help='Threshold for the model certainty to run the classifier',
                           default=MODEL_CERTAINTY_THRESHOLD)
    argparser.add_argument('--iou-gt-threshold', type=float, required=False,
                           help='Threshold for the ground truth IOU', default=IOU_GT_THRESHOLD)
    argparser.add_argument('--save-detection-results', action='store_true',
                           help='If specified, detection results will be saved to an output folder')
    argparser.add_argument('--detection-output-folder', type=Path, required=False,
                           help='Path to the output folder where detection results will be saved',
                           default=DETECTION_OUTPUT_FOLDER)

    args = argparser.parse_args()
    if args.class_model_name not in ['ground_truth', None] and not args.class_model_weights:
        print(f'Classification model weights must be specified for model {args.class_model_name}')
        exit(1)

    if not args.no_classifier and args.class_model_name is None:
        print(f'Classification model name must be specified if classifier is enabled')
        exit(1)

    if not args.output_file_name.endswith('.json'):
        print(f'Output file name must end with .json, appending .json to the end of the file name')
        args.output_file_name += '.json'

    # Run the evaluation
    evaluate(args)
