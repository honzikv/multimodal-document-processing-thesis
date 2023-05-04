import os

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format, DatasetEvaluator
from detectron2.engine import DefaultTrainer

from detectron_init import TEST_DATASET_NAME


class HeimatkundeTrainer(DefaultTrainer):
    """
    Modified version of the default trainer which uses the COCOEvaluator
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        return COCOEvaluator(TEST_DATASET_NAME, output_dir=output_folder)
