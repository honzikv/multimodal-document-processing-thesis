import argparse

from pathlib import Path

from dataset.classifier_dataset_mapper import ClassifierDatasetMapper

DEFAULT_TRAIN_JSON_PATH = Path('/mnt/c/ml-data/heimatkunde/train.json')
DEFAULT_TEST_JSON_PATH = Path('/mnt/c/ml-data/heimatkunde/test.json')
DEFAULT_TRAIN_OUTPUT_PATH = Path('/mnt/c/ml-data/heimatkunde/classification/train/train.json')
DEFAULT_TEST_OUTPUT_PATH = Path('/mnt/c/ml-data/heimatkunde/classification/test/test.json')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train-json-path', type=Path, required=False,
                           help='Path to the train json file',
                           default=DEFAULT_TRAIN_JSON_PATH)
    argparser.add_argument('--test-json-path', type=Path, required=False,
                           help='Path to the test json file',
                           default=DEFAULT_TEST_JSON_PATH)
    argparser.add_argument('--train-output-path', type=Path, required=False,
                           help='Path to the train output json file',
                           default=DEFAULT_TRAIN_OUTPUT_PATH)
    argparser.add_argument('--test-output-path', type=Path, required=False,
                           help='Path to the test output json file',
                           default=DEFAULT_TEST_OUTPUT_PATH)
    argparser.add_argument('--ocr-lang', type=str, required=False,
                           help='Language for the OCR model. Should be the name of a specific '
                                '.traineddata file without the traineddata extension',
                           default=None)  # None set it to the default value

    args = argparser.parse_args()

    ClassifierDatasetMapper(
        json_path=args.train_json_path,
        output_json_path=args.train_output_path,
        resize_method='resize',
        ocr_lang=args.ocr_lang,
    ).process()

    ClassifierDatasetMapper(
        json_path=args.test_json_path,
        output_json_path=args.test_output_path,
        resize_method='resize',
        ocr_lang=args.ocr_lang,
    ).process()
