import datasets
import json

from pathlib import Path
from typing import List, Tuple
from PIL import Image


def preprocess_tokens(tokens: List[str]):
    """
    Preprocesses tokens, substituting ſ with s and lowercasing
    """
    mapped = []
    for token in tokens:
        token = token.replace('ſ', 's')
        token = token.lower()
        mapped.append(token)

    return mapped


def load_image(image_path: Path):
    image = Image.open(image_path)
    # image = image.convert('RGB')
    width, height = image.size
    return image, (width, height)


def normalize_bboxes(bboxes: List[Tuple[float, float, float, float]], size):
    return [[
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ] for bbox in bboxes]


class HeimatkundeClassificationDataset(datasets.GeneratorBasedBuilder):
    """
    Heimatkunde dataset config for custom loading via Huggingface
    """

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='heimatkunde_classification', version=VERSION, description="Heimatkunde"),
    ]

    def _info(self):
        # Specify features and return dataset info
        features = datasets.Features(
            {
                'id': datasets.Value('int32'),
                'image': datasets.features.Image(),
                'tokens': datasets.Sequence(datasets.Value('string')),
                'bboxes': datasets.Sequence(datasets.Sequence(datasets.Value('int32'))),
                'label': datasets.Value('int32'),
                'bbox_features': datasets.Sequence(datasets.Value('float32')),
            }
        )
        return datasets.DatasetInfo(
            description="Heimatkunde",
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_path = Path(self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': data_path / 'train' / 'train.json',
                    'images_path': data_path / 'train' / 'images',
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': data_path / 'test' / 'test.json',
                    'images_path': data_path / 'test' / 'images',
                    'split': 'test',
                }
            ),
        ]

    def _generate_examples(self, filepath: Path, images_path: Path, split: str):
        """
        Generates examples from the Heimatkunde dataset
        Args:
            filepath: Path to the JSON file containing the dataset
            images_path: Path to the directory containing the images
            split: Split name - 'train' or 'test'
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for example in data:
            sample_id = example['id']
            image_path = images_path / f'{sample_id}.jpg'
            image, size = load_image(image_path)

            if not image_path.exists():
                raise FileNotFoundError(f'Image {image_path} does not exist')

            tokens = preprocess_tokens(example['tokens'])
            bboxes = normalize_bboxes(example['token_bboxes'], size)
            bbox_features = example['bbox_features']
            label = example['label']

            yield sample_id, {
                'id': sample_id,
                'image': image,
                'tokens': tokens,
                'bboxes': bboxes,
                'label': label,
                'bbox_features': bbox_features,
            }
