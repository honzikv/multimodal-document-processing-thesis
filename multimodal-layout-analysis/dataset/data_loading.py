import json
from typing import Callable, Dict, Any, Union

import datasets

from pathlib import Path
from torch.utils.data import WeightedRandomSampler


def get_weighted_random_sampler(dataset: datasets.Dataset, n_classes: int) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for a dataset.
    Args:
        dataset: train dataset
        n_classes: number of classes
    """
    total = len(dataset['label'])
    class_counts = [0 for _ in range(n_classes)]
    for label_id in dataset['label']:
        class_counts[label_id] += 1

    class_weights = [total / class_count for class_count in class_counts]
    example_weights = [class_weights[label_id] for label_id in dataset['label']]

    return WeightedRandomSampler(example_weights, total)


def load_unprocessed_dataset(dataset_root_path: Union[str, Path]):
    if not isinstance(dataset_root_path, Path):
        dataset_root_path = Path(dataset_root_path)

    labels = dataset_root_path / 'train' / 'classes.json'

    with open(labels, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    label2id = labels['label2id']
    id2label = labels['id2label']

    dataset = datasets.load_dataset(
        'dataset/huggingface/heimatkunde_classification_dataset.py',
        data_dir=str(dataset_root_path),
    )

    return dataset, label2id, id2label


def load_dataset(dataset_root_path: Path,
                 prepare_examples_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                 return_in_torch_format=True,
                 remove_non_label_cols=True,
                 cols_to_keep=None):
    """
    Loads the dataset and calls batched version of prepare_examples_fn. This function takes in a set of examples just
    as a normal prepare_examples in huggingface.

    Args:
        dataset_root_path: path to the dataset root directory
        prepare_examples_fn: function that takes in a set of examples and returns a set of examples
        return_in_torch_format: whether to return the dataset in torch format - calls set_format('torch')
        remove_non_label_cols: remove previous columns that are not mapped in the prepare_examples_fn, column labels is
            always kept
        cols_to_keep: columns to keep in the dataset - additional columns to keep alongside the label column. Has only
            effect if remove_non_label_cols is True

    Returns:
        train_ds: train dataset
        test_ds: test dataset
        n_classes: number of classes
        class_names: list of class names
        weighted_random_sampler: weighted random sampler for the train dataset - this is used to balance the training
    """
    dataset, label2id, id2label = load_unprocessed_dataset(dataset_root_path)
    cols_to_remove = dataset['train'].column_names

    if remove_non_label_cols:
        cols_to_remove.remove('label')

    for col in cols_to_keep or []:
        cols_to_remove.remove(col)

    train_ds = dataset['train'].map(
        prepare_examples_fn,
        batched=True,
        remove_columns=cols_to_remove if remove_non_label_cols else None,
    )

    test_ds = dataset['test'].map(
        prepare_examples_fn,
        batched=True,
        remove_columns=cols_to_remove if remove_non_label_cols else None,
    )

    weighted_random_sampler = get_weighted_random_sampler(
        dataset=train_ds,
        n_classes=len(label2id),
    )

    if return_in_torch_format:
        train_ds.set_format('torch')
        test_ds.set_format('torch')

    n_classes = len(id2label)
    class_names = [None for _ in range(n_classes)]
    for label, idx in label2id.items():
        class_names[idx] = label

    return train_ds, test_ds, n_classes, class_names, weighted_random_sampler
