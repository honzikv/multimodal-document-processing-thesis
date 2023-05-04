import os
import datasets
import json
import argparse

DEFAULT_DATA_DIR = '/mnt/c/ml-data/heimatkunde/classifier'


def print_label_distribution(dataset, label_ids):
    total = len(dataset['label'])

    for label_id, name in label_ids.items():
        print(
            f'Label: {label_id}: {dataset["label"].count(label_id)}'
            f' ({dataset["label"].count(label_id) / total * 100:.2f}%)')
    print(f'Total: {total}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                           required=False, help='Path to benchmark directory')

    args = argparser.parse_args()
    data_dir = args.data_dir

    with open(os.path.join(data_dir, 'train', 'classes.json'), 'r', encoding='utf-8') as f:
        labels = json.load(f)

    id2label = labels['id2label']
    print(id2label)
    label_ids = {int(label): name for label, name in id2label.items()}

    dataset = datasets.load_dataset('dataset/huggingface/heimatkunde_classification_dataset.py',
                                    data_dir=data_dir)

    print('Train label distributions:\n'
          '---------------------------')
    print_label_distribution(dataset['train'], label_ids)
    print('Test label distributions:\n'
          '--------------------------')
    print_label_distribution(dataset['test'], label_ids)

    print('Merged label distributions:\n'
          '--------------------------')
    merged_labels = {'label': dataset['train']['label'] + dataset['test']['label']}
    print_label_distribution(merged_labels, label_ids)
