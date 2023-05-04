import torch.utils.data
import transformers
import os

from models.bert.preprocessing import prepare_examples
from dataset import data_loading
from models.bert.bert import BertLightning, DEFAULT_BERT_MODEL_NAME
from training_base import load_training_args, create_trainer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

MODEL_NAME_CHECKPOINT = 'BERT'
WANDB_PROJECT = 'heimatkunde-bert-classification'
DEFAULT_HYPERPARAMS = {
    'learning_rate': 1e-5,
    'epochs': 20,
    'batch_size': 4,
    'optimizer': 'adamw',
    'bert_model_name': DEFAULT_BERT_MODEL_NAME,
    'warmup_steps': 1000,
    'dataset_key': 'BERT_MASKED_DATA_DIR',
}


def main():
    # Load environment config and hyperparams
    environment_config, hyperparams = load_training_args(default_hyperparams=DEFAULT_HYPERPARAMS)

    bert_model_name = hyperparams['bert_model_name']
    dataset_key = hyperparams['dataset_key']
    print(f'Using dataset: {dataset_key} -> {environment_config[dataset_key]}')

    # Create BERT tokenizer for the given BERT model
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model_name)

    # Load the dataset
    train_dataset, test_dataset, n_classes, class_names, random_sampler = data_loading.load_dataset(
        dataset_root_path=environment_config[dataset_key],
        prepare_examples_fn=lambda examples: prepare_examples(examples, bert_tokenizer),
        remove_non_label_cols=True,
    )

    batch_size = hyperparams['batch_size']
    print(f'Creating data loaders with batch size: {batch_size}')

    # Create train and test data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Creating model...')
    print(f'Using BERT model: {bert_model_name}')
    print(f'Training for {hyperparams["epochs"]} epochs.')
    print(f'Using optimizer: {hyperparams["optimizer"]}')
    print(f'Using learning rate: {hyperparams["learning_rate"]}')
    print(f'Using warmup steps: {hyperparams["warmup_steps"]}')

    total_steps = None if hyperparams['warmup_steps'] is None else len(train_loader) * hyperparams['epochs']

    # Create the model
    model = BertLightning(
        n_classes=n_classes,
        model_name=bert_model_name,
        learning_rate=hyperparams['learning_rate'],
        optimizer_fn_name=hyperparams['optimizer'],
        class_names=class_names,
        warmup_steps=hyperparams['warmup_steps'],
        total_steps=total_steps,
    )

    # Create the trainer
    trainer = create_trainer(
        seed=environment_config['SEED'],
        epochs=hyperparams['epochs'],
        model_name_checkpoint=MODEL_NAME_CHECKPOINT,
        wandb_project=None if environment_config['no_wandb'] else WANDB_PROJECT,
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
