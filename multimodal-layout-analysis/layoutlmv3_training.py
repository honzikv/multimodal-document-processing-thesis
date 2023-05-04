import torch.utils.data

from transformers import AutoProcessor

from training_base import load_training_args, create_trainer
from models.layoutlmv3.layoutlmv3 import LayoutLMv3Lightning
from models.layoutlmv3.layoutlmv3_preprocessing import prepare_examples

from dataset import data_loading

MODEL_NAME_CHECKPOINT = 'LAYOUTLMV3'
WANDB_PROJECT = 'heimatkunde-layoutlmv3-classification'
LAYOUTLMV3_PROCESSOR = 'microsoft/layoutlmv3-base'
DEFAULT_HYPERPARAMS = {
    'precision': 16,
    'learning_rate': 3e-5,
    'epochs': 10,
    'batch_size': 4,
    'optimizer': 'adamw',
    'warmup_steps': 1000,
    'dataset_key': 'CLASSIFIER_DATA_DIR',
}


def main():
    # Load environment config and hyperparams
    environment_config, hyperparams = load_training_args(default_hyperparams=DEFAULT_HYPERPARAMS)

    dataset_key = hyperparams['dataset_key']
    print(f'Using dataset: {dataset_key} -> {environment_config[dataset_key]}')

    # Create LayoutLMv3 processor
    processor = AutoProcessor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)

    # Load the dataset
    train_dataset, test_dataset, n_classes, class_names, random_sampler = data_loading.load_dataset(
        dataset_root_path=environment_config[dataset_key],
        prepare_examples_fn=lambda examples: prepare_examples(examples, processor),
        remove_non_label_cols=True,
    )

    batch_size = hyperparams['batch_size']
    print(f'Creating data loaders with batch size: {batch_size}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Creating model...')
    print(f'Training for {hyperparams["epochs"]} epochs')
    print(f'Using optimizer: {hyperparams["optimizer"]}')
    print(f'Using learning rate: {hyperparams["learning_rate"]}')
    print(f'Using warmup steps: {hyperparams["warmup_steps"]}')

    total_steps = None if hyperparams['warmup_steps'] is None else len(train_loader) * hyperparams['epochs']

    # Create the model
    model = LayoutLMv3Lightning(
        n_classes=n_classes,
        learning_rate=hyperparams['learning_rate'],
        optimizer_fn_name=hyperparams['optimizer'],
        class_names=class_names,
        warmup_steps=hyperparams['warmup_steps'],
        total_steps=total_steps,
    )

    # Create the trainer
    trainer = create_trainer(
        float_bit_precision=hyperparams['precision'],
        seed=environment_config['SEED'],
        epochs=hyperparams['epochs'],
        model_name_checkpoint=MODEL_NAME_CHECKPOINT,
        wandb_project=None if environment_config['no_wandb'] else WANDB_PROJECT,
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
