import torch.utils.data
import transformers

from models.vit.preprocessing import prepare_examples
from dataset import data_loading
from models.vit.vit import DEFAULT_VIT_MODEL_NAME

from models.vit.vit import ViTModelLightning
from training_base import load_training_args, create_trainer

MODEL_NAME_CHECKPOINT = 'VIT'
WANDB_PROJECT = 'heimatkunde-vit-classification'
DEFAULT_HYPERPARAMS = {
    'learning_rate': 1e-5,
    'epochs': 20,
    'batch_size': 6,
    'optimizer': 'adamw',
    'vit_model_name': 'microsoft/swin-base-patch4-window7-224-in22k',  # or "microsoft/swin-tiny-patch4-window7-224"
    'warmup_steps': 1500,
    'precision': 16,
    'dataset_key': 'CLASSIFIER_DATA_DIR',
}


def main():
    # Load environment config and hyperparams
    environment_config, hyperparams = load_training_args(default_hyperparams=DEFAULT_HYPERPARAMS)

    vit_model_name = hyperparams['vit_model_name']
    dataset_key = hyperparams['dataset_key']
    print(f'Using dataset: {dataset_key} -> {environment_config[dataset_key]}')

    # Create ViT feature extractor for the given ViT model
    vit_feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(vit_model_name)

    train_dataset, test_dataset, n_classes, class_names, random_sampler = data_loading.load_dataset(
        dataset_root_path=environment_config[dataset_key],
        prepare_examples_fn=lambda examples: prepare_examples(examples, vit_feature_extractor),
        remove_non_label_cols=True,
        cols_to_keep=['bbox_features']
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'],
                                               sampler=random_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    print(f'Creating model...')
    print(f'Using ViT model: {vit_model_name}')
    print(f'Training for {hyperparams["epochs"]} epochs')
    print(f'Using optimizer: {hyperparams["optimizer"]}')
    print(f'Using learning rate: {hyperparams["learning_rate"]}')
    print(f'Using warmup steps: {hyperparams["warmup_steps"]}')

    total_steps = None if hyperparams['warmup_steps'] is None else len(train_loader) * hyperparams['epochs']

    # Create model
    model = ViTModelLightning(
        n_classes=n_classes,
        model_name=vit_model_name,
        learning_rate=hyperparams['learning_rate'],
        optimizer_fn_name=hyperparams['optimizer'],
        class_names=class_names,
        warmup_steps=hyperparams['warmup_steps'],
        total_steps=total_steps,
    )

    # Create trainer
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
