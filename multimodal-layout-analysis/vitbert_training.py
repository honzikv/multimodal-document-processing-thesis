import torch.utils.data
import transformers

from models.fusion.vitbert import ViTBertLightning
from models.fusion.vitbert_preprocessing import prepare_examples
from dataset import data_loading
from models.bert.bert import DEFAULT_BERT_MODEL_NAME
from models.vit.vit import DEFAULT_VIT_MODEL_NAME
from training_base import load_training_args, create_trainer

MODEL_NAME_CHECKPOINT = 'VITBERT'
WANDB_PROJECT = 'heimatkunde-vitbert-classification'
DEFAULT_HYPERPARAMS = {
    'learning_rate': 3e-5,
    'epochs': 10,
    'batch_size': 4,
    'optimizer': 'adamw',
    'bert_model_name': DEFAULT_BERT_MODEL_NAME,
    'vit_model_name': DEFAULT_VIT_MODEL_NAME,  # or "microsoft/swin-tiny-patch4-window7-224"
    'use_bbox_features': False,
    'warmup_steps': 1500,
    'precision': 16,
    'dataset_key': 'CLASSIFIER_DATA_DIR',
    'image_output_activation_name': 'relu',
    'text_output_activation_name': 'relu',
    'text_features_dropout': .3,
    'image_features_dropout': .3,
}


def main():
    # Load environment config and hyperparams
    environment_config, hyperparams = load_training_args(default_hyperparams=DEFAULT_HYPERPARAMS)

    bert_model_name = hyperparams['bert_model_name']
    vit_model_name = hyperparams['vit_model_name']
    dataset_key = hyperparams['dataset_key']
    print(f'Using dataset: {dataset_key} -> {environment_config[dataset_key]}')

    # Create BERT tokenizer for the given BERT model
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model_name)

    # Create ViT feature extractor for the given ViT model
    vit_feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(vit_model_name)

    train_dataset, test_dataset, n_classes, class_names, random_sampler = data_loading.load_dataset(
        dataset_root_path=environment_config[dataset_key],
        prepare_examples_fn=lambda examples: prepare_examples(examples, bert_tokenizer, vit_feature_extractor),
        remove_non_label_cols=True,
        cols_to_keep=['bbox_features']
    )

    batch_size = hyperparams['batch_size']
    print(f'Creating data loaders with batch size: {batch_size}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Creating model...')
    print(f'Using BERT model: {bert_model_name}')
    print(f'Using ViT model: {vit_model_name}')
    print(f'Training for {hyperparams["epochs"]} epochs')
    print(f'Using optimizer: {hyperparams["optimizer"]}')
    print(f'Using learning rate: {hyperparams["learning_rate"]}')
    print(f'Using warmup steps: {hyperparams["warmup_steps"]}')

    total_steps = None if hyperparams['warmup_steps'] is None else len(train_loader) * hyperparams['epochs']

    # Create model
    model = ViTBertLightning(
        bert_model_name=bert_model_name,
        vit_model_name=vit_model_name,
        n_classes=n_classes,
        learning_rate=hyperparams['learning_rate'],
        optimizer_fn_name=hyperparams['optimizer'],
        class_names=class_names,
        warmup_steps=hyperparams['warmup_steps'],
        total_steps=total_steps,
        use_bbox_features=hyperparams['use_bbox_features'],
        image_output_activation_name=hyperparams['image_output_activation_name'],
        text_output_activation_name=hyperparams['text_output_activation_name'],
        text_features_dropout=hyperparams['text_features_dropout'],
        image_features_dropout=hyperparams['image_features_dropout'],
        text_features_size_half=hyperparams['text_features_size'],
        image_features_size_half=hyperparams['image_features_size'],
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
