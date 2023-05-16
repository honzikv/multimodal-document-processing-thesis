import argparse
import torch.utils.data
import pytorch_lightning as pl

from pathlib import Path

from pytorch_lightning.loggers import WandbLogger
from transformers import AutoProcessor, AutoTokenizer, ViTFeatureExtractor
from dataset import data_loading
from models.fusion.vitbert import ViTBertLightning
from models.layoutlmv3.layoutlmv3 import LayoutLMv3Lightning
from models.layoutlmv3.layoutlmv3_preprocessing import prepare_examples as prepare_layoutlmv3_examples
from models.fusion.vitbert_preprocessing import prepare_examples as prepare_vibert_examples

DEFAULT_DATASET_PATH = Path('/mnt/c/ml-data/heimatkunde/classifier/')
VIT_MODEL = 'google/vit-base-patch16-224-in21k'
BERT_MODEL = 'bert-base-german-cased'

MODEL_NAMES = ['layoutlmv3', 'vibert']


def main(args):
    if args.model_name == 'layoutlmv3':
        processor = AutoProcessor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
        prepare_examples = lambda examples: prepare_layoutlmv3_examples(examples, processor)
        model_class_name = LayoutLMv3Lightning
    elif args.model_name == 'vibert':
        vit_feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL)
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        prepare_examples = lambda examples: prepare_vibert_examples(examples, bert_tokenizer, vit_feature_extractor)
        model_class_name = ViTBertLightning
    else:
        raise ValueError(f'Invalid model name: {args.model_name}')

    _, test_dataset, n_classes, _, _ = data_loading.load_dataset(
        args.dataset_root_path,
        prepare_examples,
        remove_non_label_cols=True,
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    model = model_class_name.load_from_checkpoint(
        checkpoint_path=args.model_checkpoint_path,
        n_classes=n_classes,
    )

    wandb_logger = WandbLogger(project='multimodal-classification-eval')
    wandb_logger.watch(model, log='all', log_freq=100)
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0
    )

    trainer.test(model, dataloaders=test_dataloader)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset-root-path', type=Path, required=False,
                           default=DEFAULT_DATASET_PATH)
    argparser.add_argument('--model-name', type=str, required=True)
    argparser.add_argument('--model-checkpoint-path', type=Path, required=True)

    args = argparser.parse_args()

    if args.model_name not in MODEL_NAMES:
        print(f'Invalid model name: {args.model_name}')
        print(f'Valid model names: {MODEL_NAMES}')
        exit(1)

    main(args)
