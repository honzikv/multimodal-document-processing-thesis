# Multi-modal layout analysis

This repository contains the code for training several variants of a multi-modal classifier
as well as evaluation of the multi-modal layout analysis.

## Classifier training

Before you start the training, it is necessary to configure `environment.yaml` file to
point to the directory of your classifier annotations. E.g.:

```yaml
CLASSIFIER_DATA_DIR: '/ml-data/heimatkunde-dataset/classifier/'
```

There are several scripts for training:

- `bert_training.py` - training script for BERT-based classifier
- `vit_training.py` - training script for ViT and SwinV2-based classifiers
- `layoutlmv3_training.py` - training script for LayoutLMv3 classifier
- `vitbert_training.py` - training script for BERT + ViT/SwinV2-based classifier

Each script uses the same set of arguments:

- `--environment-config-path` - path to the environment.yaml file which contains environment settings
- `--hyperparams-path` - path to the model's hyperparameters. Hyperparams are stored in the `hyperparams` directory
- `--force-seed` - forces seed, overriding one in the environment.yaml file
- `--no-wandb` - disables wandb logging

The training is performed by simply running the script, for example:

```zsh
python vitbert_training.py --environment-config-path environment.yaml \
                           --hyperparams-path hyperparams/vitbert.yaml \
                           --force-seed 42
```

## Evaluation

Evaluation is performed by running the `eval_multimodal.py` script.
The script uses the following arguments:

- `seg-model-name` - name of the segmentation model, must be one of `layoutlmv3`, `mask_rcnn`, or `yolo`
- `seg-model-path` - path to the segmentation model's configuration (LayoutLMv3 / mask_rcnn) or weights (YOLO)
- `class-model-name` - name of the classification model, must be one of `layoutlmv3` or `vitbert`
- `class-model-weights` - path to the `.ckpt` file with the classification model's weights
- `dataset-path` - path to the root of the dataset to evaluate on, e.g. `/ml-data/mydataset`
- `no-classifier` - if specified no classifier is used (optional)

Note that the evaluation is relatively long due to OCR and can take several tens of minutes.


## Folder / File Structure

In addition to the training and eval scripts there are following directories:

- `config` - configuration files for running the models / hyperparameter templates
- `dataset` - code relevant to loading and preprocessing the dataset
  - `huggingface/heimatkunde_classification_dataset.py` - script to load the data via `datasets`
- `ditod` - code for LayoutLMv3 inference
- `eval` - code for evaluation
- `models` - implementation of the models

### Helper scripts

There are several helper scripts that are not necessary for training / evaluation:

- `classification_dataset_info.py` - prints information about the dataset
- `detectron_find_best_checkpoint.py` - finds the best checkpoint for Detectron2 models
- `generate_hyperparams.py` - generates hyperparams from given yaml template - e.g. `config/hyperparam_templates/bert.yaml`
- `generate_wandb_summary.py` - generates summary for each wandb run - best F1 + other metrics
