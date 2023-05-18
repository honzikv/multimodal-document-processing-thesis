# LayoutLMv3 instance segmentation

This directory contains code to train the LayoutLMv3 instance segmentation model on
our dataset.

## Downloading the model

Before training, it is necessary to download the model.

The pre-training-only variant can be downloaded here: https://github.com/microsoft/unilm/tree/master/layoutlmv3#pre-trained-models

The PubLayNet variant can be downloaded here: https://huggingface.co/HYPJUDY/layoutlmv3-base-finetuned-publaynet

## Training

It is necessary to configure the model `model_config.yaml` file. To do so,
modify following keys to point to the correct locations:

- `MODEL.WEIGHTS` - path to the pre-trained LayoutLMv3 model with either .pth or .bin extension
- `HEIMATKUNDE_DATA_DIR_TRAIN` - path to the train dataset directory
- `HEIMATKUNDE_DATA_DIR_TEST` - path to the test dataset directory

To train the model, run the `train.py` script as follows:

```zsh
python train.py --config-file model_config.yaml \
                --no-wandb
```

## Inference

To run inference on the trained model, run the `inference.py` script as follows:

```zsh
python inference.py --image-path /path/to/image/or/image/folder \
                    --config-path model_config.yaml \
                    --output-path /path/to/segmentation/outputs
```
