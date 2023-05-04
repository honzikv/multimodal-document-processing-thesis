# LayoutLMv3 instance segmentation

This directory contains code to train the LayoutLMv3 instance segmentation model on
our dataset.

## Training

Before training the model it is necessary to configure the model `.yaml` file. To do so,
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