# Mask R-CNN instance segmentation

This directory contains code to train the Mask R-CNN instance segmentation model on
our dataset.

## Training

To train the model, run the `train.py` script as follows:

```zsh
python train.py --train-json /path/to/dataset/train.json \
                --test-json /path/to/dataset/test.json \
                --config-yaml mask_rcnn_config.yaml \
                --no-wandb
```

The config `.yaml` file contains simple configuration parameters for the model. The default
configuration is the same as one used in the thesis.

During the training, the model is automatically evaluated on the test set after each epoch. Alternatively,
the inference can be run via the `inference.py` script like so:

```zsh
python inference.py --image-path /path/to/image/or/image/folder \
                    --config-path /path/to/config/file \
                    --output-path /path/to/segmentation/outputs
```