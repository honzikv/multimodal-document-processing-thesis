# YOLOv8 instance segmentation

This directory contains code to train the YOLOv8 instance segmentation model on
our dataset.

## Training

To train the model, run the `train.py` script as follows:

```zsh
python train.py --dataset-yaml /path/to/dataset/yolo/dataset.yaml \
                --config-yaml 640p.yaml \
                --no-wandb
```

Note that training either model on GPU requires a lot of VRAM and 8GB might not be
sufficient for the 1280p variant.