# Multi-modal Document Processing
This repository contains all relevant code to reproduce 
experiments performed in my masters thesis.

## The Heimatkunde dataset
The Heimatkunde dataset is hosted on a different GitHub repository, which can be accessed here: https://github.com/honzikv/heimatkunde-dataset


## Environment

The experiments are reproducible on Linux and macOS. Windows is also compatible via
the Windows Subsystem for Linux (WSL). The following instructions assume that you are
using a Linux distribution, however the steps are similar for macOS and WSL.

The following is required to run the experiments:

- Anaconda, miniconda, or similar tool that can easily setup new Python 3.9 environment
- 8+ GB of RAM and ideally a discrete GPU with 8+ GB of VRAM for faster training
- Tesseract OCR installed and registered in the PATH

### Installation

1. Download the dataset, e.g. via `git clone https://github.com/honzikv/heimatkunde-dataset`
2. Install Tesseract (https://tesseract-ocr.github.io/tessdoc/Installation.html) either via your package manager (e.g. `sudo apt install tesseract-ocr
`) or build it from source.
    - Tesseract needs to be registered in the PATH for the scripts to be runnable
    - You need to copy the pre-trained fraktur model `fraktur_custom.tessdata` from the `ocr` directory to the tessdata directory. This directory should be in the install path
    of Tesseract.
3. Create new conda environment and install all dependencies:
   - `conda create --name multimodal python=3.9`
   - `conda activate multimodal`
   - Install PyTorch - navigate to https://pytorch.org/get-started/locally/ for details
     - All experiments were run with CUDA 11.1
     - `conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia`
   - Build and install Detectron2 via GitHub
     - `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
   - Install our modified version of LayoutLMv3 which does not have conflicts with the `transformers` library - this is located in
     the `layoutlmv3` directory. Navigate to the directory and install it via: `pip install -e .`
   - Install rest of the requirements: `pip install -r requirements.txt`

## Creating document layout analysis dataset from scratch (not necessary)

To create the DLA dataset from scratch two scripts are needed. The first one creates COCO and YOLO annotations from
the dataset ZIP file, while the other converts the COCO annotations to ones digestible by classifier.

### Creating COCO and YOLO annotations 

To create the COCO and YOLO annotations from the dataset ZIP file, 
navigate to the `preprocessing` directory and run the `cvat_preprocessing.py` script
as follows:

```zsh
python cvat_preprocessing.py --zip-path /path/to/zip/file.zip \
                             --output-path /path/to/output/directory \
                             --create-yolo \
                             --max-image-width 1280 
```

This will generate the dataset in the output directory. The structure of the dataset is as follows:

```
images
yolo
    - test
        - images
        - labels
    - train
        - images
        - labels
    - dataset.yaml
train.json
test.json
```

Where `train.json` and `test.json` are the COCO annotations for the train and test set respectively, `images` contains
all images from the dataset.

In case of YOLO, the structure follows latest YOLOv8 format. The `dataset.yaml` file contains the class names and
the **relative** paths to the train and test images. If the paths do not work for any reason, 
you can manually change them to absolute paths.

### Creating classifier annotations

To create the classifier annotations, navigate to the `multimodal-layout-analysis` directory and run the
`build_classification_dataset.py` script as follows:

```zsh
python build_classification_dataset.py --train-json-path /path/to/dataset/train.json \
                                       --test-json-path /path/to/dataset/test.json \
                                       --train-output-path /path/to/dataset/classifier/train/train.json \
                                       --test-output-path /path/to/dataset/classifier/test/test.json
```

This will create the classifier annotations in the output directory. Note that the `train-output-path` and 
`test-output-path` should have the same parent directory. The structure of the classifier dataset should look
as follows:

```
train
    - images
    - classes.json
    - train.json
test
    - images
    - classes.json
    - test.json
```

`train.json` and `test.json` are simple JSON arrays that contain annotations for each image. `classes.json` contains
information about the classes in the dataset and should be identical for both splits.

## Experiments

Each experiment is located in its own directory with a dedicated README file.
The README file contains instructions on how to run the experiment and the structure
of the directory. Note that **scripts for any experiment must be run from its corresponding directory**, otherwise
the paths will not be resolved correctly.

### Wandb Logging ⚠️⚠️⚠️

Note that all our experiments use Wandb logging which needs to be turned off if you do not have a Wandb account. To do
so, simply run the corresponding training script with the `--no-wandb` flag.

### Structure

The experiments are located in following directories:

- `multimodal-layout-analysis` - This folder contains code to train each multi-modal classifier and
 evaluate it alongside the instance segmentation model
- `layoutlmv3-segmentation` - This folder contains code to train LayoutLMv3-based model for instance segmentation
- `ocr` - This folder contains code to train the OCR model
- `yolo-segmentation` - This folder contains code to train the YOLOv8 for instance segmentation
- `mask-rcnn-segmentation` - This folder contains code to train the Mask R-CNN for instance segmentation
