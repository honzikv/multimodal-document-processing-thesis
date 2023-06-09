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
     - All experiments were run with CUDA 11.6
     - `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`
   - Build and install Detectron2 via GitHub
     - `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
     - Refer to the official troubleshooting guide if you encounter any compilation issues: https://detectron2.readthedocs.io/en/latest/tutorials/install.html#common-installation-issues
   - Install our modified version of LayoutLMv3 which does not have conflicts with the `transformers` library - this is located in
     the `layoutlmv3` directory. Navigate to the directory and install it via: `pip install -e .`
   - Install rest of the requirements: `pip install -r requirements.txt`

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
