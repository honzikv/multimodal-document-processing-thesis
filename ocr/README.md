# OCR training and evaluation for our dataset

The training and evaluation datasets are located in the `train` and `test` folder.

## Training

Training is performed via the `tesstrain` tool from the `tesseract` repository: https://github.com/tesseract-ocr/tesstrain

The tesstrain tool must be setup first. To do so, follow the instructions in the `tesstrain` repository.
Subsequently, the training can be performed by modifying the training.sh script, where following variables must be provided:

- `TESSTRAIN_FOLDER` - Path to the tesstrain folder
- `TESSERACT_FOLDER` - Path to the tesseract folder
- `MODEL_NAME` - Name of the model to be trained
- `LANG` - Language of the model to be trained
- `ITERATIONS` - Number of iterations to train the model for, e.g. 50k


## Evaluation

To evaluate the `fraktur_custom` model on our dataset, run the following command:

```zsh
python evaluate_ocr_model.py --data-dir /path/to/dataset/ocr/test --model-name fraktur_custom
```

## Trained model

The `fraktur_custom.traineddata` is the best model we trained on the Heimatkunde dataset. To use it
simply copy it in your `tessdata` directory in the Tesseract installation folder.