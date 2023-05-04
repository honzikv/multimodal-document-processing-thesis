# OCR training and evaluation for our dataset

The training and evaluation datasets are located in the `train` and `test` folder.

To evaluate the `fraktur_custom` model on our dataset, run the following command:

```zsh
python evaluate_ocr_model.py --data-dir test --model-name fraktur_custom
```