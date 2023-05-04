import os
import argparse
import logging
import random
import string

import pytesseract as tess
import cv2

from typing import List, Tuple
from dataclasses import dataclass
from evaluate import load

CONFIG = '--oem 1 --psm 6'  # PSM 6 assumes uniform block of text, oem 1 enables LSTM

# Logging configuration
logging.basicConfig(level=logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')
__logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TessModel:
    """
    Simple Tesseract configuration data object
    """
    name: str
    config: str  # passed to tesseract as string, e.g. '--oem 1 --psm 6'
    character_noise: float = 0.0  # random noise introduced to the characters, used as an experiment in the thesis
    sub_special_chars: bool = False  # substitute _ and # with s and -


def load_page_folder(page_path: os.path, image_extensions: List[str]) -> Tuple[List[str], List[str]]:
    images, texts = [], []

    for image_name in [file for file in os.listdir(page_path) if os.path.splitext(file)[1] in image_extensions]:
        # Find .txt file with the same name
        text_name = image_name.split('.')[0] + '.txt'
        text_path = os.path.join(page_path, text_name)

        # If both files exist, add them to the lists
        if os.path.exists(text_path):
            images.append(os.path.join(page_path, image_name))
            texts.append(text_path)
            continue

        __logger.info(
            f'No .gt.txt file found for {image_name}, the file will be skipped...')

    return images, texts


def load_dataset(dataset_root: os.path, image_extensions=None) -> Tuple[List[str], List[str]]:
    """
    Load all images and corresponding ground truth text files from a dataset directory.
    The directory must be structured in a way where each page is in a separate folder.
    Each page folder must contain the images and the corresponding ground truth text files with the same name.
    The text files must have the extension .txt - e.g. 1.jpg and 1.txt represent the same image and text.
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.png']

    images, texts = [], []  # images are feature, texts are labels

    # Load for each page folder
    for page_folder in os.listdir(dataset_root):
        page_path = os.path.join(dataset_root, page_folder)
        if os.path.isdir(page_path):
            images_page, texts_page = load_page_folder(page_path, image_extensions)
            images.extend(images_page)
            texts.extend(texts_page)

    return images, texts


def load_text(text_file: str) -> str:
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess(text, sub_special_chars=False):
    """
    Preprocess text for evaluation - ſ and s are the same
    """
    if sub_special_chars:
        text = text.replace('_', 's')
        text = text.replace('#', '-')

    text = text.replace('ſ', 's')  # Replace long s with normal s
    text = text.strip()  # Remove leading and trailing whitespace
    return text


def maybe_add_noise(original_token: str, noise_prob: float):
    result = original_token
    for i in range(len(original_token)):
        if result[i] == ' ':
            continue

        if random.random() > noise_prob:
            continue

        result = result[:i] + random.choice(string.ascii_letters) + result[i + 1:]

    return result


def make_pred(tess_model: TessModel, image_path: str) -> str:
    """
    Make a prediction with a model on an image
    """
    text = preprocess(tess.image_to_string(cv2.imread(image_path), lang=tess_model.name, config=tess_model.config),
                      sub_special_chars=tess_model.sub_special_chars)
    return maybe_add_noise(text, tess_model.character_noise)


def get_word_count(text: str) -> int:
    """
    Get the number of words in a text
    """
    return len(text.split(' '))


def get_total_word_count(texts: List[str]) -> int:
    """
    Get the total number of words in a list of texts
    """
    return sum([get_word_count(text) for text in texts])


def evaluate_model(image_files: List[str], texts, tess_model: TessModel):
    """
    Evaluate a model on a dataset
    """
    __logger.info(f'Evaluating model {tess_model.name}...')

    preds = []
    __logger.info(f'Generating predictions...')
    for i, image_path in enumerate(image_files):
        pred = make_pred(tess_model, image_path)
        preds.append(pred)

        if i % 50 == 49:
            __logger.info(f'Generated prediction for {i + 1}/{len(image_files)} testing samples...')

    __logger.info(f'Generated predictions for {len(image_files)}/{len(image_files)} testing samples...')
    __logger.info(f'Calculating CER and WER...')

    # Calculate CER and WER
    cer_eval, wer_eval = load('cer'), load('wer')
    cer, wer = cer_eval.compute(references=texts, predictions=preds), wer_eval.compute(references=texts,
                                                                                       predictions=preds)
    __logger.info(f'Finished evaluating model {tess_model.name}...')
    __logger.info(f'CER: {cer * 100}%')
    __logger.info(f'WER: {wer * 100}%')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-dir', type=str,
                           required=True, help='Path to evaluation directory')
    argparser.add_argument('--models', nargs='+',
                           required=True, help='Models (languages in tessdata folder) to benchmark')
    argparser.add_argument('--random-character-noise', type=float, default=0.0,
                           required=False, help='Probability of a character being replaced with a random character. '
                                                'Note that this is applied to each model')
    argparser.add_argument('--sub-special', required=False, action='store_true',
                           default=False, help='Substitute special "_" with "s" symbol and "#" with "-"')

    args = argparser.parse_args()

    image_files, text_files = load_dataset(args.data_dir)
    texts = [load_text(text_file) for text_file in text_files]

    # Compute number of words
    total_words = get_total_word_count(texts)
    print(f'Total number of words: {total_words}')

    if not 0 <= args.random_character_noise <= 1:
        print(f'Invalid random character noise probability: {args.random_character_noise}')
        exit(1)

    for model in args.models:
        tess_model = TessModel(model, CONFIG, args.random_character_noise, args.sub_special)
        evaluate_model(image_files, texts, tess_model)


if __name__ == '__main__':
    main()
