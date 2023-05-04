from typing import List, Dict, Any

from models.preprocessing import ClassifierPreprocessor


def prepare_examples(examples, bert_tokenizer, vit_feature_extractor):
    images = examples['image']
    words = [' '.join(sequence) for sequence in examples['tokens']]
    labels = examples['label']

    bert_encoding = bert_tokenizer(words, padding='max_length', truncation=True, return_tensors='pt')
    vit_encoding = vit_feature_extractor(images, return_tensors='pt')

    # Combine the two encodings and return
    return {
        'input_ids': bert_encoding['input_ids'],
        'attention_mask': bert_encoding['attention_mask'],
        'pixel_values': vit_encoding['pixel_values'],
        'label': labels,
        'bbox_features': examples['bbox_features'],
    }


class ViTBertFusionClassifierPreprocessor(ClassifierPreprocessor):

    def __init__(self, bert_tokenizer, vit_feature_extractor):
        self._bert_tokenizer = bert_tokenizer
        self._vit_feature_extractor = vit_feature_extractor

    def preprocess(self, tokens: List[str], bboxes: List[tuple], image) -> List[Dict[str, Any]]:
        return self._bert_tokenizer(' '.join(tokens), padding='max_length', truncation=True,
                                    return_tensors='pt') | self._vit_feature_extractor(image, return_tensors='pt')
