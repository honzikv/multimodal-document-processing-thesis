from typing import List

import torch.nn

from models.base_lightning_model import MultimodalLightningModuleForClassification, ACTIVATIONS
from models.vit.vit import ViTFeatureExtractor
from models.bert.bert import BertFeatureExtractor


class BboxFeatureExtractor(torch.nn.Module):

    def __init__(self, activation, hidden_size=64, output_size=16, input_size=5):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.activation = activation
        self._output_features_size = output_size

    def forward(self, bbox_features):
        out = self.input_layer(bbox_features)
        out = self.activation(out)
        out = self.output_layer(out)
        return self.activation(out)

    @property
    def output_size(self):
        return self._output_features_size


class ViTBertLightning(MultimodalLightningModuleForClassification):

    def __init__(self,
                 vit_model_name: str,
                 bert_model_name: str,
                 n_classes: int,
                 learning_rate: float = 5e-5,
                 warmup_steps: int = None,
                 total_steps: int = None,
                 text_features_size_half=128,
                 image_features_size_half=128,
                 text_features_dropout=.3,
                 image_features_dropout=.3,
                 image_output_activation_name='relu',
                 text_output_activation_name='relu',
                 bbox_features_activation_name='relu',
                 fusion_activation_name='relu',
                 use_bbox_features=False,
                 max_seq_len=512,
                 optimizer_fn_name=None,
                 class_names: List[str] = None):
        """
        Creates a new instance of the ViT + BERT multimodal model
        Args:
            vit_model_name: The name of the ViT model to use
            bert_model_name: The name of the BERT model to use
            n_classes: The number of classes
            learning_rate: Learning rate of the model
            warmup_steps: Number of warmup steps
            total_steps: Total number of steps
            text_features_size_half: The size of the text features // 2
            image_features_size_half: The size of the image features // 2

        """
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            optimizer_fn_name=optimizer_fn_name,
            class_names=class_names,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.hparams['bert_model_name'] = bert_model_name
        self.hparams['vit_model_name'] = vit_model_name
        self.hparams['use_bbox_features'] = use_bbox_features
        self.hparams['image_features_size'] = image_features_size_half
        self.hparams['text_features_size'] = text_features_size_half
        self.hparams['image_features_dropout'] = image_features_dropout
        self.hparams['text_features_dropout'] = text_features_dropout
        self.hparams['image_output_activation_name'] = image_output_activation_name
        self.hparams['text_output_activation_name'] = text_output_activation_name
        self.hparams['bbox_features_activation_name'] = bbox_features_activation_name
        self.hparams['fusion_activation_name'] = fusion_activation_name
        self.save_hyperparameters()

        self._use_bbox_features = use_bbox_features
        self.loss = torch.nn.NLLLoss()

        self.text_model = BertFeatureExtractor(
            model_name=bert_model_name,
            max_seq_len=max_seq_len,
            output_features_size_half=text_features_size_half,
            output_activation=ACTIVATIONS[text_output_activation_name](),
            dropout=text_features_dropout,
        )

        self.image_model = ViTFeatureExtractor(
            model_name=vit_model_name,
            output_features_size_half=image_features_size_half,
            output_activation=ACTIVATIONS[image_output_activation_name](),
            dropout=image_features_dropout,
        )

        if self._use_bbox_features:
            self.bbox_model = BboxFeatureExtractor(activation=ACTIVATIONS[bbox_features_activation_name]())
            fusion_in_features = self.text_model.output_size + self.image_model.output_size + \
                                 self.bbox_model.output_size
        else:
            fusion_in_features = self.text_model.output_size + self.image_model.output_size

        self.fusion_input = torch.nn.Linear(
            in_features=fusion_in_features,
            out_features=fusion_in_features,
        )
        self.fusion_activation = ACTIVATIONS[fusion_activation_name]()
        self.fusion_output = torch.nn.Linear(
            in_features=fusion_in_features,
            out_features=n_classes,
        )

    def forward(self, pixel_values, input_ids, attention_mask, bbox_features=None):
        image_features = self.image_model(pixel_values)
        text_features = self.text_model(input_ids, attention_mask)

        if self._use_bbox_features:
            bbox_features = self.bbox_model(bbox_features)
            out = torch.cat([image_features, text_features, bbox_features], dim=1)
        else:
            out = torch.cat([image_features, text_features], dim=1)

        out = self.fusion_input(out)
        out = self.fusion_activation(out)
        out = self.fusion_output(out)
        out = torch.log_softmax(out, dim=1)
        return out

    def perform_inference_on_batch(self, batch):
        pixel_values, input_ids, attention_mask = batch['pixel_values'], batch['input_ids'], batch['attention_mask']
        bbox_features = batch['bbox_features'] if self._use_bbox_features else None

        pred = self(pixel_values, input_ids, attention_mask, bbox_features)
        pred = torch.argmax(pred, dim=1)

        return pred

    def get_batch_outputs(self, batch):
        images, input_ids, attention_mask, labels = batch['pixel_values'], batch['input_ids'], \
            batch['attention_mask'], batch['label']
        bbox_features = batch['bbox_features'] if self._use_bbox_features else None

        pred = self(images, input_ids, attention_mask, bbox_features)
        loss = self.loss(pred, labels)
        pred = torch.argmax(pred, dim=1)

        return labels, loss, pred
