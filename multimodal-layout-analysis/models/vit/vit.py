import torch

from typing import List, Optional
from transformers import AutoModel, ViTForImageClassification, AutoModelForSequenceClassification, \
    AutoModelForImageClassification

from models.base_lightning_model import MultimodalLightningModuleForClassification

DEFAULT_VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'


class ViTModelLightning(MultimodalLightningModuleForClassification):
    """
    Vision Transformer model for classification
    """

    def __init__(self,
                 n_classes: int,
                 model_name: str = DEFAULT_VIT_MODEL_NAME,
                 learning_rate: float = 5e-5,
                 optimizer_fn_name=None,
                 class_names: List[str] = None,
                 warmup_steps: Optional[int] = None,
                 total_steps: Optional[int] = None):
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            optimizer_fn_name=optimizer_fn_name,
            class_names=class_names,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values, labels=None):
        return self.model(
            pixel_values=pixel_values,
            labels=labels,
        )

    def get_batch_outputs(self, batch):
        pixel_values, labels = batch['pixel_values'], batch['label']
        out = self(pixel_values, labels=labels)
        pred = out.logits
        loss = out.loss
        pred = torch.argmax(pred, dim=1)

        return labels, loss, pred

    def perform_inference_on_batch(self, batch, labels=None):
        pixel_values = batch['pixel_values']
        pred = self(pixel_values, labels).logits
        pred = torch.argmax(pred, dim=1)

        if labels is not None:
            return pred, labels

        return pred


class ViTFeatureExtractor(torch.nn.Module):

    def __init__(self,
                 model_name: str = DEFAULT_VIT_MODEL_NAME,
                 output_features_size_half: int = 128,
                 output_activation=torch.nn.ReLU(),
                 dropout=.3):
        """
        Creates a new instance of the ViT feature extractor

        Args:
            output_features_size_half: The size of the output features. This is the size of the hidden state of the
                BiLSTM - i.e. the real output size is twice this value
            model_name: name of the model to use. This should be a Vision Transformer
            output_activation: The activation function to use on the output
            dropout: The dropout to use
        """
        super().__init__()
        self.vit = AutoModel.from_pretrained(model_name)
        self.bilstm = torch.nn.LSTM(
            input_size=self.vit.config.hidden_size,
            hidden_size=output_features_size_half,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.output_activation = output_activation
        self._output_features_size = output_features_size_half

    def forward(self, pixel_values):
        out = self.vit(pixel_values).last_hidden_state
        out, _ = self.bilstm(out)
        out = self.dropout(out[:, -1, :])  # Take the last output of the BiLSTM

        return self.output_activation(out)

    @property
    def output_size(self):
        return self._output_features_size * 2
