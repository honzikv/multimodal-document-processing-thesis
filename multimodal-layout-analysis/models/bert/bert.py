import torch

from typing import List, Optional
from transformers import AutoModelForSequenceClassification, AutoModel

from models.base_lightning_model import MultimodalLightningModuleForClassification

DEFAULT_BERT_MODEL_NAME = 'bert-base-german-cased'


class BertLightning(MultimodalLightningModuleForClassification):
    """
    BERT model for text classification.
    """

    def __init__(self,
                 n_classes: int,
                 learning_rate: float = 5e-5,
                 optimizer_fn_name=None,
                 class_names: List[str] = None,
                 warmup_steps: Optional[int] = None,
                 total_steps: Optional[int] = None,
                 model_name: str = DEFAULT_BERT_MODEL_NAME):
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            optimizer_fn_name=optimizer_fn_name,
            class_names=class_names,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self._bert_model_name = model_name
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=n_classes,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def get_batch_outputs(self, batch):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        out = self(input_ids, attention_mask, labels=labels)
        pred = out.logits
        loss = out.loss
        pred = torch.argmax(pred, dim=1)

        return labels, loss, pred

    def perform_inference_on_batch(self, batch, labels=None):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        pred = self(input_ids, attention_mask, labels).logits
        pred = torch.argmax(pred, dim=1)

        return pred


class BertFeatureExtractor(torch.nn.Module):
    """
    Feature extractor for BERT - extract text modality for the multimodal classifier.
    """

    def __init__(self,
                 model_name: str,
                 max_seq_len: int,
                 output_features_size_half: int,
                 output_activation,
                 dropout=.3):
        """
        Creates a new instance of the TextModel
        Args:
            model_name: Transformer model name
            max_seq_len: The maximum sequence length
            output_features_size_half: The size of the output features
            output_activation: The activation function to use for the output layer
            dropout: The dropout rate
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self._output_features_size = output_features_size_half

        # Network architecture
        self.bert_model = AutoModel.from_pretrained(model_name)

        # To extract features and reduce dimensionality we use a BiLSTM which takes the output of the transformer model
        self.bilstm = torch.nn.LSTM(
            input_size=self.bert_model.config.hidden_size,
            hidden_size=output_features_size_half,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.output_activation = output_activation

    @property
    def output_size(self):
        return self._output_features_size * 2

    def forward(self, input_ids, attention_mask):
        # We only care about the last hidden state of the transformer
        out = self.bert_model(input_ids, attention_mask).last_hidden_state
        out, _ = self.bilstm(out)
        out = self.dropout(out[:, -1, :])  # Take the last output of the BiLSTM
        out = self.output_activation(out)
        return out
