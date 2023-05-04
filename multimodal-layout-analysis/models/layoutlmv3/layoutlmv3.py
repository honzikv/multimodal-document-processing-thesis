from typing import List

import torch
import math

from transformers import LayoutLMv3ForSequenceClassification

from models.base_lightning_model import MultimodalLightningModuleForClassification


class LayoutLMv3Lightning(MultimodalLightningModuleForClassification):

    def __init__(self,
                 n_classes: int,
                 input_size=224,
                 learning_rate: float = 1e-5,
                 optimizer_fn_name=None,
                 class_names: List[str] = None,
                 warmup_steps=None,
                 total_steps=None):
        """
        Creates a new instance of the LayoutLMv3Lightning
        Args:
            n_classes: The number of classes to predict - i.e. size of the output layer
            input_size: The input size of the images. This value should be kept at 224 for the best results. It can be
                altered if you also change the model preprocessor but it will initialize the model with random weights
                for higher input sizes
            learning_rate: Learning rate for the optimizer
            optimizer_fn_name: The optimizer function to use
            warmup_steps: The number of warmup steps to use
            total_steps: Total number of steps the model will be trained for. This should be computed as:
                total_steps = len(train_dataloader) * num_train_epochs
        """
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            optimizer_fn_name=optimizer_fn_name,
            class_names=class_names,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            'microsoft/layoutlmv3-base',
            input_size=input_size,
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )

    def get_lr(self, step):
        if step < self.warmup_steps:
            lr = self.learning_rate * float(step) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr = self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr

    def forward(self, input_ids, attention_mask, bboxes, pixel_values, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bboxes,
            pixel_values=pixel_values,
            labels=labels,
        )

    def get_batch_outputs(self, batch):
        input_ids, attention_mask, pixel_values = batch['input_ids'], batch['attention_mask'], batch['pixel_values']
        labels, bboxes = batch['label'], batch['bbox']
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            bboxes=bboxes,
            labels=labels,
        )

        loss = outputs.loss
        pred = outputs.logits
        pred = torch.argmax(pred, dim=1)

        return labels, loss, pred

    def perform_inference_on_batch(self, batch):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            bboxes=batch['bbox'],
        )
        pred = outputs.logits
        pred = torch.argmax(pred, dim=1)
        return pred
