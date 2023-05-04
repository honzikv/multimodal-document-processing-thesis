import torch
import wandb
import numpy as np
import pytorch_lightning as pl

from abc import abstractmethod
from typing import List, Optional
from torchmetrics import Accuracy, F1Score, Precision, Recall
from transformers import get_linear_schedule_with_warmup

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'mish': torch.nn.Mish,
    'silu': torch.nn.SiLU,
    'gelu': torch.nn.GELU,
}


class MultimodalLightningModuleForClassification(pl.LightningModule):
    """
    A base class for multimodal model variants.
    This contains all the boilerplate code for training and validation

    How to use this module:
        1. Create a new class that inherits from this class
        2. Implement the forward method
        3. Implement get_batch_outputs and perform_inference_on_batch methods

    Train and validation steps call get_batch_outputs and use the returned values to compute the loss and metrics. They
    also automatically log metrics such as loss, accuracy, precision, recall and f1 score.
    """

    def __init__(self,
                 n_classes: int,
                 learning_rate: float = 5e-5,
                 optimizer_fn_name=None,
                 class_names: List[str] = None,
                 warmup_steps: Optional[int] = None,
                 total_steps: Optional[int] = None):
        """
        Creates a new instance of the MultimodalLightningModule
        Args:
            n_classes: The number of classes to predict
            learning_rate: Learning rate for the optimizer
            optimizer_fn_name: Name of the optimizer to use
            class_names: The names of the classes - this is used for the confusion matrix
            warmup_steps: The number of warmup steps to use
            total_steps: Total number of steps the model will be trained for. This should be computed as:
                total_steps = len(train_dataloader) * num_train_epochs
        """
        super().__init__()

        # Hyperparams
        self._learning_rate = learning_rate
        self._n_classes = n_classes
        self._optimizer_fn = optimizer_fn_name

        # LR warmup
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

        self._class_names = class_names if class_names else [str(i) for i in range(n_classes)]
        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []

        # Train and val / test metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro')
        self.train_f1 = F1Score(task='multiclass', num_classes=n_classes, average='macro')
        self.train_precision = Precision(task='multiclass', num_classes=n_classes, average='macro')
        self.train_recall = Recall(task='multiclass', num_classes=n_classes, average='macro')

        self.val_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro')
        self.val_f1 = F1Score(task='multiclass', num_classes=n_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=n_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=n_classes, average='macro')

        # Save number of classes so that the trained model does not need to include it in when loading the model
        self.hparams['n_classes'] = n_classes
        self.hparams['optimizer_fn_name'] = optimizer_fn_name
        self.hparams['class_names'] = class_names
        self.hparams['warmup_steps'] = warmup_steps
        self.hparams['total_steps'] = total_steps
        self.save_hyperparameters()

    @abstractmethod
    def get_batch_outputs(self, batch):
        """
        Returns the labels, loss and predictions for a batch

        Args:
            batch: The batch to process - this should be an object containing features and labels
        Returns:
            labels: The labels for the batch - torch.Tensor
            loss: The loss for the batch - float
            preds: The predictions for the batch - torch.Tensor
        """
        pass

    @abstractmethod
    def perform_inference_on_batch(self, batch):
        """
        Returns the predictions for a batch

        Args:
            batch: The batch to process - this should be an object containing features
        Returns:
            preds: The predictions for the batch - torch.Tensor
        """
        pass

    def inference_on_batch(self, batch):
        """
        Returns the batch prediction as a tensor
        """
        was_training = self.training
        if was_training:
            self.eval()

        with torch.no_grad():
            pred = self.perform_inference_on_batch(batch)

        if was_training:
            self.train()

        return pred

    def compute_metrics(self, pred, labels, split):
        if split == 'train':
            self.train_accuracy(pred, labels)
            self.train_precision(pred, labels)
            self.train_recall(pred, labels)
            self.train_f1(pred, labels)
        elif split == 'val':
            self.val_accuracy(pred, labels)
            self.val_precision(pred, labels)
            self.val_recall(pred, labels)
            self.val_f1(pred, labels)
        else:
            # TODO maybe implement for test step
            raise ValueError(f'Invalid step: {split}')

    def log_metrics(self, loss, split):
        """
        Logs the metrics for the current step
        Args:
            loss: The loss for the current step
            split: The split for the current step - 'train' or 'val'. For test split use the 'val' value
        """
        metric_suffixes = ['loss', 'acc', 'f1', 'precision', 'recall']

        if split == 'train':
            metrics = [f'train_{suffix}' for suffix in metric_suffixes]
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)
            self.log(metrics[0], loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(metrics[1], self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.log(metrics[2], self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
            self.log(metrics[3], self.train_precision, on_step=True, on_epoch=True, prog_bar=True)
            self.log(metrics[4], self.train_recall, on_step=True, on_epoch=True, prog_bar=True)
        elif split == 'val':
            metrics = [f'val_{suffix}' for suffix in metric_suffixes]
            self.log(metrics[0], loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(metrics[1], self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.log(metrics[2], self.val_f1, on_step=True, on_epoch=True, prog_bar=True)
            self.log(metrics[3], self.val_precision, on_step=True, on_epoch=True, prog_bar=True)
            self.log(metrics[4], self.val_recall, on_step=True, on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f'Invalid step: {split}')

    def training_step(self, batch, batch_idx):
        labels, loss, preds = self.get_batch_outputs(batch)
        self.compute_metrics(preds, labels, 'train')
        self.log_metrics(loss, 'train')
        self.train_labels += labels.tolist()
        self.train_predictions += preds.tolist()

        return loss

    def validation_step(self, batch, batch_idx):
        labels, loss, pred = self.get_batch_outputs(batch)
        self.compute_metrics(pred, labels, 'val')
        self.log_metrics(loss, 'val')
        self.val_labels += labels.tolist()
        self.val_predictions += pred.tolist()

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        """
        Compute train confusion matrix and log it to wandb
        """
        if wandb.run is not None:
            self.logger.experiment.log({
                'train_confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.array(self.train_labels),
                    preds=np.array(self.train_predictions),
                    class_names=self._class_names,
                )})

        self.train_labels.clear()
        self.train_predictions.clear()

    def on_validation_epoch_end(self) -> None:
        """
        Compute validation confusion matrix and log it to wandb
        """
        if wandb.run is not None:
            self.logger.experiment.log({
                'val_confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.array(self.val_labels),
                    preds=np.array(self.val_predictions),
                    class_names=self._class_names,
                )})

        self.val_labels.clear()
        self.val_predictions.clear()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler (if learning rate warmup is enabled)
        """
        optimizer_fn = OPTIMIZERS.get(self._optimizer_fn, None)
        if optimizer_fn is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self._learning_rate)
        else:
            optimizer = optimizer_fn(self.parameters(), lr=self._learning_rate)

        if self._warmup_steps is not None:
            if self._total_steps is None or self._warmup_steps is None:
                raise ValueError('Must specify total_steps and warmup_steps if using warmup')

            if self._total_steps < self._warmup_steps:
                raise ValueError(
                    f'total_steps must be greater than warmup_steps. Got total_steps: {self._total_steps}, '
                    f'warmup_steps: {self._warmup_steps}'
                )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self._warmup_steps,
                num_training_steps=self._total_steps
            )

            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

        return optimizer
