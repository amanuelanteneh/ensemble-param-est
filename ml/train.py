import os
from enum import Enum
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from optuna import Trial, TrialPruned
import mlflow
import pymc_marketing.mlflow
import arviz as az

from ml.models import BayesianInferenceModel, DeepEnsemble
from ml.loss import MSLELoss
from ml.data import QuantumJumpDataset


class EarlyStopping:
    def __init__(self, patience: int, lower_is_better: bool = True) -> None:
        """
        Early stopping mechanism to prevent overfitting.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            lower_is_better: If True, lower values of the monitored metric are better.
                             Default is True, meaning that lower values of the metric indicate better performance.
        """
        self.patience = patience
        self.lower_is_better = lower_is_better
        self.best_score = float("inf") if lower_is_better else float("-inf")
        self.counter = 0

    def should_stop(self, current_score: float) -> bool:
        """
        Check if training should be stopped based on the current score.
        Args:
            current_score: The current score of the monitored metric.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """

        if (self.lower_is_better and current_score < self.best_score) or (
            not self.lower_is_better and current_score > self.best_score
        ):
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class BaseNetworkTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        batch_size: int,
        num_epochs: int,
        val_metric_type: Enum,
        loss_fn: Callable = MSLELoss,
        device: str = "cpu",
        patience: int = 10,
    ) -> None:
        """
        A trainer object to train a neural network such as the
        MLP or LSTM net in models.py

        Args:
            model: The neural network model to train.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay (L2 regularization) for the optimizer.
            beta1: Beta1 parameter for Adam optimizer.
            beta2: Beta2 parameter for Adam optimizer.
            batch_size: Batch size for training.
            num_epochs: Number of epochs to train the model.
            val_metric_type: Validation metric/loss function type to use for early stopping and hyperparameter tuning.
                             Not to be confused with the loss function used for training.
            loss_fn: Loss function to use for training.
            device: Device to use for training (e.g., "cpu" or "cuda").
            patience: Number of epochs with no improvement after which training will be stopped.
        """

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.build_optimizer()

        self.scaler = torch.amp.GradScaler()

        self.loss_fn = loss_fn()

        self.val_metric_name = val_metric_type.name
        val_metric_cls, lower_is_better = val_metric_type.value
        self.val_metric_fn = val_metric_cls()
        self.stopper = EarlyStopping(patience=patience, lower_is_better=lower_is_better)

    def build_optimizer(self) -> None:
        """
        Build an optimizer for the model.
        Args:
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            beta1: Beta1 parameter for Adam optimizer.
            beta2: Beta2 parameter for Adam optimizer.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )

    def training_step(self, input: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Perform a training step on the model. Uses Automatic Mixed Precision (AMP) for training.
        Args:
            input: Input tensor for the model.
            labels: Ground truth labels for the input.

        Returns:
            The loss value for the training step.
        """
        self.model.train()

        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device):
            output = self.model(input)
            loss = self.loss_fn(output, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # original backward process that doesn't use AMP
        # loss.backward()
        # self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validation_step(
        self, input: torch.Tensor, labels: torch.Tensor
    ) -> tuple[float, float]:
        """
        Perform a validation step on the model.
        Args:
            input: Input tensor for the model.
            labels: Ground truth labels for the input.
        NOTE: Both the training loss function and the validation metric are computed during validation
              and logged since they are not necessarily the same. This helps when one wants to plot learning curves
              as it is best to plot the same metric for both training and validation.
        """
        self.model.eval()
        output = self.model(input)
        val_loss = self.loss_fn(output, labels)
        val_metric = self.val_metric_fn(output, labels)

        return val_loss.item(), val_metric.item()

    def train(
        self,
        train_dataset: QuantumJumpDataset,
        val_dataset: QuantumJumpDataset,
        epochs_per_val: int,
        trial: Trial = None,
    ) -> dict[str, float]:
        """
        Train the model on the given training and validation datasets.

        Args:

        """
        self.model.train()
        self.model = self.model.to(self.device)

        # set to 0 since we can load all data into memory and we don't do data augmentation
        num_workers = 0

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        for epoch in range(1, self.num_epochs + 1):
            epoch_train_loss = 0

            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                epoch_train_loss += self.training_step(inputs, labels)

            # perform validation every `epochs_per_val` epochs
            if epoch % epochs_per_val == 0:
                stop_early, epoch_val_loss, epoch_metric = self.validate(
                    epoch, val_dataloader
                )

                if stop_early:
                    mlflow.log_metric("early_stopping_epoch", epoch)
                    print(f"Stopping early at epoch {epoch}/{self.num_epochs + 1}")
                    break

                if trial is not None:
                    trial.report(epoch_metric, step=epoch)
                    if trial.should_prune():
                        raise TrialPruned()

            epoch_train_loss /= len(train_dataset)
            mlflow.log_metric("train/epoch_loss", epoch_train_loss, step=epoch)

        return {"val_loss": epoch_val_loss, self.val_metric_name: epoch_metric}

    def validate(
        self, epoch: int, val_dataloader: DataLoader
    ) -> tuple[bool, float, float]:
        """
        Perform a pass over the validation dataset and log the validation loss and metric.
        NOTE: batch size of val dataloader is set to the size of the val
        dataset. This *can* be problematic if you can't load all the validation data into
        memory but for our purposes the data usually can easily fit into memory.
        """
        epoch_val_loss = 0
        epoch_metric = 0
        for inputs, labels in val_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            val_loss, val_metric = self.validation_step(inputs, labels)
            epoch_val_loss += val_loss
            epoch_metric += val_metric

        epoch_val_loss /= len(val_dataloader.dataset)
        epoch_metric /= len(val_dataloader.dataset)
        mlflow.log_metric("val/epoch_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric(f"val/epoch_{self.val_metric_name}", epoch_metric, step=epoch)

        return self.stopper.should_stop(epoch_metric), epoch_val_loss, epoch_metric


class DeepEnsembleTrainer:
    def __init__(
        self,
        ensemble_model: DeepEnsemble,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        batch_size: int,
        num_epochs: int,
        val_metric_type: Enum,
        loss_fn: Callable = nn.GaussianNLLLoss,
        device: str = "cpu",
        patience: int = 10,
        use_adversarial: bool = False,
    ) -> None:
        """
        Args:
            use_adversarial: use FGSM (Fast Gradient Sign Method) to perform adversarial training
        """

        self.ensemble_model = ensemble_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_epochs = num_epochs

        self.build_optimizers()

        self.device = device

        self.loss_fn = loss_fn()

        self.scaler = torch.amp.GradScaler()

        self.val_metric_name = val_metric_type.name
        val_metric_cls = val_metric_type.value[0]
        self.val_metric_fn = val_metric_cls()

        self.adversarial = use_adversarial
        # see Algorithm 1 in deep ensemble paper
        if self.adversarial:
            self.epsilon = 0.01 * self.ensemble_model[0].max_val

    def build_optimizers(self) -> None:
        """
        Build optimizers for the ensemble model.
        """

        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2),
            )
            for model in self.ensemble_model.models
        ]

    def training_step(
        self, model_idx: int, input: torch.Tensor, labels: torch.Tensor
    ) -> float:
        self.ensemble_model[model_idx].train()

        self.optimizers[model_idx].zero_grad()

        with torch.amp.autocast(device_type=self.device):
            # Output shape is (B, 2N) where B is batch size and N is the output dimension of the dataset.
            # First N elements are predictions for the N parameters, last N elements are the uncertainties.
            if self.adversarial:
                # require grad to compute grad with respect to input for adversarial training
                input.requires_grad = True
                output = self.ensemble_model[model_idx](input)
                mus, vars = (
                    output[:, : output.shape[1] // 2],
                    output[:, output.shape[1] // 2 :],
                )
                loss = self.loss_fn(mus, labels, vars)
                # keep computation graph for later gradient computation with total loss
                loss.backward(retain_graph=True)
                grad_x = input.grad.detach()

                adversarial_example = input + self.epsilon * grad_x.sign()
                output = self.ensemble_model[model_idx](adversarial_example)
                mus_adv, vars_adv = (
                    output[:, : output.shape[1] // 2],
                    output[:, output.shape[1] // 2 :],
                )
                loss += self.loss_fn(mus_adv, labels, vars_adv)  # compute total loss
                self.ensemble_model[model_idx].zero_grad()  # clear previous grads

            else:
                output = self.ensemble_model[model_idx](input)
                mus, vars = (
                    output[:, : output.shape[1] // 2],
                    output[:, output.shape[1] // 2 :],
                )
                loss = self.loss_fn(mus, labels, vars)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizers[model_idx])
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def validation_step(
        self, model_idx: int, input: torch.Tensor, labels: torch.Tensor
    ) -> tuple[float, float]:
        self.ensemble_model[model_idx].eval()
        # Output shape is (B, 2N) where B is batch size and N is the output dimension of the dataset.
        # First N elements are predictions for the N parameters, last N elements are the uncertainties.
        output = self.ensemble_model[model_idx](input)
        mus, vars = (
            output[:, : output.shape[1] // 2],
            output[:, output.shape[1] // 2 :],
        )
        val_loss = self.loss_fn(mus, labels, vars)
        val_metric = self.val_metric_fn(mus, labels)

        return val_loss.item(), val_metric.item()

    def train(
        self,
        train_dataset: QuantumJumpDataset,
        val_dataset: QuantumJumpDataset,
        epochs_per_val: int,
    ) -> dict[str, float]:
        # set to 0 since we can load all data into memory and we don't do data augmentation
        num_workers = 0
        epoch_val_loss = 0

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        for idx, model in enumerate(self.ensemble_model.models):
            model.train()
            model = model.to(self.device)

            for epoch in range(1, self.num_epochs + 1):
                epoch_train_loss = 0

                for inputs, labels in train_dataloader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    epoch_train_loss += self.training_step(idx, inputs, labels)

                # perform validation every `epochs_per_val` epochs
                if epoch % epochs_per_val == 0:
                    epoch_val_loss, epoch_metric = self.validate(
                        idx, epoch, val_dataloader
                    )

                epoch_train_loss /= len(train_dataset)
                mlflow.log_metric(
                    f"train/epoch_loss_model_{idx + 1}", epoch_train_loss, step=epoch
                )

            model = model.to("cpu")  # move model back to cpu to save GPU memory

        return {"val_loss": epoch_val_loss, self.val_metric_name: epoch_metric}

    def validate(
        self, model_idx: int, epoch: int, val_dataloader: DataLoader
    ) -> tuple[float, float]:
        """
        Perform a pass over the validation dataset and log the validation loss and metric.
        NOTE: batch size of val dataloader is set to the size of the val
        dataset. This *can* be problematic if you can't load all the validation data into
        memory but for our purposes the data usually can easily fit into memory.
        """
        epoch_val_loss = 0
        epoch_metric = 0
        for inputs, labels in val_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            val_loss, val_metric = self.validation_step(model_idx, inputs, labels)
            epoch_val_loss += val_loss
            epoch_metric += val_metric

        epoch_val_loss /= len(val_dataloader.dataset)
        epoch_metric /= len(val_dataloader.dataset)
        mlflow.log_metric(
            f"val/epoch_loss_model_{model_idx + 1}", epoch_val_loss, step=epoch
        )
        mlflow.log_metric(
            f"val/epoch_{self.val_metric_name}__model_{model_idx + 1}",
            epoch_metric,
            step=epoch,
        )

        return epoch_val_loss, epoch_metric


class BayesianInferenceModelTrainer:
    def __init__(self, hdi_prob: float = 0.95) -> None:
        self.hdi_prob = hdi_prob

    def train(self, model: BayesianInferenceModel, **kwargs) -> None:
        """
        Train (i.e. fit) the Bayesian inference model using the provided parameters.
        Args:
            model: An instance of BayesianInferenceModel.
            kwargs: Additional keyword arguments to pass to the fit method.
        """

        mlflow.log_param("hdi_prob", self.hdi_prob)

        trace = model.fit(**kwargs)

        summary = az.summary(trace, hdi_prob=self.hdi_prob)
        hdi_tails = (1 - 0.95) / 2 * 100
        # log metrics for parameter estimations
        for var_name in model.free_rvs:
            for stat in (
                "mean",
                "sd",
                "r_hat",
                "ess_bulk",
                "ess_tail",
                f"hdi_{round(hdi_tails, 2)}%",
                f"hdi_{100 - round(hdi_tails, 2)}%",
            ):
                # NOTE: Removing '%' from hdi interval name since mlflow doesn't allow that character in metric names
                mlflow.log_metric(f"{var_name}_{stat.replace('%', '')}", summary[stat][var_name])

        return
