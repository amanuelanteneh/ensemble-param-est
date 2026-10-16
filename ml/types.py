from typing import Callable
from enum import Enum
from dataclasses import dataclass

from torch import nn
from optuna import Trial

from ml.models import MLPNetwork, DeepEnsemble, HistogramNetwork, DeepSetNetwork
from ml.loss import MSLELoss, RMSELoss
from ml.train import BaseNetworkTrainer, DeepEnsembleTrainer


class ModelType(Enum):
    """
    Enum for different types of models and their corresponding trainer class.
    """

    mlp = (MLPNetwork, BaseNetworkTrainer)
    hist = (HistogramNetwork, BaseNetworkTrainer)
    deepset = (DeepSetNetwork, BaseNetworkTrainer)
    ensemble = (DeepEnsemble, DeepEnsembleTrainer)


class LossType(Enum):
    """
    Enum for different types of loss functions and their corresponding boolean value
    indicating if lower values are better If True lower values are better.
    """

    mse = (nn.MSELoss, True)
    mae = (nn.L1Loss, True)
    msle = (MSLELoss, True)
    rmse = (RMSELoss, True)
    gnll = (nn.GaussianNLLLoss, True) 


class ActivationType(Enum):
    """
    Enum for different activation functions.
    """

    relu = nn.ReLU
    tanh = nn.Tanh
    lrelu = nn.LeakyReLU
    elu = nn.ELU
    gelu = nn.GELU
    silu = nn.SiLU


@dataclass
class Hyperparameters:
    lr: float = 1e-5
    weight_decay: float = 0
    beta1: float = 0.999
    beta2: float = 0.99
    num_epochs: int = 2
    batch_size: int = 64
    dropout: float = 0.0
    patience: int = 5
    activation: str = ActivationType.relu.name
    loss_fn: Callable = nn.MSELoss
    num_bins: int = 700

    @classmethod
    def from_optuna_trial(cls, trial: Trial) -> "Hyperparameters":
        num_bins = trial.suggest_int("num_bins", 200, 710, step=10)

        loss_fn_type = trial.suggest_categorical("training_loss_function", ["msle", "rmse"])
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        beta1 = trial.suggest_float("beta1", 0.8, 0.999)
        beta2 = trial.suggest_float("beta2", 0.8, 0.999)
        num_epochs = trial.suggest_int("num_epochs", 50, 300, step=5)
        
        batch_size = trial.suggest_categorical(
            "batch_size", [2**n for n in range(6, 12)]
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        patience = trial.suggest_int("patience", 4, 10, step=1)
        
        weight_decay = 0  # trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        activation = "relu"  # trial.suggest_categorical("activation", ["elu", "relu"])

        loss_fn = LossType[loss_fn_type].value[0]

        return cls(
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            num_epochs=num_epochs,
            batch_size=batch_size,
            dropout=dropout,
            patience=patience,
            activation=activation,
            loss_fn=loss_fn,
            num_bins=num_bins
        )
