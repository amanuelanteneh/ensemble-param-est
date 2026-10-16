from typing import Callable
from dataclasses import asdict

from optuna import Trial
import mlflow
from mlflow.models import infer_signature

from ml.models import HistogramNetwork
from ml.types import Hyperparameters, LossType, ActivationType
from ml.data import QuantumJumpDataset


def objective(
    trial: Trial,
    model_cls: Callable,
    trainer_cls: Callable,
    train_dataset: QuantumJumpDataset,
    val_dataset: QuantumJumpDataset,
    device: str,
    objective_metric: str,
    epochs_per_val: int,
    hidden_layer_sizes: list[int]
) -> float:
    """
    Objective function for hyperparameter tuning using Optuna.

    Args:
        trial: The Optuna trial object.
        model_cls: The model class to instantiate.
        trainer_cls: The trainer class to instantiate and use for training.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        objective_metric: The metric to optimize (e.g., 'rmse').
        epochs_per_val: Number of epochs to train before validation.
         hidden_layer_sizes: The sizes of the hidden layers

    Returns:
        float: The objective metric to minimize.
    """

    hyperparams = asdict(Hyperparameters.from_optuna_trial(trial))

    # pop to avoid passing to the trainer constructor when using ** unpacking
    dropout = hyperparams.pop("dropout")
    activation = hyperparams.pop("activation")
    num_bins = hyperparams.pop("num_bins", 0)

    if model_cls == HistogramNetwork:
        model = model_cls(
            output_dim=train_dataset.labels.shape[1],
            num_bins=num_bins,
            activation=ActivationType[activation].value,
            dropout=dropout,
            hidden_dims=hidden_layer_sizes
        )
    else:
        model = model_cls(
            input_dim=train_dataset.data.shape[-1],
            output_dim=train_dataset.labels.shape[1],
            activation=ActivationType[activation].value,
            dropout=dropout,
        )

    trainer = trainer_cls(
        model,
        device=device,
        val_metric_type=LossType[objective_metric],
        **hyperparams,
    )

    with mlflow.start_run(nested=True, run_name=f"Trial {trial.number}") as child_run:
        trial.set_user_attr("run_id", child_run.info.run_id)
        trial.set_user_attr("run_name", child_run.info.run_name)

        mlflow.log_param("model", model_cls.__name__)
        mlflow.log_param("hidden_layer_sizes", str(hidden_layer_sizes))
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("activation", activation)
        mlflow.log_param("num_bins", num_bins)
        mlflow.log_param("train_loss_function", hyperparams.pop("loss_fn").__name__)
        mlflow.log_param("val_loss_function", LossType[objective_metric].value[0].__name__)
        mlflow.log_params(hyperparams)

        metrics = trainer.train(train_dataset, val_dataset, epochs_per_val, trial)
        mlflow.log_metrics(metrics)

        # log model
        mlflow.pytorch.log_model(
            model,
            name="torch_model",
            signature=infer_signature(train_dataset[0][0].numpy()),
        )
        mlflow.end_run()  # end the run

    return metrics[objective_metric]
