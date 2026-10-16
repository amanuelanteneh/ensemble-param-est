from typing import Callable

import numpy as np
import torch
from torch import nn, repeat_interleave

from ml.models import DeepEnsemble


@torch.no_grad()
def compute_test_error(
    model: nn.Module,
    error_fn: Callable,
    features: torch.Tensor,
    labels: torch.Tensor,
    compute_vars: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], list[float]]:
    """
    Args:
        model: the model to compute error for
        error_fn: the error metric to compute
        features: the input data to the model (list of time delays) shape for paper test set (B, 10000, 48)
        labels: the ground truth labels (true parameter values)
    """

    errors = []
    biases = []
    vars = []

    for trajectories, delta in zip(features, labels):
        if isinstance(model, DeepEnsemble):
            batch_output, batch_vars = model(trajectories)
        else:
            batch_output = model(trajectories)

        repeated_labels = repeat_interleave(
            delta, repeats=batch_output.shape[0]
        ).reshape(-1, 1)

        error = error_fn(batch_output, repeated_labels)
        errors.append(error.cpu().numpy())

        # compute biases
        mean_output = torch.mean(batch_output, dim=0).cpu().item()
        biases.append(mean_output - delta.cpu())

        if compute_vars:
            var_output = torch.mean(batch_output**2).cpu().item() - mean_output**2
            vars.append(var_output)

    if compute_vars:
        return errors, biases, vars
    else:
        return errors, biases
