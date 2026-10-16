from typing import Callable

from torch import nn, Tensor, stack, cat
import torch.nn.functional as F
import arviz as az
import pymc as pm

from ml.layers import HistogramLayer, SoftHistogramLayer


class MLPNetwork(nn.Module):
    """
    A simple MLP network (fully connected feed forward neural network).

    Args:
        input_dim: Size of the input features.
        hidden_dims: Size of the hidden layers.
        output_dim: Size of the output layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128, 64, 32],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: Size of the input features.
            hidden_dim: the sizes of the hidden layers,
                         default values from https://doi.org/10.1088/2058-9565/ad3c68
            output_dim: Size of the output layer.
            activation: Activation function to use in the hidden layers.
            dropout: Dropout rate for regularization.
        """

        super(MLPNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.activation = activation

        self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.hidden_layers.append(self.activation())
        self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.hidden_layers.append(nn.Dropout(p=dropout))

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.hidden_layers.append(self.activation())
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.hidden_layers.append(nn.Dropout(p=dropout))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP network.

        Args:
        - x: Input tensor of shape (batch_size, input_dim).

        Returns:
        - Output tensor of shape (batch_size, output_dim).
        """

        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out



class HistogramNetwork(nn.Module):
    def __init__(
        self,
        num_bins: int = 700,
        max_val: float = 100,
        min_val: float = 0,
        init_sigma: float = 1.0,
        hidden_dims: list[int] = [100, 50, 30],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        super(HistogramNetwork, self).__init__()

        self.max_val = max_val
        self.min_val = min_val
        self.activation = activation

        self.hidden_layers = nn.ModuleList()

        # Soft histogram layer
        self.hidden_layers.append(
            SoftHistogramLayer(
                num_bins=num_bins,
                max_val=self.max_val,
                min_val=self.min_val,
                init_sigma=init_sigma,
            )
        )
        first_linear_input_dim = num_bins

        # Fully connected stack
        self.hidden_layers.append(nn.Linear(first_linear_input_dim, hidden_dims[0]))
        self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.hidden_layers.append(self.activation())

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.hidden_layers.append(self.activation())
            self.hidden_layers.append(nn.Dropout(p=dropout))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:

        # Fully connected stack
        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)
        return out

class ConvNetwork(nn.Module):
    def __init__(
        self,
        out_features: int = 200,
        hidden_dims: list[int] = [100, 50, 30],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        super(ConvNetwork, self).__init__()

        self.activation = activation

        self.hidden_layers = nn.ModuleList()

        # Expect input shape: (batch_size, seq_len, 1)
        # Conv1d expects: (batch_size, channels, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_features // 2,
            kernel_size=6,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(out_features // 2)

        self.conv2 = nn.Conv1d(
            in_channels=out_features // 2,
            out_channels=out_features,
            kernel_size=6,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(out_features)

        # After global pooling we will have size (batch_size, out_features)
        first_linear_input_dim = out_features

        # Fully connected stack
        self.hidden_layers.append(nn.Linear(first_linear_input_dim, hidden_dims[0]))
        self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.hidden_layers.append(self.activation())

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.hidden_layers.append(self.activation())
            self.hidden_layers.append(nn.Dropout(p=dropout))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Input: (batch, seq_len, 1)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation()(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=2)  # -> (batch, out_features)

        # Fully connected stack
        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)
        return out


class EnsembleComponent(HistogramNetwork):

    def __init__(
        self,
        num_bins: int = 700,
        max_val: float = 100,
        min_val: float = 0,
        init_sigma: float = 1.0,
        hidden_dims: list[int] = [100, 50, 30],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        output_dim *= 2

        super().__init__(
            num_bins=num_bins,
            max_val=max_val,
            min_val=min_val,
            init_sigma=init_sigma,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:

        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)

        vars_start = out.shape[1] // 2
        mus = out[:, :vars_start]
        vars = F.softplus(out[:, vars_start:]) + 1e-6

        return cat([mus, vars], dim=1)


class EnsembleComponentConv(ConvNetwork):

    def __init__(
        self,
        out_features: int = 200,
        hidden_dims: list[int] = [100, 50, 30],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        output_dim *= 2

        super().__init__(
            out_features=out_features,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:

        x = x.unsqueeze(1)  # (batch, 1, seq_len)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation()(x)

        x = x.mean(dim=2)

        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)

        vars_start = out.shape[1] // 2
        mus = out[:, :vars_start]
        vars = F.softplus(out[:, vars_start:]) + 1e-6

        return cat([mus, vars], dim=1)

# TODO: Merge add ability to use soft histogram OR hard histogram layer

class DeepEnsemble(nn.Module):
    def __init__(self, models: list[EnsembleComponent]):
        """
        Initializes a Deep Ensemble model.
        """
        super(DeepEnsemble, self).__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)

    def __getitem__(self, model_idx: int) -> nn.Module:
        """
        Get a specific model from the ensemble.
        Args:
            idx: Index of the model to retrieve.
        Returns:
            The model at the specified index.
        """
        return self.models[model_idx]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        NOTE: We can probably use vmap to parallelize this inference
        """
        # NOTE: outputs shape: (num_models, batch_size, 2*num_outputs)
        outputs = stack([model(x) for model in self.models])
        
        # First `num_outputs` elements are the means, the remaining are the variances
        ensemble_means, ensemble_vars = (
            outputs[:, :, : outputs.shape[2] // 2],
            outputs[:, :, outputs.shape[2] // 2 :],
        )

        # see page 5 of paper
        aggregated_means = ensemble_means.mean(dim=0)
        aggregated_vars = (ensemble_vars + ensemble_means**2).mean(dim=0) - aggregated_means**2
        
        return aggregated_means, aggregated_vars


class HistNetwork(nn.Module):
    """
    A PyTorch implementation of the HistDense network with the default architecture being
    the same as in https://doi.org/10.1088/2058-9565/ad3c68.

    Args:

    """

    def __init__(
        self,
        num_bins: int = 700,
        taumax: float = 100,
        trainable: bool = True,
        hidden_dims: list[int] = [100, 50, 30],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_bins: Number of bins in the histogram. Default same as in paper.
            taumax: Max value in the histogram. Default same as in paper.
            hidden_dims: the sizes of the hidden layers,
                         default values from https://doi.org/10.1088/2058-9565/ad3c68
            output_dim: Size of the output layer.
            activation: Activation function to use in the hidden layers.
            dropout: Dropout rate for regularization.
        """

        super(HistNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.activation = activation

        self.hidden_layers.append(
            HistogramLayer(num_bins=num_bins, taumax=taumax, trainable=trainable)
        )
        self.hidden_layers.append(nn.Linear(num_bins, hidden_dims[0]))
        self.hidden_layers.append(self.activation())

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.hidden_layers.append(self.activation())
            self.hidden_layers.append(nn.Dropout(p=dropout))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP network.

        Args:
        - x: Input tensor of shape (batch_size, input_dim).

        Returns:
        - Output tensor of shape (batch_size, output_dim).
        """

        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out


class DeepSetNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims_phi: list[int] = [64, 64, 64],
        hidden_dims_rho: list[int] = [64, 32, 16],
        phi_activation: Callable = nn.ReLU,
        rho_activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        NOTE: For deep set networks the
              input_dim should be the dimension of the *elements* in the set.
        Args:
            input_dim: Dimension of each *element* in the set.
            output_dim: Dimension of the output.
            hidden_dims_rho: Hidden dimensions for the rho mapping.
            hidden_dims_phi: Hidden dimensions for the phi mapping.
            phi_activation: Activation function for phi layers.
            rho_activation: Activation function for rho layers.
            dropout: Dropout rate to apply after each layer.
        """
        super().__init__()

        # phi maps each element x ∈ X
        # For some reason using BatchNorm layer messes up phi network
        phi_layers = [
            nn.Linear(input_dim, hidden_dims_phi[0]),
            phi_activation(),
            # nn.Dropout(p=dropout),
        ]

        for i in range(len(hidden_dims_phi) - 1):
            phi_layers.extend(
                [
                    nn.Linear(hidden_dims_phi[i], hidden_dims_phi[i + 1]),
                    phi_activation(),
                    # nn.Dropout(p=dropout),
                ]
            )

        phi_layers.append(nn.Linear(hidden_dims_phi[-1], hidden_dims_phi[-1]))
        self.phi = nn.Sequential(*phi_layers)

        # rho maps the sum of phi(x_i)
        rho_layers = [
            nn.Linear(hidden_dims_phi[-1], hidden_dims_rho[0]),
            rho_activation(),
            # nn.BatchNorm1d(hidden_dims_rho[0]),
            nn.Dropout(p=dropout),
        ]

        for i in range(len(hidden_dims_rho) - 1):
            rho_layers.extend(
                [
                    nn.Linear(hidden_dims_rho[i], hidden_dims_rho[i + 1]),
                    rho_activation(),
                    # nn.BatchNorm1d(hidden_dims_rho[i + 1]),
                    nn.Dropout(p=dropout),
                ]
            )

        rho_layers.append(nn.Linear(hidden_dims_rho[-1], output_dim))
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x: Tensor):
        """
        Args:
        x: input tensor of shape (batch_size, set_size, input_dim) where
                `input_dim` is the dimension of each *element* in the set
                e.g. in the two-level system these are scalars so it is 1.
                `set_size` is the number of elements in the set, e.g. number of jumps.
                `batch_size` is the number of sets in the batch, e.g. number of trajectories.
        """
        x = self.phi(x)  # shape: (batch_size, set_size, hidden_dim)
        # sum over set elements -> (batch_size, hidden_dim)
        # add up phi representations of each element in the set
        x = x.sum(dim=1)
        out = self.rho(x)  # final output
        return out


class BayesianInferenceModel:
    """
    A class for performing Bayesian inference using PyMC.
    """

    def __init__(self, model: pm.Model):
        self.model = model
        self.free_rvs = [rv.name for rv in model.free_RVs]

    def fit(self, **kwargs) -> az.InferenceData:
        """
        Fit the Bayesian model using the provided parameters.

        Args:
        - kwargs: Additional keyword arguments to pass to the sampling function.

        Returns:
        - trace: The trace of the sampled parameters.
        """

        with self.model as model:
            trace = pm.sample(model=model, **kwargs)

        return trace
