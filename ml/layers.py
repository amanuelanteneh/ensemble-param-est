import torch
from torch import nn
from torch import Tensor


class SoftHistogramLayer(nn.Module):
    def __init__(
        self,
        num_bins: int = 700,
        min_val: float = 0.0,
        max_val: float = 100.0,
        init_sigma: float = 0.1,
        normalize: bool = True,
    ) -> None:
        """
        A custom layer which converts a 1D tensor, e.g. list of time delay measurements,
        into a `soft` histogram using a Gaussian kernel to assign a weight to each bin
        for each point essentially implementing a simple kernel density estimation protocol. 
        This is fully differentiable unlike a standard `hard` histogram.

        Args:
            

        """
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.normalize = normalize

        # make log(sigma) trainable to ensure sigma > 0
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))

        # fixed bin centers
        bin_width = (max_val - min_val) / self.num_bins
        bin_centers = torch.linspace(
            min_val + bin_width / 2, max_val - bin_width / 2, self.num_bins
        )
        # register bin_centers to buffer tensor to make it *non*-trainable
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, N)
            NOTE: Tensors do not need to be of same shape (N)

        Returns:
            Tensor: Soft histogram counts of shape (batch_size (B), num_bins)
        """
        if x.ndim != 2:
            raise ValueError(f"Expected input of shape (batch_size, N), got {x.shape}")

        sigma = torch.exp(self.log_sigma)

        # Shape: (B, N, 1) - (1, 1, num_bins) → (B, N, num_bins)
        x_exp = x.unsqueeze(-1)  # (B, N, 1)
        bin_centers_exp = self.bin_centers.view(1, 1, -1)  # (1, 1, num_bins)
        diff = (x_exp - bin_centers_exp) ** 2

        # Gaussian kernel weights
        weights = torch.exp(-diff / (2 * sigma**2))  # (B, N, num_bins)

        # Sum over N to get per-sample histogram
        soft_counts = weights.sum(dim=1)  # (B, num_bins)

        if self.normalize:
            # Normalize histograms so each row sums to 1
            soft_counts = soft_counts / (soft_counts.sum(dim=1, keepdim=True) + 1e-8)

        return soft_counts


class HistogramLayer(nn.Module):
    def __init__(
        self,
        num_bins: int,
        taumax: float = 100.0,
        trainable: bool = True,
        negative: float = 0.0,
    ) -> None:
        """
        A PyTorch implementation of the custom HistDense layer
        presented in https://doi.org/10.1088/2058-9565/ad3c68
        based on the publicly available GitHub code.
        """
        super().__init__()
        self.taumax = taumax
        self.num_bins = num_bins
        self.trainable = trainable
        self.negative = negative

        width_0 = taumax / num_bins
        shifts_init = torch.linspace(
            -negative * taumax + width_0 / 2,
            taumax - width_0 / 2,
            num_bins,
            dtype=torch.float32,
        ).reshape(num_bins, 1)

        self.width = nn.Parameter(
            torch.tensor([width_0], dtype=torch.float32), requires_grad=trainable
        )
        self.shifts = nn.Parameter(shifts_init, requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return self.histogram_sigmoid(x, self.num_bins, self.shifts, self.width)

    @staticmethod
    def theta(x: Tensor) -> Tensor:
        return (x >= 0).float()

    @staticmethod
    def sigmoid(x: Tensor, factor: float = 20.0) -> Tensor:
        return torch.sigmoid(factor * x)

    @staticmethod
    def theta_prod(x: Tensor, width: float = 1.0) -> Tensor:
        return HistogramLayer.theta(x + width / 2) * HistogramLayer.theta(
            -x + width / 2
        )

    @staticmethod
    def theta_prod_sigmoid(
        x: Tensor, factor: float = 20.0, width: float = 1.0
    ) -> Tensor:
        return HistogramLayer.sigmoid(x + width / 2, factor) * HistogramLayer.sigmoid(
            -x + width / 2, factor
        )

    @staticmethod
    def histogram_sigmoid(
        input_tensor: Tensor,  # shape: (batch_size, dim)
        num_bins: int,
        shifts: Tensor,  # shape: (num_bins, dim)
        width: Tensor,  # scalar, shape (num_bins,), or shape (num_bins, dim)
    ) -> Tensor:
        """
        Differentiable histogram with sigmoid-shaped bins for 2D batched input.

        Returns:
            Tensor of shape (batch_size, num_bins)
        """
        B, D = input_tensor.shape  # batch size and input dim

        # Expand tensors for broadcasting
        input_exp = input_tensor.unsqueeze(1)  # (B, 1, D)
        shifts_exp = shifts.unsqueeze(0)  # (1, num_bins, D)
        input_shifted = input_exp - shifts_exp  # (B, num_bins, D)

        factor = 20.0

        # Handle width broadcasting
        if width.ndim == 0:
            width = width.expand(num_bins, D)  # scalar → (num_bins, D)
        elif width.ndim == 1:
            width = width.unsqueeze(1).expand(
                num_bins, D
            )  # (num_bins,) → (num_bins, D)
        elif width.ndim == 2 and width.shape[1] != D:
            raise ValueError(f"Width shape {width.shape} incompatible with dim={D}")

        width = width.unsqueeze(0)  # (1, num_bins, D)

        input_theta = torch.sigmoid(
            (input_shifted + width / 2) * factor
        ) * torch.sigmoid((-input_shifted + width / 2) * factor)  # (B, num_bins, D)

        histogram = input_theta.sum(dim=-1)  # (B, num_bins)

        return histogram
