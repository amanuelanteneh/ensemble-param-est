import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchvision.transforms import Compose


class QuantumJumpDataset(Dataset):
    def __init__(
        self,
        data_array: NDArray[np.float64],
        labels_array: NDArray[np.float64],
        transform: bool = False, 
        label_noise_std: float = 0.0,
        feature_noise_std: float = 0.0,
        seed: int | None = None
    ):
        """
        A class to hold quantum jump data. 
        If noise is added we add Gaussian noise drawn from N(0, noise_std).
        """
        self.feature_noise_std = feature_noise_std
        self.label_noise_std = label_noise_std

        if self.feature_noise_std > 0:
            data_array = add_feature_noise(data_array, self.feature_noise_std, seed)
        
        if self.label_noise_std > 0:
            labels_array = add_label_noise(labels_array, self.label_noise_std, seed)

        # Convert NumPy arrays to PyTorch Tensors
        self.data = torch.from_numpy(data_array).float()
        self.labels = torch.from_numpy(labels_array).float()

        if transform:
            self.transform = Compose([PermuteTensor()])
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        label = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class PermuteTensor:
    def __init__(self):
        """
        Custom transformation that randomly permute the elements of a 1D tensor.
        This is used for data augmentation of the input vectors
        which consist of waiting times between photon detections.
        Since these waiting times are independent due to the nature of
        our particular TLS system, we can permute them without losing information
        and potentially reducing the amount of data needed for training
        and improving generalization.
        """
        pass

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor: A 1D tensor to be permuted.

        Returns:
            Tensor: A new tensor with the same elements as the input tensor, but in a random
            order.
        """

        if tensor.ndim != 1:
            raise ValueError("Input tensor must be 1D.")

        perm = torch.randperm(tensor.size(0))

        return tensor[perm]


def add_label_noise(labels: np.ndarray, std: float, seed: int | None = None) -> np.ndarray:
    """
    Adds Gaussian noise with mean 0 and standard deviation `std` 
    to labels
    """
    if seed is not None:
       # set random seed for reproducibility of datasets
       np.random.seed(seed)
    
    noise = np.random.normal(loc=0.0, scale=std, size=labels.shape)
    labels = np.abs(labels + noise)
    return labels

def add_feature_noise(features: np.ndarray, std: float, seed: int | None = None) -> np.ndarray:
    """
    Adds Gaussian noise with mean 0 and standard deviation `std` 
    to features
    """
    if seed is not None:
       # set random seed for reproducibility of datasets
       np.random.seed(seed)
    
    noise = np.random.normal(loc=0.0, scale=std, size=features.shape)
    features = (features + noise).clip(min=0.001)
    return features
