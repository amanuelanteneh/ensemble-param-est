import torch
from torch import nn, Tensor, log, clamp, sqrt


class MSLELoss(nn.Module):
    """
    Custom Mean Squared Logarithmic Error (MSLE) loss 
    for PyTorch.

    NOTE: Both `pred` and `actual` should be > 0 due to the log.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)
        
    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        """
        NOTE: As done by the TensorFlow/Keras implementation of 
        this loss we clamp `preds` such that they are at least 1e-7
        since values of 0 or less will cause errors in the evaluation of the log.
        See: https://www.tensorflow.org/api_docs/python/tf/keras/losses/MSLE

        Args:
            preds: Model outputs (batch_size, N)
            actual: Ground truth (label) values (batch_size, N)

        Returns:
            Average MSLE over the batch
        """
        
        pred = clamp(pred, min=1e-7)
        loss = self.mse(log(pred + 1), log(actual + 1))
        return loss.mean()


class RMSELoss(nn.Module):
    """
    Custom Root Mean Squared Error (RMSE) loss for PyTorch.
    See: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        self.mse = nn.MSELoss(**kwargs)
        
    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        # add 1e-8 for numerical stability
        return sqrt(self.mse(pred, actual) + 1e-8)

