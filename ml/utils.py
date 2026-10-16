from torch import nn

def count_trainable_parameters(model: nn.Module) -> int:
    """
    Computes the number of trainable parameters
    in a PyTorch model

    Args:
        model: The torch model to compute number of parameters of
    Returns:
        The number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """
    Computes the size of a PyTorch model in megabytes.
    See: https://discuss.pytorch.org/t/finding-model-size/130275/2

    Args:
        model: The torch model to compute size of
    Returns:
        The size of the model in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_in_mb = (param_size + buffer_size) / 1024**2

    return size_in_mb