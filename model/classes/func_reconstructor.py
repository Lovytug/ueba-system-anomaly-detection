import torch.nn.functional as F


def reconstruction_error(x, x_hat):
    return F.mse_loss(
        x_hat,
        x,
        reduction="none"
    ).mean(dim=1)