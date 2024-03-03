import torch.nn as nn


class GaussianNLL(nn.Module):
    """
    Compute the Gaussian negative log-likelihood of observations x under the distribution defined by mu and log-variance logvar.

    Parameters:
    - x: Tensor of observations. Shape: [batch_size, num_features]
    - mu: Mean of the Gaussian distribution. Shape: [batch_size, num_features]
    - logvar: Log variance of the Gaussian distribution. Shape: [batch_size, num_features]

    Returns:
    - Negative log-likelihood of observing x under the Gaussian distribution defined by mu and logvar.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["sum", "mean"]:
            raise ValueError(f"Unsupported reduction '{reduction}'. Use 'sum' or 'mean'.")
        self.reduction = reduction

    def forward(self, x, mu, logvar):
        # ignore the log(2*pi) term to the log variance part of the NLL calculation
        nll = 0.5 * (
            logvar
            + (x - mu) ** 2 / (logvar.exp())
        ).sum(dim=-1)  # Sum over features for each item in the batch

        if self.reduction == "mean":
            return nll.mean()

        elif self.reduction == "sum":
            return nll.sum()

        else:
            raise NotImplementedError
