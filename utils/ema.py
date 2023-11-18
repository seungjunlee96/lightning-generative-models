from copy import deepcopy

import torch
from torch.nn import Module


class ExponentialMovingAverage:
    """
    Maintains an Exponential Moving Average (EMA) version of a PyTorch model.

    Provides a smoothed model version for better generalization, especially during evaluation.
    """

    def __init__(
        self,
        model: Module,
        beta: float = 0.995,
        start_step: int = 500,
        update_interval: int = 10,
    ) -> None:
        """
        Args:
            model (Module): The initial model to create the EMA from.
            beta (float): Decay factor for the EMA.
            start_step (int): The step from which to start updating the EMA.
            update_interval (int): Interval in steps for EMA updates.
        """
        self.beta = beta
        self.start_step = start_step
        self.update_interval = update_interval
        self.ema_model = self._initialize_ema_model(model)

    def __call__(self, x):
        return self.ema_model(x)

    def _initialize_ema_model(self, model: Module) -> Module:
        """Initialize the EMA model using the provided model's parameters."""
        ema_model = deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)
        return ema_model

    def step(self, model: Module, current_step: int) -> None:
        """Update the EMA model based on the current training step."""
        if current_step > self.start_step and current_step % self.update_interval == 0:
            self._update_average_parameters(model)

    @torch.no_grad()
    def _update_average_parameters(self, model: Module) -> None:
        """
        Update the EMA model's parameters using the provided model.
        """
        for ema_param, model_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.copy_(self.beta * ema_param + (1 - self.beta) * model_param)

    def to(self, device: torch.device) -> "ExponentialMovingAverage":
        """Move the EMA model to the specified device and return the updated object."""
        self.ema_model = self.ema_model.to(device)
        return self
