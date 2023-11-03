import torch.nn.functional as F
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_residual_hiddens: int,
    ):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, hidden_dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(in_channels, hidden_dim, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)
