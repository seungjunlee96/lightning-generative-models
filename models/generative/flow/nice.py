import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam


class CouplingNet(nn.Module):
    """A small feed-forward network for the coupling layer transformation."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super(CouplingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super(CouplingLayer, self).__init__()
        self.transformation = CouplingNet(dim // 2, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        if reverse:
            y2 = x2 - self.transformation(x1)
        else:
            y2 = x2 + self.transformation(x1)
        return torch.cat([x1, y2], dim=1)


class ScalingLayer(nn.Module):
    def __init__(self, dim: int):
        super(ScalingLayer, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return (
            x * torch.exp(self.log_scale)
            if not reverse
            else x * torch.exp(-self.log_scale)
        )


class NICE(pl.LightningModule):
    def __init__(self, input_dim: int, n_coupling_layers: int = 4):
        super(NICE, self).__init__()
        self.layers = nn.ModuleList(
            [CouplingLayer(input_dim) for _ in range(n_coupling_layers)]
        )
        self.scaling_layer = ScalingLayer(input_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            x = self.scaling_layer(x, reverse)
            for layer in reversed(self.layers):
                x = layer(x, reverse)
        else:
            for layer in self.layers:
                x = layer(x)
            x = self.scaling_layer(x)
        return x

    def _compute_loss(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        log_likelihood = -0.5 * z.pow(2).sum(1) - 0.5 * z.size(1) * torch.log(
            2 * torch.tensor(torch.pi)
        )
        log_abs_det_jacobian = self.scaling_layer.log_scale.sum()
        return (log_likelihood - log_abs_det_jacobian).mean()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self(x)
        loss = -self._compute_loss(z, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


# Example usage:
# model = NICE(input_dim=784)
# trainer = pl.Trainer(max_epochs=10)
# Assuming train_loader is the DataLoader for your training data
# trainer.fit(model, train_loader)
