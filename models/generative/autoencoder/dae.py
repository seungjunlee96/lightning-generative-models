from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.optim.optimizer import Optimizer

from utils.visualization import make_grid


class Encoder(nn.Module):
    """
    Encoder module: Fully connected neural network for encoding images.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Add more layers if needed

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.model(x)
        return x


class Decoder(nn.Module):
    """
    Decoder module: Fully connected neural network for decoding images.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = x.view(x.size(0), 1, 28, 28)  # Reshape back to image dimensions
        return x


class DAE(pl.LightningModule):
    """
    DenoisingAutoencoder module: Implements a denoising autoencoder using an Encoder and a Decoder.
    Supports different types of noise addition for robust training.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        metric (nn.Module): Loss function.
        noise_type (str): Type of noise to add to the input.
        noise_level (float): Intensity of the noise.
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 28,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.metric = nn.MSELoss()
        self.noise_type = noise_type
        self.noise_level = noise_level

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reconstructed output.
        """
        return self.decoder(self.encoder(x))

    def _common_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Training step for the autoencoder.

        Args:
            batch (Tuple[Tensor, Tensor]): A batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: The training loss.
        """
        x, _ = batch
        noisy_x = self.add_noise(x)
        representation = self.encoder(noisy_x)
        x_hat = self.decoder(representation)
        loss = self.metric(x, x_hat)
        if not self.training and batch_idx == 0:
            self._log_images(x_hat, fig_name="Random Generation")
        return loss

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Training step for the autoencoder.

        Args:
            batch (Tuple[Tensor, Tensor]): A batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: The training loss.
        """
        loss = self._common_step(batch, batch_idx)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Training step for the autoencoder.

        Args:
            batch (Tuple[Tensor, Tensor]): A batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: The training loss.
        """
        loss = self._common_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        return loss

    def add_noise(self, x: Tensor) -> Tensor:
        """
        Adds noise to the input tensor based on the specified noise type and level.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Noisy input tensor.
        """
        if self.noise_type == "gaussian":
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        elif self.noise_type == "salt_and_pepper":
            return self.add_salt_and_pepper_noise(x, self.noise_level)
        else:
            raise ValueError("Invalid noise type specified")

    @staticmethod
    def add_salt_and_pepper_noise(
        x: Tensor,
        noise_level: float,
    ) -> Tensor:
        """
        Adds salt and pepper noise to the input tensor.

        Args:
            x (Tensor): Input tensor.
            noise_level (float): Intensity of the noise.

        Returns:
            Tensor: Input tensor with salt and pepper noise added.
        """
        # Create masks for salt and pepper noise
        salt_mask = torch.rand_like(x) < (noise_level / 2)
        pepper_mask = torch.rand_like(x) < (noise_level / 2)

        x = torch.where(salt_mask, torch.ones_like(x), x)
        x = torch.where(pepper_mask, torch.zeros_like(x), x)
        return x

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer, Optimizer]]:
        """
        Configures the optimizers for training the autoencoder.

        Returns:
            Union[Optimizer, Tuple[Optimizer, Optimizer]]: The optimizer or a tuple of (optimizer, scheduler).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @torch.no_grad()
    def _log_images(self, images: Tensor, fig_name: str):
        # Normalize the images to the range [0, 255] for visualization
        images = ((images + 1.0) / 2.0) * 255.0
        images = images.clamp(0, 255).byte().detach().cpu()

        # Create a grid of images and log it
        fig = make_grid(images)
        self.logger.experiment.log(
            {fig_name: [wandb.Image(fig)]},
            step=self.global_step,
        )
