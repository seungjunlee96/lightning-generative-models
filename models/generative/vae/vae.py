"""
References:
    https://keras.io/examples/generative/vae/
    https://www.tensorflow.org/tutorials/generative/cvae
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid


class Encoder(nn.Module):
    """
    Encoder defines the approximate posterior distribution q(z|x),
    which takes an input image as observation and outputs a latent representation defined by
    mean (mu) and log variance (log_var). These parameters parameterize the Gaussian
    distribution from which we can sample latent variables.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
    ):
        super(Encoder, self).__init__()
        self.img_shape: List[int] = [img_channels, img_size, img_size]
        self.latent_dim: int = latent_dim

        # Neural network layers to process the input image
        self.layers = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )

        # Layers to produce mu and log_var
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_flatten = x.view(x.size(0), -1)
        out = self.layers(x_flatten)
        return self.mu(out), self.log_var(out)


class Decoder(nn.Module):
    """
    Decoder defines the conditional distribution of the observation p(x|z),
    which takes a latent sample as input and outputs the parameters for a conditional distribution of the observation.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
    ):
        super(Decoder, self).__init__()
        self.img_shape: List[int] = [img_channels, img_size, img_size]
        self.latent_dim: int = latent_dim

        # Neural network layers to process the latent variable and produce the reconstructed image
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, np.prod(self.img_shape)),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        out = self.layers(z)
        return out.view(out.size(0), *self.img_shape)

    def random_sample(self, batch_size: int) -> Tensor:
        z = torch.randn([batch_size, self.latent_dim], device=self.device)
        return self(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class VAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE): Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114

    VAE is a generative model that learns a probabilistic mapping between the input data
    space and a latent space. The encoder maps an input to a distribution in the latent space,
    and the decoder maps points in the latent space back to the data space. VAE introduces a
    regularization term to ensure that the learned latent space is continuous, making it suitable
    for generative tasks.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int = 20,
        lr: float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
        kld_weight: float = 1e-2,
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            img_channels=img_channels, img_size=img_size, latent_dim=latent_dim
        )
        self.decoder = Decoder(
            img_channels=img_channels, img_size=img_size, latent_dim=latent_dim
        )

        self.z = torch.randn([16, latent_dim])
        self.summary()

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from the latent Gaussian distribution.

        This is a key component of VAEs, enabling backpropagation through the random sampling step.

        Args:
            mu (Tensor): Mean of the Gaussian distribution.
            log_var (Tensor): Log variance of the Gaussian distribution.

        Returns:
            Tensor: Sampled latent variable.
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def _common_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, split: str
    ) -> Tensor:
        assert split in ["train", "val", "test"]
        x, c = batch
        x_hat, mu, log_var = self(x)

        recon_loss = l1_loss(x_hat, x)
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self.hparams.kld_weight * kld

        self.log_dict(
            {
                f"{split}_loss": loss,
                f"{split}_recon_loss": recon_loss,
                f"{split}_kld": kld,
            },
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

        if not self.training:
            """Log and cache latent variables for visualization."""
            if batch_idx == 0:
                self._log_images(
                    self.decoder(self.z.to(self.device)),
                    "Random Generation",
                )

                self.latents = []
                self.conds = []

            z = self.reparameterize(mu, log_var).detach().cpu()
            c = c.cpu()

            self.latents.append(z)
            self.conds.append(c)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._common_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._log_latent_embeddings()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def _log_images(self, images: Tensor, fig_name: str):
        fig = make_grid(
            tensor=images,
            value_range=(-1, 1),
            normalize=True,
        )
        self.logger.experiment.log(
            {fig_name: [wandb.Image(fig)]}, step=self.global_step
        )

    @torch.no_grad()
    def _log_latent_embeddings(self):
        """Log the latent space embeddings to WandB for visualization."""
        z = torch.cat(self.latents)
        c = torch.cat(self.conds)
        data = {f"z_{i}": z[:, i] for i in range(z.size(1))}
        data["c"] = c
        data = pd.DataFrame(data)
        self.logger.experiment.log(
            {"latent space": wandb.Table(data=data)}, step=self.global_step
        )

        del self.latents
        del self.conds

    def summary(
        self,
        col_names: List[str] = [
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
    ):
        x = torch.randn(
            [
                1,
                self.hparams.img_channels,
                self.hparams.img_size,
                self.hparams.img_size,
            ]
        )

        summary(
            self,
            input_data=x,
            col_names=col_names,
        )
