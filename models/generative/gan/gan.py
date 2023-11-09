import os
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam

from utils.visualization import make_grid


class Generator(nn.Module):
    """
    Generator module for a GAN.

    The generator's primary role is to map latent space vectors (random noise)
    to data space. As the GAN trains, the generator progressively refines its
    ability to create fake images in hopes of fooling the discriminator into
    believing they're real.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
    ):
        super(Generator, self).__init__()
        self.img_shape = [img_channels, img_size, img_size]
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, np.prod(self.img_shape)),
            nn.LeakyReLU(0.2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z).view(-1, *self.img_shape)

    def random_sample(self, batch_size: int) -> torch.Tensor:
        z = torch.randn([batch_size, self.latent_dim], device=self.device)
        return self(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class Discriminator(nn.Module):
    """
    Discriminator module for a GAN.

    Acting as a binary classifier, the discriminator tries to distinguish
    between genuine and fake data. It outputs a scalar probability that the
    input image is real (as opposed to fake). During training, its objective
    is to improve its accuracy in this classification task.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
    ):
        super(Discriminator, self).__init__()
        img_shape = [img_channels, img_size, img_size]
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flatten = x.view(x.size(0), -1)
        out = self.model(x_flatten)
        return out.squeeze()


class GAN(pl.LightningModule):
    """
    Complete GAN module, housing both the generator and the discriminator.

    The adversarial nature of the GAN framework comes from the generator's
    goal to produce data that the discriminator mistakes as real, while the
    discriminator aims to get better at distinguishing real data from fake.
    This results in a two-player minimax game, as described in the original GAN paper.
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 28,
        latent_dim: int = 100,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
        loss_type: str = "non-saturating",
        ckpt_path: str = "",
    ):
        super(GAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
        )
        self.discriminator = Discriminator(
            img_channels=img_channels,
            img_size=img_size,
        )

        if os.path.exists(ckpt_path):
            self.load_from_checkpoint(ckpt_path)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, _ = batch
        d_optim, g_optim = self.optimizers()

        # Train Discriminator
        if self.global_step % 2 == 0:
            loss_dict = self._calculate_d_loss(x)
            d_optim.zero_grad()
            self.manual_backward(loss_dict["d_loss"])
            d_optim.step()

        # Train Generator
        else:
            loss_dict = self._calculate_g_loss(x.size(0))
            g_optim.zero_grad()
            self.manual_backward(loss_dict["g_loss"])
            g_optim.step()

        self.log_dict(loss_dict, on_step=True, prog_bar=True)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, _ = batch
        loss_dict = self._calculate_g_loss(x.size(0))
        self.log("val_loss", loss_dict["g_loss"], on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self._log_images(fig_name="Random Generation", batch_size=16)

    def configure_optimizers(self):
        d_optim = Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        g_optim = Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        return [d_optim, g_optim], []

    def _calculate_d_loss(self, x: Tensor) -> Tensor:
        """
        Calculate the discriminator loss.

        The discriminator loss consists of two parts:
        1. Loss for real images: Should be classified as 1 (or real).
        2. Loss for fake images: Should be classified as 0 (or fake).

        The total discriminator loss is the sum of the above two losses.
        """
        logits_real = self.discriminator(x)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

        x_hat = self.generator.random_sample(x.size(0))
        logits_fake = self.discriminator(x_hat.detach())
        d_loss_fake = bce_with_logits(logits_fake, torch.zeros_like(logits_fake))

        d_loss = (d_loss_real + d_loss_fake) / 2

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict

    def _calculate_g_loss(self, batch_size: int) -> Tensor:
        """
        Calculate the generator's loss.

        The generator aims to produce fake data that the discriminator classifies as real.
        This method generates fake data, passes it through the discriminator, and computes
        the loss based on how well the generator fooled the discriminator.
        """
        x_hat = self.generator.random_sample(batch_size)
        logits_fake = self.discriminator(x_hat)

        if self.hparams.loss_type == "min-max":
            # Original generator loss based on min-max game with value function V(G,D).
            g_loss = -bce_with_logits(logits_fake, torch.zeros_like(logits_fake))

        elif self.hparams.loss_type == "non-saturating":
            # Non-saturating loss address the vanishing gradients problem in GANs,
            # especially early in the training when the generator is poor.
            g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))

        loss_dict = {
            "g_loss": g_loss,
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def _log_images(self, fig_name: str, batch_size: int):
        sample_images = self.generator.random_sample(batch_size=batch_size)
        sample_images = ((sample_images + 1.0) / 2.0) * 255.0
        sample_images = sample_images.clamp(0, 255).byte().detach().cpu()
        fig = make_grid(sample_images)
        self.logger.experiment.log(
            {fig_name: [wandb.Image(fig)]}, step=self.global_step
        )
