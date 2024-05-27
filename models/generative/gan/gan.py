from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid


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
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, np.prod(self.img_shape)),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z).view(-1, *self.img_shape)

    def random_sample(self, batch_size: int) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
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
        calculate_metrics: bool = False,
        metrics: List[str] = [],
        summary: bool = True
    ):
        super(GAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.calculate_metrics = calculate_metrics
        self.metrics = metrics

        self.G = Generator(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
        )
        self.D = Discriminator(
            img_channels=img_channels,
            img_size=img_size,
        )

        if self.metrics:
            self.fid = FrechetInceptionDistance() if "fid" in self.metrics else None
            self.kid = KernelInceptionDistance(subset_size=100) if "kid" in self.metrics else None
            self.inception_score = InceptionScore() if "is" in self.metrics else None

        self.z = torch.randn([64, latent_dim])
        if summary:
            self.summary()

    def forward(self, z: Tensor) -> Tensor:
        return self.G(z)

    def _common_step(
        self,
        batch: Tuple[Tensor, Tensor],
        mode: str,
    ) -> None:
        x, _ = batch
        x_hat = self.G.random_sample(x.size(0))
        d_optim, g_optim = self.optimizers()

        # Train Discriminator
        loss_dict = self._calculate_d_loss(x, x_hat)
        if self.training:
            d_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["d_loss"])
            d_optim.step()

        # Train Generator
        loss_dict.update(self._calculate_g_loss(x_hat))
        if self.training:
            g_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["g_loss"])
            g_optim.step()

        loss_dict = {f"{mode}_{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        return x, x_hat, loss_dict

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        _, _, loss_dict = self._common_step(batch, "train")
        return loss_dict

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, x_hat, _ = self._common_step(batch, "val")

        if self.calculate_metrics:
            self.update_metrics(x, x_hat)

    def on_validation_epoch_end(self):
        self._log_images(fig_name="Random Generation", batch_size=16)

        if self.calculate_metrics:
            metrics = self.compute_metrics()

            self.log_dict(
                metrics,
                prog_bar=True,
                logger=True,
                sync_dist=torch.cuda.device_count() > 1,
            )

            self.fid.reset() if "fid" in self.metrics else None
            self.kid.reset() if "kid" in self.metrics else None
            self.inception_score = InceptionScore().to(self.device) if "is" in self.metrics else None

    def update_metrics(self, x, x_hat):
        # Update metrics with real and generated images
        x = (
            x
            .add_(1.0)
            .mul_(127.5)
            .byte()
        )
        x_hat = (
            x_hat
            .add_(1.0)
            .mul_(127.5)
            .byte()
        )

        if "fid" in self.metrics:
            self.fid.update(x, real=True)
            self.fid.update(x_hat, real=False)

        if "kid" in self.metrics:
            self.kid.update(x, real=True)
            self.kid.update(x_hat, real=False)

        if "is" in self.metrics:
            self.inception_score.update(x_hat)

    def compute_metrics(self) -> Dict[str, Tensor]:
        fid_score = self.fid.compute() if "fid" in self.metrics else None
        kid_mean, kid_std = self.kid.compute() if "kid" in self.metrics else None, None
        is_mean, is_std = self.inception_score.compute() if "is" in self.metrics else None, None

        metrics = {
            "fid_score": fid_score,
            "mean_kid_score": kid_mean,
            "std_kid_score": kid_std,
            "mean_inception_score": is_mean,
            "std_inception_score": is_std,
        }
        return metrics

    def configure_optimizers(self):
        d_optim = Adam(
            self.D.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        g_optim = Adam(
            self.G.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        return [d_optim, g_optim], []

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Dict:
        """
        Calculate the discriminator loss.

        The discriminator loss consists of two parts:
        1. Loss for real images: Should be classified as 1 (or real).
        2. Loss for fake images: Should be classified as 0 (or fake).

        The total discriminator loss is the sum of the above two losses.
        """
        logits_real = self.D(x)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

        logits_fake = self.D(x_hat.detach())
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

    def _calculate_g_loss(self, x_hat: Tensor) -> Dict:
        """
        Calculate the generator's loss.

        The generator aims to produce fake data that the discriminator classifies as real.
        This method generates fake data, passes it through the discriminator, and computes
        the loss based on how well the generator fooled the discriminator.
        """
        logits_fake = self.D(x_hat)

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
        sample_images = self.G(self.z.to(self.device))
        fig = make_grid(
            tensor=sample_images,
            value_range=(-1, 1),
            normalize=True,
        )
        self.logger.experiment.log(
            {fig_name: [wandb.Image(fig)]},
            step=self.global_step,
        )

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
            self.G,
            input_data=self.z,
            col_names=col_names,
        )

        summary(
            self.D,
            input_data=x,
            col_names=col_names,
        )
