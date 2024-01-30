"""
DCGAN (Deep Convolutional Generative Adversarial Networks) Architecture:

DCGANs are a type of GAN that use convolutional and convolutional-transpose layers
in the discriminator and generator, respectively. They introduced several architectural
guidelines to stabilize the training of GANs, resulting in higher quality image generation.

Reference: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import os
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import List, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam
from torchinfo import summary
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from utils.visualization import make_grid


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    return model


class Generator(nn.Module):
    def __init__(
        self,
        img_channels: int,
        latent_dim: int,
    ) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            self._block(latent_dim, 1024, 4, 1, 0),
            self._block(1024, 512, 4, 2, 1),
            self._block(512, 256, 4, 2, 1),
            self._block(256, 128, 4, 2, 1),
            self._block(128, img_channels, 4, 2, 1, final_layer=True),
        )
        self.model = initialize_weights(self.model)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        final_layer: bool = False,
    ) -> nn.Sequential:
        """
        Returns a block for the generator containing
            a fractional-strided convolution,
            batch normalization,
            and a ReLU activation for all layers except for the output, which uses the Tanh activation.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels) if not final_layer else nn.Identity(),
            nn.ReLU(inplace=True) if not final_layer else nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z)

    def random_sample(self, batch_size: int) -> Tensor:
        z = torch.randn([batch_size, self.latent_dim, 1, 1], device=self.device)
        return self(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class Discriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
    ) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(img_channels, 64, 4, 2, 1, use_bn=False),
            self._block(64, 128, 4, 2, 1, use_bn=True),
            self._block(128, 256, 4, 2, 1, use_bn=True),
            self._block(256, 512, 4, 2, 1, use_bn=True),
            self._block(512, 1, 4, 1, 0, use_bn=False, final_layer=True),
        )
        self.model = initialize_weights(self.model)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_bn: bool = True,
        final_layer: bool = False,
    ) -> nn.Sequential:
        """
        Returns a block for the discriminator containing
        - a strided convolution,
        - optional batch normalization,
        - and a LeakyReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if not final_layer else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze()


class DCGAN(pl.LightningModule):
    """
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications.
    Comparatively, unsupervised learning with CNNs has received less attention.
    In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning.
    We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning.
    Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator.
    Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
        lr: float,
        b1: float,
        b2: float,
        weight_decay: float,
        ckpt_path: str = "",
        calculate_metrics: bool = False,
    ) -> None:
        super(DCGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.calculate_metrics = calculate_metrics

        self.generator = Generator(img_channels=img_channels, latent_dim=latent_dim)
        self.discriminator = Discriminator(img_channels=img_channels)

        self.fid = FrechetInceptionDistance()
        self.kid = KernelInceptionDistance(subset_size=100)
        self.inception_score = InceptionScore()

        if os.path.exists(ckpt_path):
            self.load_from_checkpoint(ckpt_path)

        self.fixed_z = torch.randn([16, latent_dim, 1, 1])
        self.summary()

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, _ = batch
        x_hat = self.generator.random_sample(x.size(0))
        d_optim, g_optim = self.optimizers()

        # Train Discriminator
        if self.global_step % 2 == 0:
            loss_dict = self._calculate_d_loss(x, x_hat)
            d_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["d_loss"])
            d_optim.step()

        # Train Generator
        else:
            loss_dict = self._calculate_g_loss(x_hat)
            g_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["g_loss"])
            g_optim.step()

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, _ = batch
        x_hat = self.generator.random_sample(x.size(0))

        loss_dict = self._calculate_d_loss(x, x_hat)
        loss_dict.update(self._calculate_g_loss(x_hat))

        self.log(
            "val_loss",
            loss_dict["g_loss"],
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        self._log_images(fig_name="Random Generation", batch_size=16)
        if self.calculate_metrics:
            self.update_metrics(x, x_hat)

    def on_validation_epoch_end(self):
        metrics = self.compute_metrics()

        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

        self.fid.reset()
        self.kid.reset()
        self.inception_score = InceptionScore().to(self.device)

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

        self.fid.update(x, real=True)
        self.fid.update(x_hat, real=False)

        self.kid.update(x, real=True)
        self.kid.update(x_hat, real=False)

        self.inception_score.update(x_hat)

    def compute_metrics(self):
        fid_score = self.fid.compute()
        kid_mean, kid_std = self.kid.compute()
        is_mean, is_std = self.inception_score.compute()

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
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        g_optim = Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return [d_optim, g_optim], []

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        logits_real = self.discriminator(x)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

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

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        logits_fake = self.discriminator(x_hat)
        g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))

        loss_dict = {"g_loss": g_loss}
        return loss_dict

    @torch.no_grad()
    def _log_images(self, fig_name: str, batch_size: int):
        sample_images = self.generator(self.fixed_z.to(self.device))
        fig = make_grid(
            sample_images
            .add_(1.0)
            .mul_(127.5)
            .byte()
            .detach()
            .cpu()
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
        z = torch.randn([1, self.hparams.latent_dim, 1, 1])
        x = torch.randn(
            [
                1,
                self.hparams.img_channels,
                self.hparams.img_size,
                self.hparams.img_size,
            ]
        )

        summary(
            self.generator,
            input_data=z,
            col_names=col_names,
        )

        summary(
            self.discriminator,
            input_data=x,
            col_names=col_names,
        )
