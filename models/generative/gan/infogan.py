from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam
from torchinfo import summary
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import make_grid

from utils.loss_functions import GaussianNLL


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
        img_size: int,
        img_channels: int,
        latent_dim: int,
        categorical_code_dim: int,
        continuous_code_dim: int,
    ) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_code_dim = categorical_code_dim
        self.continuous_code_dim = continuous_code_dim
        code_dim = categorical_code_dim + continuous_code_dim

        if img_size == 64:
            # CelebA
            self.model = nn.Sequential(
                self._block(latent_dim + code_dim, 1024, 4, 1, 0),
                self._block(1024, 512, 4, 2, 1),
                self._block(512, 256, 4, 2, 1),
                self._block(256, 128, 4, 2, 1),
                self._block(128, img_channels, 4, 2, 1, final_layer=True),
            )
        elif img_size == 28:
            # MNIST
            self.model = nn.Sequential(
                self._block(latent_dim + code_dim, 256, 7, 1, 0),
                self._block(256, 128, 4, 2, 1),
                self._block(128, img_channels, 4, 2, 1, final_layer=True),
            )

        else:
            raise NotImplementedError

        # Initialize weights
        self.apply(initialize_weights)

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

    def forward(
        self,
        z: Tensor,
        categorical_c: Tensor,
        continuous_c: Tensor,
    ) -> Tensor:
        input = torch.cat(
            [z, categorical_c, continuous_c],
            dim=1,
        ).unsqueeze(-1).unsqueeze(-1).to(self.device)
        return self.model(input)

    def generate_codes(
        self,
        batch_size: int = 1,
        transition: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        z = torch.randn([batch_size, self.latent_dim], device=self.device)

        if transition:
            assert batch_size % self.categorical_code_dim == 0

            # Categorical codes: Use torch.arange for a simple transition by stepping through categories
            if batch_size < self.categorical_code_dim:
                categories = torch.arange(0, self.categorical_code_dim, device=self.device)
                categorical_c = F.one_hot(categories, num_classes=self.categorical_code_dim).float()
                categorical_c = categorical_c.repeat((batch_size // self.categorical_code_dim) + 1, 1)
            else:
                step_size = batch_size // self.categorical_code_dim
                categories = torch.arange(0, self.categorical_code_dim, device=self.device).repeat_interleave(step_size)
                categorical_c = F.one_hot(categories, num_classes=self.categorical_code_dim).float()

            # Continuous codes: Linear interpolation (torch.linspace is still more suitable here)
            start = torch.rand(1, self.continuous_code_dim, device=self.device)
            end = torch.rand(1, self.continuous_code_dim, device=self.device)
            alpha = torch.linspace(0, 1, steps=batch_size, device=self.device).view(-1, 1)
            continuous_c = start * (1 - alpha) + end * alpha

        else:
            # Generate random continuous codes
            random_categorical = torch.randint(0, self.categorical_code_dim, (batch_size,), device=self.device)
            categorical_c = F.one_hot(random_categorical, num_classes=self.categorical_code_dim).float()

            # Generate random categorical codes
            continuous_c = torch.rand(batch_size, self.continuous_code_dim, device=self.device)

        return z, categorical_c, continuous_c

    def random_sample(self, batch_size: int) -> Tensor:
        z, categorical_c, continuous_c = self.generate_codes(batch_size=batch_size)
        x_hat = self(z, categorical_c, continuous_c)
        return x_hat

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        categorical_code_dim: int,
        continuous_code_dim: int,
    ):
        super(Discriminator, self).__init__()
        self.categorical_code_dim = categorical_code_dim
        self.continuous_code_dim = continuous_code_dim

        if img_size == 64:
            self.feature_extractor = nn.Sequential(
                self._block(img_channels, 64, 4, 2, 1, use_bn=False),
                self._block(64, 128, 4, 2, 1, use_bn=True),
                self._block(128, 256, 4, 2, 1, use_bn=True),
                self._block(256, 512, 4, 2, 1, use_bn=True),
            )
            feature_dim = 512

        elif img_size == 28:
            self.feature_extractor = nn.Sequential(
                self._block(img_channels, 64, 4, 2, 1, use_bn=False),
                self._block(64, 128, 4, 2, 1, use_bn=True),
                self._block(128, 256, 7, 1, 0),
            )
            feature_dim = 256

        else:
            raise NotImplementedError

        # Final layer for real vs fake classification
        self.final_layer = nn.Linear(feature_dim, 1)

        # Auxiliary layer for Q network to predict latent codes
        self.q_network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, categorical_code_dim + 2 * continuous_code_dim,),
        )

        # Initialize weights
        self.apply(initialize_weights)

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.feature_extractor(x)
        b, c, h, w = features.size()
        features_flat = features.view(b, c, h * w).mean(-1)
        real_fake_output = self.final_layer(features_flat)
        code_pred = self.q_network(features_flat)
        Q_cat_logits, Q_cont_mu, Q_cont_logvar = torch.split(
            code_pred,
            [self.categorical_code_dim, self.continuous_code_dim, self.continuous_code_dim],
            dim=1,
        )
        return real_fake_output, Q_cat_logits, Q_cont_mu, Q_cont_logvar


class InfoGAN(pl.LightningModule):
    """
    InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
    - Paper: https://arxiv.org/abs/1606.03657
    - Official Code: https://github.com/openai/InfoGAN
    - InfoGAN is an extension of Generative Adversarial Networks (GANs) focusing on information theory.
    - It learns disentangled representations in a completely unsupervised manner.
    - Maximizes mutual information between a subset of latent variables and observations.
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 100,
        categorical_code_dim: int = 10,
        continuous_code_dim: int = 2,
        lambda_cat: float = 1,
        lambda_cont: float = 0.1,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.99,
        weight_decay: float = 1e-5,
        calculate_metrics: bool = False,
        metrics: List[str] = [],
    ) -> None:
        super(InfoGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.calculate_metrics = calculate_metrics
        self.metrics = metrics

        self.G = Generator(
            img_size=img_size,
            img_channels=img_channels,
            latent_dim=latent_dim,
            categorical_code_dim=categorical_code_dim,
            continuous_code_dim=continuous_code_dim,
        )
        self.D = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
            categorical_code_dim=categorical_code_dim,
            continuous_code_dim=continuous_code_dim,
        )

        self.loss_Q_cat = nn.CrossEntropyLoss()
        self.loss_Q_cont = GaussianNLL()

        if self.metrics:
            self.init_metrics(reset=False)

        (
            self.z,
            self.categorical_c,
            self.continuous_c
        ) = self.G.generate_codes(batch_size=100, transition=True)
        self.summary()

    def forward(
        self,
        z: Tensor,
        categorical_c: Tensor,
        continuous_c: Tensor,
    ) -> Tensor:
        return self.G(z, categorical_c, continuous_c)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, _ = batch
        batch_size = x.size(0)
        z, categorical_c, continuous_c = self.G.generate_codes(batch_size)
        x_hat = self.G(z, categorical_c, continuous_c)
        d_optim, g_optim, q_optim = self.optimizers()

        # Train Discriminator
        loss_dict = self._calculate_d_loss(x, x_hat)
        d_optim.zero_grad(set_to_none=True)
        self.manual_backward(loss_dict["d_loss"])
        d_optim.step()

        # Train Generator
        loss_dict.update(self._calculate_g_loss(x_hat))
        g_optim.zero_grad(set_to_none=True)
        self.manual_backward(loss_dict["g_loss"])
        g_optim.step()

        # Train Q
        loss_dict.update(self._calculate_mi_loss(x_hat, categorical_c, continuous_c))
        q_optim.zero_grad(set_to_none=True)
        self.manual_backward(loss_dict["mi_loss"])
        q_optim.step()

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, _ = batch
        batch_size = x.size(0)
        z, categorical_c, continuous_c = self.G.generate_codes(batch_size)
        x_hat = self.G(z, categorical_c, continuous_c)

        loss_dict = self._calculate_d_loss(x, x_hat)
        loss_dict.update(self._calculate_g_loss(x_hat))
        loss_dict.update(self._calculate_mi_loss(x_hat, categorical_c, continuous_c))

        self.log(
            "val_loss",
            loss_dict["g_loss"],
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        self._log_images(fig_name="Random Generation")
        if self.calculate_metrics:
            self.update_metrics(x, x_hat)

    def on_validation_epoch_end(self):
        if self.metrics:
            metrics = self.compute_metrics()

            self.log_dict(
                metrics,
                prog_bar=True,
                logger=True,
                sync_dist=torch.cuda.device_count() > 1,
            )

            self.init_metrics(reset=True)

    def init_metrics(self, reset: bool = False):
        if reset:
            self.fid.reset() if "fid" in self.metrics else None
            self.kid.reset() if "kid" in self.metrics else None
            self.inception_score = InceptionScore().to(self.device) if "is" in self.metrics else None

        else:
            self.fid = FrechetInceptionDistance() if "fid" in self.metrics else None
            self.kid = KernelInceptionDistance(subset_size=100) if "kid" in self.metrics else None
            self.inception_score = InceptionScore() if "is" in self.metrics else None

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
        q_optim = Adam(
            list(self.D.parameters()) + list(self.G.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )

        return [d_optim, g_optim, q_optim], []

    def _calculate_d_loss(
        self,
        x: Tensor,
        x_hat: Tensor,
    ) -> Dict[str, Tensor]:
        logits_real, _, _, _ = self.D(x)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

        logits_fake, _, _, _ = self.D(x_hat.detach())
        d_loss_fake = bce_with_logits(logits_fake, torch.zeros_like(logits_fake))

        # Combine the losses
        d_loss = (d_loss_real + d_loss_fake) / 2

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict

    def _calculate_g_loss(self, x_hat: Tensor) -> Dict[str, Tensor]:
        logits_fake, _, _, _ = self.D(x_hat)
        g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))

        loss_dict = {"g_loss": g_loss}
        return loss_dict

    def _calculate_mi_loss(self, x_hat: Tensor, categorical_c: Tensor, continuous_c: Tensor) -> Dict[str, Tensor]:
        _, Q_fake_logits, Q_fake_cont_mu, Q_fake_cont_logvar = self.D(x_hat)
        loss_categorical = self.loss_Q_cat(Q_fake_logits, categorical_c)
        loss_continuous = self.loss_Q_cont(continuous_c, Q_fake_cont_mu, Q_fake_cont_logvar)

        mi_loss = (
            self.hparams.lambda_cat * loss_categorical
            + self.hparams.lambda_cont * loss_continuous
        )

        loss_dict = {
            "mi_loss": mi_loss,
            "mi_categorical": loss_categorical,
            "mi_continuous": loss_continuous,
        }

        return loss_dict

    @torch.no_grad()
    def _log_images(self, fig_name: str):
        sample_images = self.G(self.z, self.categorical_c, self.continuous_c)
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
            input_data=(self.z, self.categorical_c, self.continuous_c),
            col_names=col_names,
        )
        summary(
            self.D,
            input_data=x,
            col_names=col_names,
        )
