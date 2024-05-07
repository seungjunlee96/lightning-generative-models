from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid


class Generator(nn.Module):
    """
    Generator for the Conditional Generative Adversarial Network (CGAN).

    This model generates fake images given a noise vector `z` and class labels `c`.
    """

    def __init__(
        self,
        img_channels: int,
        latent_dim: int,
        num_classes: int,
    ) -> None:
        """
        Args:
            img_channels (int): Number of channels in the output image.
            latent_dim (int): Dimensionality of the latent noise vector.
            num_classes (int): Number of classes for conditional generation.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Initial transformation: Transforms the concatenated noise and class label into a feature map
        self.initial = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 7 * 7 * 256),
            nn.LeakyReLU(0.2),
        )

        # Deconvolution layers: Upsample the feature map to produce the fake image
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, img_channels, 3, 2, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass for the generator.

        Args:
            z (Tensor): Latent noise vector of shape (batch_size, latent_dim).
            c (Tensor): One-hot encoded class labels of shape (batch_size, num_classes).

        Returns:
            Tensor: Generated fake images.
        """
        x = torch.cat([z, c], dim=1)
        x = self.initial(x)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator for the Conditional Generative Adversarial Network (CGAN).
    This model differentiates between real and fake images given the image and class label.
    """

    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        """
        Args:
            img_channels (int): Number of channels in the input image.
            num_classes (int): Number of classes for conditional discrimination.
        """
        super().__init__()

        # Convolution layers: Process the concatenated image and class label to produce a decision
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, 64, 3, 2, 1),  # [64, 14, 14]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),  # [128, 7, 7]
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # [128 * 7, 7]
            nn.Dropout(dropout),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass for the discriminator.

        Args:
            x (Tensor): Real or fake images.
            c (Tensor): One-hot encoded class labels.

        Returns:
            Tensor: Decision scores for the images. Higher values indicate "real", lower values indicate "fake".
        """
        c = self._expand_label(c, x.size())
        x = self.model(torch.cat([x, c], dim=1))
        return x

    def _expand_label(self, c: Tensor, size: torch.Size) -> Tensor:
        """
        Expand label tensor to match the size of the input image tensor.

        Args:
            c (Tensor): One-hot encoded class labels.
            size (torch.Size): Size of the input image tensor.

        Returns:
            Tensor: Expanded class labels.
        """
        c = c.view(-1, c.size(1), 1, 1)
        return c.expand(size[0], c.size(1), size[2], size[3])


class CGAN(pl.LightningModule):
    """
    Conditional Generative Adversarial Network (CGAN) using PyTorch Lightning.

    This model consists of a generator and a discriminator that are trained alternately.
    """

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 100,
        img_channels: int = 1,
        img_size: int = 28,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
    ) -> None:
        """
        Args:
            num_classes (int): Number of classes for conditional GAN.
            latent_dim (int): Dimensionality of the latent noise vector.
            img_channels (int): Number of channels in the image.
            img_size (int): Size of the image (assumed square).
            lr (float): Learning rate for the Adam optimizers.
            b1 (float): Beta1 coefficient for the Adam optimizers.
            b2 (float): Beta2 coefficient for the Adam optimizers.
            weight_decay (float): Weight decay for the Adam optimizers.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.G = Generator(
            img_channels=img_channels,
            latent_dim=latent_dim,
            num_classes=num_classes,
        )
        self.D = Discriminator(
            img_channels=img_channels,
            num_classes=num_classes,
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.z = torch.randn([num_classes, latent_dim])
        self.summary()

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass using the generator.

        Args:
            z (Tensor): Latent noise vector.
            c (Tensor): One-hot encoded class labels.

        Returns:
            Tensor: Generated fake images.
        """
        return self.G(z, c)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """
        Training step for the CGAN. This includes one step of training the discriminator and one step of training the generator.

        Args:
            batch (Tuple[Tensor, Tensor]): Input data batch. The first tensor is the real images, and the second tensor is the class labels.

        Returns:
            Dict[str, Tensor]: Dictionary of loss values and other relevant metrics.
        """
        x, c = batch
        c = F.one_hot(c, num_classes=self.num_classes).float()
        z = torch.randn(x.size(0), self.latent_dim).to(self.device)
        x_hat = self.G(z, c)

        d_optimizer, g_optimizer = self.optimizers()

        # Train Discriminator
        loss_dict = self._calculate_d_loss(x, x_hat, c)
        d_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss_dict["d_loss"])
        d_optimizer.step()

        # Train Generator
        loss_dict.update(self._calculate_g_loss(x_hat, c))
        g_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss_dict["g_loss"])
        g_optimizer.step()

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        return loss_dict

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        Validation step for the CGAN. This involves generating images and computing the generator loss.

        Args:
            batch (Tuple[Tensor, Tensor]): Input data batch for validation.
            batch_idx (int): Batch index.
        """
        x, c = batch
        c = F.one_hot(c, num_classes=self.num_classes).float()
        z = torch.randn(x.size(0), self.latent_dim).to(self.device)
        x_hat = self.G(z, c)

        loss_dict = self._calculate_d_loss(x, x_hat, c)
        loss_dict.update(self._calculate_g_loss(x_hat, c))

        self.log(
            "val_loss",
            loss_dict["g_loss"],
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        if batch_idx == 0:
            self._log_images(fig_name="Random Generation")

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List]:
        """
        Configure the optimizers for the generator and the discriminator.

        Returns:
            Tuple[List[torch.optim.Optimizer], List]: A tuple containing the list of optimizers and an empty list (since we don't use learning rate schedulers here).
        """
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

    def _calculate_d_loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        c: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Calculate the discriminator's loss.

        Args:
            x (Tensor): Real images.
            c (Tensor): One-hot encoded class labels.

        Returns:
            Dict[str, Tensor]: Dictionary of loss values and other relevant metrics.
        """
        logits_real = self.D(x, c)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

        logits_fake = self.D(x_hat.detach(), c)
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

    def _calculate_g_loss(
        self,
        x_hat: Tensor,
        c: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Calculate the generator's loss.

        Args:
            x (Tensor): Real images (not used in the generator loss but provided for consistency).
            c (Tensor): One-hot encoded class labels.

        Returns:
            Dict[str, Tensor]: Dictionary of loss values and other relevant metrics.
        """
        # Generate fake images
        logits_fake = self.D(x_hat, c)
        g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))

        loss_dict = {
            "g_loss": g_loss,
        }
        return loss_dict

    @torch.no_grad()
    def _log_images(self, fig_name: str) -> None:
        """
        Log generated images to Weights & Biases (wandb) for visualization.

        Args:
            fig_name (str): Name of the figure to be displayed in wandb.
            batch_size (int): Number of images to generate and log.
        """
        # Generate fake image
        z = self.z.to(self.device)
        c = F.one_hot(
            torch.arange(0, self.num_classes, device=self.device),
            num_classes=self.num_classes,
        ).float()
        sample = self.G(z, c)

        # log sample images
        fig = make_grid(
            tensor=sample,
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
        z = torch.randn([self.num_classes, self.latent_dim])
        c = F.one_hot(
            torch.arange(0, self.num_classes),
            num_classes=self.num_classes,
        ).float()

        x = torch.randn(
            [
                self.num_classes,
                self.hparams.img_channels,
                self.hparams.img_size,
                self.hparams.img_size,
            ]
        )

        summary(
            self.G,
            input_data=[z, c],
            col_names=col_names,
        )

        summary(
            self.D,
            input_data=[x, c],
            col_names=col_names,
        )
