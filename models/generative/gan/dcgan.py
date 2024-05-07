"""
DCGAN (Deep Convolutional Generative Adversarial Networks) Architecture:

DCGANs are a type of GAN that use convolutional and convolutional-transpose layers
in the discriminator and generator, respectively. They introduced several architectural
guidelines to stabilize the training of GANs, resulting in higher quality image generation.

Reference: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torch.nn as nn
from torch import List, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from models.generative.gan.gan import GAN


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
    ) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        if img_size == 64:
            self.model = nn.Sequential(
                self._block(latent_dim, 1024, 4, 1, 0),
                self._block(1024, 512, 4, 2, 1),
                self._block(512, 256, 4, 2, 1),
                self._block(256, 128, 4, 2, 1),
                self._block(128, img_channels, 4, 2, 1, final_layer=True),
            )

        elif img_size == 28:
            # MNIST
            self.model = nn.Sequential(
                self._block(latent_dim, 256, 7, 1, 0),
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
        z = torch.randn(
            [batch_size, self.latent_dim, 1, 1],
            device=self.device,
        )
        return self(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
    ) -> None:
        super(Discriminator, self).__init__()

        if img_size == 64:
            self.model = nn.Sequential(
                self._block(img_channels, 64, 4, 2, 1, use_bn=False),
                self._block(64, 128, 4, 2, 1, use_bn=True),
                self._block(128, 256, 4, 2, 1, use_bn=True),
                self._block(256, 512, 4, 2, 1, use_bn=True),
                self._block(512, 1, 4, 1, 0, use_bn=False, final_layer=True),
            )

        elif img_size == 28:
            self.model = nn.Sequential(
                self._block(img_channels, 64, 4, 2, 1, use_bn=False),
                self._block(64, 128, 4, 2, 1, use_bn=True),
                self._block(128, 256, 7, 1, 0),
                self._block(256, 1, 1, 1, 0, use_bn=False, final_layer=True),
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


class DCGAN(GAN):
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
        calculate_metrics: bool = False,
        metrics: List[str] = [],
        summary: bool = True,
    ) -> None:
        super(DCGAN, self).__init__(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
            lr=lr,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            calculate_metrics=calculate_metrics,
            metrics=metrics,
            summary=False,
        )

        self.G = Generator(
            img_size=img_size,
            img_channels=img_channels,
            latent_dim=latent_dim,
        )
        self.D = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
        )

        if self.metrics:
            self.fid = FrechetInceptionDistance() if "fid" in self.metrics else None
            self.kid = KernelInceptionDistance(subset_size=100) if "kid" in self.metrics else None
            self.inception_score = InceptionScore() if "is" in self.metrics else None

        self.z = torch.randn([16, latent_dim, 1, 1])
        if summary:
            self.summary()

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
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

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        logits_fake = self.D(x_hat)
        g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))
        loss_dict = {"g_loss": g_loss}
        return loss_dict
